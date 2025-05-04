import io
import os
import random
import json
from time import perf_counter
import logging
from urllib.parse import urlparse
from uuid import uuid4
import argparse
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Tuple, List, Protocol, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict

from pydantic import BaseModel, Field
import polars as pl
import numpy as np
import duckdb
from boto3 import client as boto_client
from gcsfs import GCSFileSystem
from sqlalchemy import create_engine
from faker import Faker
from dateutil.relativedelta import relativedelta
import math

logger = logging.getLogger("retail_gen")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

class GeoClusterSpec(BaseModel):
    name: str
    lat: float
    lon: float
    radius_km: int
    num_stores: int
    weight: float
    channel_weights: Dict[str, float] = Field(default_factory=lambda: {"online": 0.5, "in_store": 0.5})
    income_dist: Dict[str, float] = Field(default_factory=lambda: {"low": 0.3, "medium": 0.5, "high": 0.2})
    age_range: List[int] = Field(default_factory=lambda: [18, 65])

@dataclass
class RetailDataSpec:
    num_customers: int = 100
    num_products: int = 50
    num_transactions: int = 500
    reg_date_start: str = '-2y'
    data_end: str = 'today'
    seed: Optional[int] = None
    destination: str = field(default_factory=lambda: os.getenv('RETAIL_DESTINATION', './data'))
    parquet_compression: Optional[str] = None

    # Inventory & supply
    suppliers_per_product: int = 2
    return_prob: float = 0.05
    disruption_prob: float = 0.1
    disruption_duration_days: Tuple[int, int] = (3, 10)
    initial_stock_range: Tuple[int, int] = (20, 200)
    reorder_point: int = 10
    reorder_quantity: int = 100
    reorder_lead_time_days: Tuple[int, int] = (1, 5)
    
    # Product modeling
    product_categories: List[str] = field(default_factory=lambda: ['grocery', 'beverage', 'electronics', 'apparel', 'personal_care'])
    product_seasonality: List[str] = field(default_factory=lambda: ['none', 'summer', 'winter'])
    supplier_pool_size: int = 10

    # Geography & customer distribution
    geo_clusters: List[GeoClusterSpec] = field(default_factory=list)

    def __post_init__(self):
        assert self.num_customers > 0
        assert self.num_products > 0
        assert self.num_transactions > 0
        assert self.geo_clusters, "❌ You must define at least one geo_cluster"

# ---------------------------
# Utility functions
# ---------------------------

def random_date_weighted(start_date, end_date):
    """Generate random weight-based dates"""
    weights_by_month = {
        1: 0.5, 2: 0.8, 3: 1.0, 12: 2.0,  # dezembro pesa mais
    }
    delta = (end_date - start_date).days
    while True:
        offset = random.randint(0, delta)
        date = start_date + timedelta(days=offset)
        weight = weights_by_month.get(date.month, 1.0)
        if random.random() < weight / max(weights_by_month.values()):
            return date

def random_geo_around(lat, lon, radius_km=50):
    """Generate random point around given lat/lon within radius_km."""
    # 1 degree ~= 111 km
    radius_deg = radius_km / 111
    delta_lat = random.uniform(-radius_deg, radius_deg)
    delta_lon = random.uniform(-radius_deg, radius_deg)
    return round(lat + delta_lat, 6), round(lon + delta_lon, 6)

def parse_date(rel: str) -> date:
    if rel == 'today': return date.today()
    if rel.endswith('y'):
        return date.today() + relativedelta(years=int(rel[:-1]))
    if rel.endswith('d'):
        return date.today() + timedelta(days=int(rel[:-1]))
    raise ValueError("Invalid date str")

def haversine(lat1, lon1, lat2, lon2):
    R=6371; dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.asin(math.sqrt(a))

def calculate_distance(cust_lat, cust_lon, store_lat, store_lon):
    return haversine(cust_lat, cust_lon, store_lat, store_lon)

def create_transaction(cust, pid, sid, qty, price, txn_date, dist):
    return {
        'customer_id': cust['customer_id'],
        'product_id': pid,
        'store_id': sid,
        'quantity': qty,
        'total_price': price,
        'transaction_date': txn_date,
        'distance_km': round(dist, 2),
    }

def apply_returns(txn, txn_date, return_prob):
    if random.random() < return_prob:
        txn['returned'] = True
        txn['return_date'] = txn_date + timedelta(days=random.randint(1, 30))
    else:
        txn['returned'] = False
        txn['return_date'] = None
    return txn

def generate_geo_clusters(clusterization_type: str) -> List[GeoClusterSpec]:
    if clusterization_type not in CLUSTER_DATA:
        raise ValueError(f"Unknown clusterization type: {clusterization_type}")
    
    clusters = []
    for cluster_info in CLUSTER_DATA[clusterization_type]:
        clusters.append(GeoClusterSpec(**cluster_info))  # Unpack dictionary directly into GeoClusterSpec
    
    return clusters

def estimate_sla_days(from_loc, to_loc, base=500, jitter=True):
    km = haversine(from_loc, to_loc)
    days = max(1, km // base)
    return days + random.randint(-1, 2) if jitter else days


# ----------------------------
# Storage Layer
# ----------------------------
class DataSink(Protocol):
    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        ...

class CSVSink:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        output_path = self.base_path / destination if destination else self.base_path
        output_path.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            df.write_csv(output_path / f"{name}.csv")

class ParquetSink:
    def __init__(self, base_path: str, compression: Optional[str] = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        output_path = self.base_path / destination if destination else self.base_path
        output_path.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            df.write_parquet(output_path / f"{name}.parquet", compression=self.compression)

class DuckDBSink:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        schema_prefix = f"{destination}." if destination else ""
        con = duckdb.connect(self.db_path)
        for name, df in data.items():
            con.register(name, df.to_arrow())
            con.execute(f"CREATE OR REPLACE TABLE {schema_prefix}{name} AS SELECT * FROM {name}")
            con.unregister(name)
        con.close()

class JSONLSink:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        output_path = self.base_path / destination if destination else self.base_path
        output_path.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            with open(output_path / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for row in df.iter_rows(named=True):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

class SQLSink:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, future=True)

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        with self.engine.begin() as conn:
            for name, df in data.items():
                table_name = f"{destination}.{name}" if destination else name
                df.write_database(table_name=table_name, connection=conn, if_exists="replace")

class FeatherSink:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        output_path = self.base_path / destination if destination else self.base_path
        output_path.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            df.write_ipc(output_path / f"{name}.feather")

class InMemorySink:
    def __init__(self):
        self.storage = {}

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        key = destination or "default"
        self.storage[key] = data

class S3ParquetSink:
    def __init__(self, bucket: str, prefix: str = '', compression: str = "snappy"):
        self.bucket = bucket
        self.default_prefix = prefix.rstrip("/") + "/" if prefix else ''
        self.compression = compression
        self.s3 = boto_client("s3")

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None):
        prefix = destination.rstrip("/") + "/" if destination else self.default_prefix
        for name, df in data.items():
            buf = io.BytesIO()
            df.write_parquet(buf, compression=self.compression)
            key = f"{prefix}{name}.parquet"
            buf.seek(0)
            self.s3.upload_fileobj(buf, self.bucket, key)

class GCSParquetSink:
    def __init__(self, path: str, compression: str = "snappy"):
        self.default_path = path.rstrip("/") + "/" if path else ''
        self.compression = compression
        self.fs = GCSFileSystem()

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None):
        path = destination.rstrip("/") + "/" if destination else self.default_path
        for name, df in data.items():
            full_path = f"{path}{name}.parquet"
            with self.fs.open(full_path, "wb") as f:
                df.write_parquet(f, compression=self.compression)

class SQLiteSink:
    def __init__(self, db_path: str):
        self.default_db_path = db_path
        self.default_engine = create_engine(f"sqlite:///{db_path}", future=True)

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None):
        db_path = destination if destination else self.default_db_path
        engine = create_engine(f"sqlite:///{db_path}", future=True) if destination else self.default_engine
        with engine.begin() as conn:
            for name, df in data.items():
                df.write_database(table_name=name, connection=conn, if_exists="replace")

class GeoClusterManager:
    def __init__(self, geo_clusters: List[GeoClusterSpec]):
        """Initialize with a list of GeoClusterSpec instances."""
        self.geo_clusters = {cluster.name: cluster for cluster in geo_clusters}

    def assign_to_region(self, latitude: float, longitude: float) -> str:
        """Assign a point to the nearest region within range, or fallback to nearest overall."""
        closest_region = None
        closest_distance = float("inf")
        
        for region_name, cluster in self.geo_clusters.items():
            distance = haversine(latitude, longitude, cluster.lat, cluster.lon)
            if distance < closest_distance:
                closest_distance = distance
                closest_region = region_name
        
        return closest_region

class StockManager:
    def __init__(self, initial_stock_range: Tuple[int, int], lock: Lock):
        self.stock = defaultdict(lambda: random.randint(*initial_stock_range))
        self.lock = lock

    def check_and_decrement(self, store_id, product_id, qty):
        key = (store_id, product_id)
        with self.lock:
            available = self.stock[key]
            if available <= 0:
                return False  # Out of stock
            decrement_qty = min(qty, available)
            self.stock[key] -= decrement_qty
            return decrement_qty

    def get_available_stock(self, store_id, product_id):
        key = (store_id, product_id)
        return self.stock[key]

class RetailDataGenerator:
    _SINK_REGISTRY: Dict[str, Callable[[str, "RetailDataGenerator"], DataSink]] = {
        # DBs
        "duckdb": lambda path, gen: DuckDBSink(path),
        "sqlite": lambda path, gen: SQLiteSink(path),
        "postgresql": lambda uri, gen: SQLSink(uri),
        "mysql": lambda uri, gen: SQLSink(uri),

        # Local files
        "parquet": lambda path, gen: ParquetSink(path, compression=gen.spec.parquet_compression),
        "csv": lambda path, gen: CSVSink(path),
        "jsonl": lambda path, gen: JSONLSink(path),
        "feather": lambda path, gen: FeatherSink(path),

        # Cloud
        "s3": lambda uri, gen: S3ParquetSink(
            bucket=uri.split("/")[2],  # s3://bucket/key/...
            prefix="/".join(uri.split("/")[3:]),
            compression=gen.spec.parquet_compression or "snappy"
        ),
        "gcs": lambda uri, gen: GCSParquetSink(
            path=uri, compression=gen.spec.parquet_compression or "snappy"
        ),

        # Fallback
        "file": lambda path, gen: ParquetSink(path, compression=gen.spec.parquet_compression),
    }

    def __init__(self, spec: RetailDataSpec):
        self.spec = spec
        seed = spec.seed

        random.seed(seed)
        Faker.seed(seed)
        
        self.fake = Faker(seed) if seed is not None else Faker()
        self.geo_cluster_manager = GeoClusterManager(geo_clusters=spec.geo_clusters)
        self.stock_manager = StockManager(self.spec.initial_stock_range, Lock())

    def generate_customers(self) -> pl.DataFrame:
        start_date = parse_date(self.spec.reg_date_start)
        end_date = parse_date(self.spec.data_end)

        clusters: List[GeoClusterSpec] = self.spec.geo_clusters
        if not clusters:
            raise ValueError("❌ No geo_clusters defined in spec.")

        cluster_weights = [c.weight for c in clusters]
        customers = []

        for _ in range(self.spec.num_customers):
            cluster = random.choices(clusters, weights=cluster_weights)[0]

            lat, lon = random_geo_around(cluster.lat, cluster.lon, cluster.radius_km)

            registration = self.fake.date_between(start_date, end_date)

            channel = random.choices(
                population=['online', 'in_store'],
                weights=[cluster.channel_weights.get('online', 0.5),
                         cluster.channel_weights.get('in_store', 0.5)]
            )[0]

            income = random.choices(
                population=['low', 'medium', 'high'],
                weights=[cluster.income_dist.get('low', 0.3),
                         cluster.income_dist.get('medium', 0.5),
                         cluster.income_dist.get('high', 0.2)]
            )[0]

            age = random.randint(*cluster.age_range)

            customers.append({
                'customer_id': self.fake.uuid4(),
                'registration_date': registration,
                'age': age,
                'income': income,
                'channel_pref': channel,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'region': cluster.name,
            })

        logger.info(f"✅ {len(customers)} customers generated across {len(clusters)} geo clusters.")

        return pl.DataFrame(customers)

    def generate_stores(self) -> pl.DataFrame:
        stores = []
        for cluster in self.spec.geo_clusters:
            for _ in range(cluster.num_stores):
                lat, lon = random_geo_around(cluster.lat, cluster.lon, cluster.radius_km)
                stores.append({
                    'store_id': self.fake.uuid4(),
                    'latitude': round(lat, 6),
                    'longitude': round(lon, 6),
                    'region': cluster.name
                })

        logger.info(f"✅ {len(stores)} stores generated across {len(self.spec.geo_clusters)} clusters.")
        return pl.DataFrame(stores)

    def generate_suppliers(self) -> pl.DataFrame:
        suppliers_df=pl.DataFrame([
            {
                "supplier_id": f"supplier_{i}",
                "name": self.fake.company(),
                "sla_days": random.randint(1, 7)  # basic SLA per supplier
            }
            for i in range(self.spec.supplier_pool_size)
        ])

        logger.info(f"✅ {len(suppliers_df)} suppliers generated.")
        
        return suppliers_df

    def generate_products(self, suppliers: Union[List[dict], pl.DataFrame]) -> pl.DataFrame:
        categories = self.spec.product_categories
        seasonalities = self.spec.product_seasonality

        if not isinstance(suppliers, list):
            suppliers = suppliers.to_dicts()

        products = []
        for i in range(self.spec.num_products):
            category = random.choice(categories)
            seasonality = random.choice(seasonalities)
            product_id = f"prod_{i}"

            num_suppliers = max(1, min(
                self.spec.supplier_pool_size,
                np.random.poisson(lam=self.spec.suppliers_per_product)
            ))
            product_suppliers = random.sample(suppliers, k=num_suppliers)

            supplier_links = [
                {
                    "supplier_id": s["supplier_id"],
                    "sla_days": random.randint(1, 7)
                } for s in product_suppliers
            ]

            price = round(random.uniform(5.0, 200.0), 2)  # Valor entre R$5 e R$200

            products.append({
                "product_id": product_id,
                "name": self.fake.word().capitalize(),
                "category": category,
                "seasonality": seasonality,
                "price": price,
                "suppliers": supplier_links
            })

        products_df = pl.DataFrame(products)
        logger.info(f"✅ {len(products_df)} products generated.")
        return products_df

    def generate_orders_and_transactions(
        self,
        customers: pl.DataFrame,
        stores: pl.DataFrame,
        products: pl.DataFrame,
        suppliers: Dict[str, List[Dict]]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:

        custs = customers.to_dicts()
        prods = products.to_dicts()

        price_map = {
            p['product_id']: p.get('price', 1.0) or 1.0
            for p in prods
        }

        stores_map = {s['store_id']: s for s in stores.to_dicts()}
        store_ids = list(stores_map.keys())
        end = parse_date(self.spec.data_end)

        def generate_for_customer(cust):
            region = self.geo_cluster_manager.assign_to_region(cust['latitude'], cust['longitude'])
            region_store_ids = [s['store_id'] for s in stores_map.values() if s.get('region') == region]
            if not region_store_ids:
                return [], None

            order_id = str(uuid4())
            order_total = 0.0
            any_returned = False
            transactions = []

            num_items = random.randint(1, 8)
            txn_date = random_date_weighted(cust['registration_date'], end)

            for _ in range(num_items):
                pid = random.choice(prods)['product_id']
                sid = random.choice(region_store_ids)
                qty = random.randint(1, 5)

                actual_qty = self.stock_manager.check_and_decrement(sid, pid, qty)
                if actual_qty <= 0:
                    continue

                price_unit = price_map.get(pid, 1.0)
                price_total = round(price_unit * actual_qty, 2)

                dist = calculate_distance(
                    cust['latitude'], cust['longitude'],
                    stores_map[sid]['latitude'], stores_map[sid]['longitude']
                )

                txn = {
                    'order_id': order_id,
                    'customer_id': cust['customer_id'],
                    'product_id': pid,
                    'store_id': sid,
                    'quantity': actual_qty,
                    'unit_price': round(price_unit, 2),
                    'total_price': price_total,
                    'transaction_date': txn_date,
                    'distance_km': round(dist, 2),
                }

                txn = apply_returns(txn, txn_date, self.spec.return_prob)
                if not txn.get("is_return"):
                    order_total += price_total
                else:
                    any_returned = True

                transactions.append(txn)

            if not transactions:
                return [], None

            status = (
                "returned" if all(t.get("is_return") for t in transactions) else
                "partial_return" if any_returned else
                "completed"
            )

            order = {
                'order_id': order_id,
                'customer_id': cust['customer_id'],
                'store_id': transactions[0]['store_id'],
                'timestamp': txn_date,
                'total_amount': round(order_total, 2),
                'status': status,
                'channel': random.choice(['in_store', 'app', 'website'])
            }

            return transactions, order

        all_txns = []
        all_orders = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(generate_for_customer, c) for c in custs]
            for fut in as_completed(futures):
                txns, order = fut.result()
                all_txns.extend(txns)
                if order:
                    all_orders.append(order)

        logger.info(f"✅ {len(all_txns)} transactions generated across {len(all_orders)} orders.")
        return pl.DataFrame(all_txns), pl.DataFrame(all_orders)

    def simulate_inventory(
        self, 
        transactions: pl.DataFrame, 
        products: pl.DataFrame
    ) -> pl.DataFrame:
        stock = {}
        inventory = []

        # Extrair e normalizar suppliers por produto
        suppliers = {
            row['product_id']: [
                sup for sup in row.get('suppliers', [])
            ]
            for row in products.to_dicts()
        }

        # Inicializa o estoque com base nas combinações produto-loja
        for row in transactions.select(['product_id', 'store_id']).unique().to_dicts():
            stock[(row['product_id'], row['store_id'])] = random.randint(*self.spec.initial_stock_range)

        # Processa transações em ordem cronológica
        for txn in sorted(transactions.to_dicts(), key=lambda x: x['transaction_date']):
            key = (txn['product_id'], txn['store_id'])
            before = stock[key]
            sold = min(before, txn['quantity'])
            stock[key] = before - sold

            # Reposição condicional
            if stock[key] <= self.spec.reorder_point:
                supplier_options = suppliers.get(txn['product_id'], [])
                if supplier_options:
                    sup = random.choice(supplier_options)
                    lead = sup.get('sla_days', 2)
                    if random.random() < self.spec.disruption_prob:
                        lead += random.randint(*self.spec.disruption_duration_days)
                    arrival_date = txn['transaction_date'] + timedelta(days=lead)
                    stock[key] += self.spec.reorder_quantity  # Aqui ainda estamos simulando chegada imediata

            inventory.append({**txn, 'stock_before': before, 'stock_after': stock[key]})

        return pl.DataFrame(inventory)

    def compute_metrics(self, data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        logger.info("🚀 Starting metric computation")
        start_total = perf_counter()
        results = {}

        metrics_map = {
            'customer_behavior_metrics': self._customer_metrics,
            'customer_channel_preferences': self._customer_channel_preferences,
            'order_summary_metrics': self._order_metrics,
            'store_performance_metrics': self._store_metrics,
            'store_channel_distribution': self._store_channel_distribution,
            'product_return_analysis': self._product_return_analysis,
            'return_behavior_outliers': self._identify_return_outliers,
            'average_items_per_order': self._avg_items_per_order,
            'estimated_total_return_cost': self._estimated_return_cost,
        }

        for name, func in metrics_map.items():
            try:
                logger.info(f"🔍 Computing `{name}`...")
                start = perf_counter()
                result = func(data)
                elapsed = perf_counter() - start
                results[name] = result
                logger.info(f"✅ `{name}` computed in {elapsed:.3f}s — shape: {result.shape}")
            except Exception as e:
                logger.exception(f"❌ Failed to compute `{name}`: {e}")

        total_elapsed = perf_counter() - start_total
        logger.info(f"✅ All metrics computed in {total_elapsed:.2f}s")
        return results


    def _customer_metrics(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        tx = data['transactions']
        today = date.today()
        df = tx.filter(pl.col('returned') == False)

        freq = df.group_by('customer_id').len().rename({'len': 'frequency'})
        rec = df.group_by('customer_id').agg(pl.max('transaction_date').alias('last_date'))
        mon = df.group_by('customer_id').agg(pl.sum('total_price').alias('monetary'))

        rec = rec.with_columns((pl.lit(today) - pl.col('last_date')).dt.total_days().alias('recency_days'))
        cm = freq.join(rec, on='customer_id').join(mon, on='customer_id')
        cm = cm.with_columns((pl.col('monetary') / pl.col('frequency')).alias('avg_ticket'))

        dist = tx.group_by('customer_id').agg(pl.mean('distance_km').alias('avg_distance_km'))
        cm = cm.join(dist, on='customer_id')

        total_tx = tx.group_by('customer_id').len().rename({'len': 'total_tx'})
        returns = tx.filter(pl.col('returned') == True).group_by('customer_id').len().rename({'len': 'returns'})
        returns = total_tx.join(returns, on='customer_id', how='left').fill_null(0)
        returns = returns.with_columns((pl.col('returns') / pl.col('total_tx')).alias('return_rate'))

        return cm.join(returns.select(['customer_id', 'return_rate']), on='customer_id')

    def _customer_channel_preferences(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        orders = data['orders']
        return orders.group_by(['customer_id', 'channel']).len().rename({'len': 'count'})

    def _order_metrics(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        orders = data['orders']
        num_orders = orders.height
        total_sales = orders['total_amount'].sum()

        aov = total_sales / num_orders if num_orders else 0
        return_rate = orders.filter(
            pl.col('status').is_in(['returned', 'partial_return'])
        ).height / num_orders if num_orders else 0

        lead_time = (
            (orders['delivery_date'] - orders['timestamp']).dt.days.mean()
            if 'delivery_date' in orders.columns else None
        )

        return pl.DataFrame([{
            'aov': aov,
            'return_rate_orders': return_rate,
            'avg_lead_time_days': lead_time,
        }])

    def _store_metrics(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        tx = data['transactions']
        inv = data['inventory']
        orders = data['orders']

        sold = inv.group_by('store_id').agg(pl.sum('quantity').alias('total_sold'))
        lost = inv.with_columns((pl.col('quantity') - pl.col('stock_after')).alias('lost_sales'))
        lost = lost.group_by('store_id').agg(pl.sum('lost_sales').alias('lost_sales'))

        store_metrics = sold.join(lost, on='store_id')
        store_metrics = store_metrics.with_columns(
            (pl.col('total_sold') / (pl.col('total_sold') + pl.col('lost_sales'))).alias('conversion_rate')
        )

        sales = tx.group_by('store_id').agg(pl.sum('total_price').alias('total_sales'))
        order_counts = orders.group_by('store_id').len().rename({'len': 'order_count'})
        store_metrics = store_metrics.join(sales, on='store_id', how='left') \
                                     .join(order_counts, on='store_id', how='left')

        store_metrics = store_metrics.with_columns(
            (pl.col('total_sales') / pl.col('order_count')).alias('avg_order_value_store')
        )

        return_tx = tx.filter(pl.col('returned') == True)
        return_mean = return_tx.group_by('store_id').agg(
            pl.mean('total_price').alias('avg_return_value')
        )
        return store_metrics.join(return_mean, on='store_id', how='left').fill_null(0)

    def _store_channel_distribution(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        orders = data['orders']
        return orders.group_by(['store_id', 'channel']).len().rename({'len': 'count'})

    def _product_return_analysis(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        tx = data['transactions']
        total = tx.group_by('product_id').len().rename({'len': 'total_tx'})
        returned = tx.filter(pl.col('returned')).group_by('product_id').len().rename({'len': 'returns'})
        return total.join(returned, on='product_id', how='left') \
                    .fill_null(0) \
                    .with_columns((pl.col('returns') / pl.col('total_tx')).alias('return_rate'))

    def _identify_return_outliers(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        tx = data['transactions']
        returns = tx.filter(pl.col('returned') == True).group_by('customer_id').len().rename({'len': 'returns'})
        quantile_95 = returns['returns'].quantile(0.95)
        return returns.filter(pl.col('returns') >= quantile_95)

    def _avg_items_per_order(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        tx = data['transactions']
        return tx.group_by('order_id').len().mean().rename({'len': 'avg_items'})

    def _estimated_return_cost(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        tx = data['transactions'].filter(pl.col('returned') == True)
        return_cost = tx.with_columns((pl.col('quantity') * pl.col('total_price') / pl.col('quantity')).alias('unit_price'))
        return_cost = return_cost.with_columns((pl.col('quantity') * pl.col('unit_price') * 1.2).alias('estimated_cost'))
        total_cost = return_cost['estimated_cost'].sum()
        return pl.DataFrame([{'total_return_cost': total_cost}])

    def write(self, uri: str, data: Dict[str, pl.DataFrame]):
        parsed = urlparse(uri)
        scheme = parsed.scheme or "file"
        path = uri.replace(f"{scheme}://", "", 1)

        factory = self._SINK_REGISTRY.get(scheme)
        if not factory:
            raise ValueError(f"Unsupported storage scheme: '{scheme}'")

        sink = factory(path if scheme != "postgresql" else uri, self)
        sink.write(data)

    def generate_retail_data(self) -> Dict[str, pl.DataFrame]:
        customers=self.generate_customers(); 
        stores=self.generate_stores(); 
        suppliers=self.generate_suppliers()
        products=self.generate_products(suppliers)
        transactions, orders = self.generate_orders_and_transactions(customers, stores, products, suppliers)
        inventory = self.simulate_inventory(transactions, products)
        
        return {
            'customers':customers,
            'stores':stores,
            'suppliers': suppliers,
            'products':products,
            'orders': orders,
            'transactions':transactions,
            'inventory': inventory,
        }

    def run(self) -> Dict[str, pl.DataFrame]:
        start_time = perf_counter()

        try:
            logger.info("🛒 Generating retail data...")
            retail_data = self.generate_retail_data()
        except Exception as e:
            logger.error(f"❌ Error generating retail data: {e}")
            raise

        try:
            logger.info("📊 Computing retail metrics...")
            metrics = self.compute_metrics(retail_data)
        except Exception as e:
            logger.error(f"❌ Error computing metrics: {e}")
            raise

        try:
            logger.info(f"📦 Writing raw data to `{self.spec.destination}/raw`...")
            self.write(uri=self.spec.destination + "/raw", data=retail_data)

            logger.info(f"📦 Writing metrics to `{self.spec.destination}/metrics`...")
            self.write(uri=self.spec.destination + "/mart", data=metrics)
        except Exception as e:
            logger.error(f"❌ Error writing to sink: {e}")
            raise

        elapsed = perf_counter() - start_time
        logger.info(f"🏁 Synthetic retail data generation complete in {elapsed:.2f}s")
        return retail_data


# Define Clusterization Types as Labels
class ClusterizationType:
    GEOGRAPHIC_EXPANSION = "geographic"
    ECONOMIC_SEGMENTS = "economic"
    MARKET_MATURITY = "maturity"

# Cluster Data Template
CLUSTER_DATA = {
    ClusterizationType.GEOGRAPHIC_EXPANSION: [
        {"name": "LATAM", "lat": -23.5505, "lon": -46.6333, "radius_km": 100, "num_stores": 5, "weight": 0.5, 
         "channel_weights": {"online": 0.3, "in_store": 0.7}, "income_dist": {"low": 0.4, "medium": 0.4, "high": 0.2}, "age_range": [20, 65]},
        {"name": "EMEA", "lat": 48.8566, "lon": 2.3522, "radius_km": 80, "num_stores": 3, "weight": 0.3},
        {"name": "APAC", "lat": 35.6895, "lon": 139.6917, "radius_km": 60, "num_stores": 4, "weight": 0.2}
    ],
    
    ClusterizationType.ECONOMIC_SEGMENTS: [
        {"name": "LATAM", "lat": -23.5505, "lon": -46.6333, "radius_km": 100, "num_stores": 5, "weight": 0.5,
         "channel_weights": {"online": 0.3, "in_store": 0.7}, "income_dist": {"low": 0.6, "medium": 0.3, "high": 0.1}, "age_range": [20, 65]},
        {"name": "EMEA", "lat": 48.8566, "lon": 2.3522, "radius_km": 80, "num_stores": 3, "weight": 0.3,
         "channel_weights": {"online": 0.5, "in_store": 0.5}, "income_dist": {"low": 0.3, "medium": 0.5, "high": 0.2}, "age_range": [25, 60]},
        {"name": "APAC", "lat": 35.6895, "lon": 139.6917, "radius_km": 60, "num_stores": 4, "weight": 0.2,
         "channel_weights": {"online": 0.4, "in_store": 0.6}, "income_dist": {"low": 0.4, "medium": 0.4, "high": 0.2}, "age_range": [18, 55]}
    ],
    
    ClusterizationType.MARKET_MATURITY: [
        {"name": "LATAM", "lat": -23.5505, "lon": -46.6333, "radius_km": 100, "num_stores": 5, "weight": 0.5,
         "channel_weights": {"online": 0.2, "in_store": 0.8}, "income_dist": {"low": 0.5, "medium": 0.4, "high": 0.1}, "age_range": [20, 65]},
        {"name": "EMEA", "lat": 48.8566, "lon": 2.3522, "radius_km": 80, "num_stores": 3, "weight": 0.3,
         "channel_weights": {"online": 0.5, "in_store": 0.5}, "income_dist": {"low": 0.2, "medium": 0.5, "high": 0.3}, "age_range": [25, 60]},
        {"name": "APAC", "lat": 35.6895, "lon": 139.6917, "radius_km": 60, "num_stores": 4, "weight": 0.2,
         "channel_weights": {"online": 0.3, "in_store": 0.7}, "income_dist": {"low": 0.3, "medium": 0.5, "high": 0.2}, "age_range": [18, 55]}
    ]
}

# Create a function to generate GeoClusters based on the strategy type
def generate_geo_clusters(clusterization_type: str) -> List[GeoClusterSpec]:
    if clusterization_type not in CLUSTER_DATA:
        raise ValueError(f"Unknown clusterization type: {clusterization_type}")
    
    clusters = []
    for cluster_info in CLUSTER_DATA[clusterization_type]:
        clusters.append(GeoClusterSpec(**cluster_info))  # Unpack dictionary directly into GeoClusterSpec
    
    return clusters

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic retail data and metrics.")
    
    # Arguments for general data generation
    parser.add_argument(
        "--num_customers", type=int, default=500, 
        help="Number of customers to generate"
    )
    parser.add_argument(
        "--num_products", type=int, default=2000, 
        help="Number of products to generate"
    )
    parser.add_argument(
        "--num_transactions", type=int, default=10000, 
        help="Number of transactions to generate"
    )
    parser.add_argument(
        "--num_stores", type=int, default=10, 
        help="Number of stores to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--destination", type=str, default='./data', 
        help="Data sink URI (e.g. duckdb://data.db or ./data)"
    )
    
    # Argument for clusterization type
    parser.add_argument(
        "--clusterization_type", type=str, choices=[ClusterizationType.GEOGRAPHIC_EXPANSION, 
                                                   ClusterizationType.ECONOMIC_SEGMENTS, 
                                                   ClusterizationType.MARKET_MATURITY], 
        default=ClusterizationType.GEOGRAPHIC_EXPANSION,
        help="Specify the clusterization type"
    )

    args = parser.parse_args()

    # Generate geo clusters based on the specified clusterization type
    geo_clusters = generate_geo_clusters(args.clusterization_type)

    # Build spec from CLI args
    spec = RetailDataSpec(
        num_customers=args.num_customers,
        num_products=args.num_products,
        num_transactions=args.num_transactions,
        seed=args.seed,
        destination=args.destination or RetailDataSpec().destination,
        geo_clusters=geo_clusters
    )

    # Run generator
    gen = RetailDataGenerator(spec)
    gen.run()

if __name__ == '__main__':
    main()
