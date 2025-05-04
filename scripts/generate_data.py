import io
import os
import random
import json
import logging
from urllib.parse import urlparse
import argparse
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Protocol, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict


import polars as pl
import numpy as np
import duckdb
import boto3
from gcsfs import GCSFileSystem
from sqlalchemy import create_engine
from faker import Faker
from dateutil.relativedelta import relativedelta
import math

logger = logging.getLogger("retail_gen")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

@dataclass
class GeoClusterSpec:
    name: str
    lat: float
    lon: float
    radius_km: int
    num_stores: int
    weight: float
    channel_weights: Optional[Dict[str, float]] = None
    income_dist: Optional[Dict[str, float]] = None
    age_range: Optional[List[int]] = None

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

# ----------------------------
# Storage Layer
# ----------------------------
class DataSink(Protocol):
    def write(self, data: Dict[str, pl.DataFrame]): ...

class CSVSink:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def write(self, data: Dict[str, pl.DataFrame]):
        for name, df in data.items():
            df.write_csv(self.path / f"{name}.csv")

class DuckDBSink:
    def __init__(self, path: str):
        self.path = path

    def write(self, data: Dict[str, pl.DataFrame]):
        con = duckdb.connect(self.path)
        for name, df in data.items():
            con.register(name, df.to_arrow())
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name}")
            con.unregister(name)
        con.close()

class SQLSink:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, future=True)

    def write(self, data: Dict[str, pl.DataFrame]):
        with self.engine.begin() as conn:
            for name, df in data.items():
                df.write_database(table_name=name, connection=conn, if_exists="replace")

class S3ParquetSink:
    def __init__(self, bucket: str, prefix: str, compression: str = "snappy"):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.compression = compression
        self.s3 = boto3.client("s3")

    def write(self, data: Dict[str, pl.DataFrame]):
        
        for name, df in data.items():
            buf = io.BytesIO()
            df.write_parquet(buf, compression=self.compression)
            key = f"{self.prefix}{name}.parquet"
            buf.seek(0)
            self.s3.upload_fileobj(buf, self.bucket, key)

class GCSParquetSink:
    def __init__(self, path: str, compression: str = "snappy"):
        
        self.fs = GCSFileSystem()
        self.path = path.rstrip("/") + "/"
        self.compression = compression

    def write(self, data: Dict[str, pl.DataFrame]):
        for name, df in data.items():
            full_path = f"{self.path}{name}.parquet"
            with self.fs.open(full_path, "wb") as f:
                df.write_parquet(f, compression=self.compression)

class ParquetSink:
    def __init__(self, path: str, compression: Optional[str] = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def write(self, data: Dict[str, pl.DataFrame]):
        for name, df in data.items():
            df.write_parquet(self.path / f"{name}.parquet", compression=self.compression)

class JSONLSink:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def write(self, data: Dict[str, pl.DataFrame]):
        for name, df in data.items():
            with open(self.path / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for row in df.iter_rows(named=True):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

class FeatherSink:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def write(self, data: Dict[str, pl.DataFrame]):
        for name, df in data.items():
            df.write_ipc(self.path / f"{name}.feather")  # .feather é .ipc

class InMemorySink:
    def __init__(self):
        self.storage = {}

    def write(self, data: Dict[str, pl.DataFrame]):
        self.storage.update(data)

class SQLiteSink:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)

    def write(self, data: Dict[str, pl.DataFrame]):
        with self.engine.begin() as conn:
            for name, df in data.items():
                df.write_database(table_name=name, connection=conn, if_exists="replace")

class GeoClusterManager:
    def __init__(self, geo_clusters: List[dict]):
        self.regions = {}
        self._initialize_clusters(geo_clusters)
    
    def _initialize_clusters(self, geo_clusters: List[dict]):
        """Inicializa as regiões com base nos dados fornecidos."""
        for cluster in geo_clusters:
            self.regions[cluster['name']] = cluster

    def assign_to_region(self, latitude: float, longitude: float) -> str:
        """Método simples para atribuir um cliente a uma região com base na proximidade."""
        # Simulação de atribuição geográfica (não implementado de forma precisa)
        for region_name, cluster in self.regions.items():
            if self._is_within_radius(latitude, longitude, cluster):
                return region_name
        return "Unknown"

    def _is_within_radius(self, lat: float, lon: float, cluster: dict) -> bool:
        """Método simples para verificar se um ponto está dentro do raio de um cluster"""
        # Aqui, você poderia usar algo como a fórmula de Haversine para calcular a distância
        # entre as coordenadas e verificar se está dentro do raio do cluster.
        return True  # Simples, para exemplo.

    def get_store_coordinates(self, store_id: str):
        """Retorna as coordenadas de uma loja baseada no store_id"""
        # Para simplicidade, podemos retornar coordenadas fictícias
        return (0, 0)


class GeoClusterManager:
    def __init__(self):
        """
        Define broad geo-regions like LATAM, EMEA, APAC, etc.
        Each region contains a list of stores with their geographic coordinates.
        """
        self.regions = {
            "LATAM": [
                {'store_id': 'store_1', 'latitude': -23.5505, 'longitude': -46.6333},  # São Paulo
                {'store_id': 'store_2', 'latitude': -34.6037, 'longitude': -58.3816},  # Buenos Aires
                # Add more stores
            ],
            "EMEA": [
                {'store_id': 'store_3', 'latitude': 51.5074, 'longitude': -0.1278},  # London
                {'store_id': 'store_4', 'latitude': 48.8566, 'longitude': 2.3522},   # Paris
                # Add more stores
            ],
            "APAC": [
                {'store_id': 'store_5', 'latitude': 35.6762, 'longitude': 139.6503},  # Tokyo
                {'store_id': 'store_6', 'latitude': -33.8688, 'longitude': 151.2093}, # Sydney
                # Add more stores
            ]
        }

    def assign_to_region(self, customer_lat, customer_lon):
        """
        Assign a customer to a region (LATAM, EMEA, APAC) based on the nearest store.
        """
        min_distance = float('inf')
        assigned_region = None
        
        for region_name, region_stores in self.regions.items():
            for store in region_stores:
                dist = self.haversine(customer_lat, customer_lon, store['latitude'], store['longitude'])
                if dist < min_distance:
                    min_distance = dist
                    assigned_region = region_name
        
        return assigned_region



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

        # Arquivos locais
        "parquet": lambda path, gen: ParquetSink(path, compression=gen.spec.parquet_compression),
        "csv": lambda path, gen: CSVSink(path),
        "jsonl": lambda path, gen: JSONLSink(path),
        "feather": lambda path, gen: FeatherSink(path),

        # Nuvem
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
        start_date = self.parse_date(self.spec.reg_date_start)
        end_date = self.parse_date(self.spec.data_end)

        clusters: List[GeoClusterSpec] = self.spec.geo_clusters
        if not clusters:
            raise ValueError("❌ No geo_clusters defined in spec.")

        cluster_weights = [c['weight'] for c in clusters]  # Acessando 'weight' diretamente do dicionário
        customers = []

        for _ in range(self.spec.num_customers):
            cluster = random.choices(clusters, weights=cluster_weights)[0]

            lat, lon = random_geo_around(cluster['lat'], cluster['lon'], cluster['radius_km'])

            registration = self.fake.date_between(start_date, end_date)

            channel = random.choices(
                population=['online', 'in_store'],
                weights=[cluster.get('channel_weights', {}).get('online', 0.5),
                         cluster.get('channel_weights', {}).get('in_store', 0.5)]
            )[0]

            income = random.choices(
                population=['low', 'medium', 'high'],
                weights=[cluster.get('income_dist', {}).get('low', 0.3),
                         cluster.get('income_dist', {}).get('medium', 0.5),
                         cluster.get('income_dist', {}).get('high', 0.2)]
            )[0]

            age = random.randint(*cluster['age_range'])

            customers.append({
                'customer_id': self.fake.uuid4(),
                'registration_date': registration,
                'age': age,
                'income': income,
                'channel_pref': channel,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'region': cluster['name'],
            })

        logger.info(f"✅ {len(customers)} customers generated across {len(clusters)} geo clusters.")

        return pl.DataFrame(customers)

    def generate_stores(self) -> pl.DataFrame:
        stores = []
        for region in self.spec.geo_clusters:
            for _ in range(region.num_stores):
                lat, lon = random_geo_around(region.center_lat, region.center_lon, region.spread_km)
                stores.append({
                    'store_id': self.fake.uuid4(),
                    'latitude': round(lat, 6),
                    'longitude': round(lon, 6),
                    'region': region.name
                })

        logger.info(f"✅ {len(stores)} stores generated across {len(self.spec.geo_clusters)} clusters.")
        return pl.DataFrame(stores)

    def generate_suppliers(self, spec: RetailDataSpec) -> List[dict]:
        suppliers_df=pl.DataFrame([
            {
                "supplier_id": f"supplier_{i}",
                "name": self.fake.company(),
                "sla_days": random.randint(1, 7)  # basic SLA per supplier
            }
            for i in range(self.spec.supplier_pool_size)
        ])

        logger.info(f"✅ {len(suppliers_df)} supliers generated.")
        
        return suppliers_df

    def generate_products(self, suppliers: List[dict]) -> List[dict]:
        categories = self.spec.product_categories
        seasonalities = self.spec.product_seasonality

        products = []
        for i in range(self.spec.num_products):
            category = random.choice(categories)
            seasonality = random.choice(seasonalities)
            product_id = f"prod_{i}"

            # Random number of suppliers (with Poisson or range-based noise)
            num_suppliers = max(1, min(
                self.spec.supplier_pool_size,
                np.random.poisson(lam=self.spec.suppliers_per_product)
            ))
            product_suppliers = random.sample(suppliers, k=num_suppliers)

            # If modeling SLA per pair:
            supplier_links = [
                {
                    "supplier_id": s["supplier_id"],
                    "sla_days": random.randint(1, 7)  # override or complement base SLA
                } for s in product_suppliers
            ]

            products.append({
                "product_id": product_id,
                "name": self.fake.word().capitalize(),
                "category": category,
                "seasonality": seasonality,
                "suppliers": supplier_links
            })
        
        products_df=pl.DataFrame(products)
        logger.info(f"✅ {len(products_df)} supliers generated.")

        return products_df

    def generate_transactions(
            self, 
            customers: pl.DataFrame, 
            stores: pl.DataFrame, 
            products: pl.DataFrame, 
            suppliers: Dict[str, List[Dict]],
            parallel: bool = True
        ) -> pl.DataFrame:
        custs = customers.to_dicts()
        prods = products.to_dicts()
        stores_map = {s['store_id']: s for s in stores.to_dicts()}
        store_ids = list(stores_map.keys())

        end = self.parse_date(self.spec.data_end)

        # Criar a instância de controle de estoque
        stock_manager = StockManager(self.spec.initial_stock_range, Lock())

        def generate_for_customer(cust):
            region = self.geo_cluster_manager.assign_to_region(cust['latitude'], cust['longitude'])
            region_store_ids = [store['store_id'] for store in self.geo_cluster_manager.regions[region]]

            transactions = []
            for _ in range(random.randint(1, 8)):
                pid = random.choice(prods)['product_id']
                sid = random.choice(region_store_ids)  # Select store from the region

                qty = random.randint(1, 5)
                decrement_qty = stock_manager.check_and_decrement(sid, pid, qty)
                
                if decrement_qty <= 0:
                    continue  # Produto indisponível

                txn_date = random_date_weighted(cust['registration_date'], end)
                price = price_map[pid] * decrement_qty
                dist = calculate_distance(
                    cust['latitude'], cust['longitude'],
                    stores_map[sid]['latitude'], stores_map[sid]['longitude']
                )

                txn = {
                    'customer_id': cust['customer_id'],
                    'product_id': pid,
                    'store_id': sid,
                    'quantity': decrement_qty,
                    'total_price': price,
                    'transaction_date': txn_date,
                    'distance_km': round(dist, 2),
                }

                txn = apply_returns(txn, txn_date, self.spec.return_prob)

                transactions.append(txn)

            return transactions

        # Gerar transações com execução paralela ou sequencial
        all_txns = []
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(generate_for_customer, c) for c in custs]
                for fut in as_completed(futures):
                    all_txns.extend(fut.result())
        else:
            for c in custs:
                all_txns.extend(generate_for_customer(c))

        logger.info(f"✅ {len(all_txns)} transactions generated across {len(store_ids)} stores.")

        return pl.DataFrame(all_txns)

    def simulate_inventory(
        self, 
        transactions: pl.DataFrame, 
        suppliers: Dict[str, List[Dict]]
    ) -> pl.DataFrame:
        stock = {}
        metrics = []
        
        # Initialize 
        for row in transactions.select(['product_id','store_id']).unique().to_dicts():
            stock[(row['product_id'], row['store_id'])] = random.randint(*self.spec.initial_stock_range)
        # processa transações cronologicamente
        for txn in sorted(transactions.to_dicts(), key=lambda x: x['transaction_date']):
            key = (txn['product_id'], txn['store_id'])
            before = stock[key]
            sold = min(before, txn['quantity'])
            stock[key] = before - sold
            # reposição de estoque se abaixo do ponto de reorder
            if stock[key] <= self.spec.reorder_point:
                sup = random.choice(suppliers[txn['product_id']])
                lead = sup['sla']
                if random.random() < self.spec.disruption_prob:
                    lead += random.randint(*self.spec.disruption_duration_days)
                arrival_date = txn['transaction_date'] + timedelta(days=lead)
                # atualização imediata; em simulação real, considerar pending orders
                stock[key] += self.spec.reorder_quantity
            metrics.append({**txn, 'stock_before': before, 'stock_after': stock[key]})
        return pl.DataFrame(metrics)

    def compute_metrics(self, data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        tx = data['transactions']
        inv = data['inventory']
        
        # Clientes não retornados
        df = tx.filter(pl.col('returned') == False)
        
        # Frequência e monetário
        freq = df.group_by('customer_id').len().rename({'len': 'frequency'})
        rec = df.group_by('customer_id').agg(pl.max('transaction_date').alias('last_date'))
        mon = df.group_by('customer_id').agg(pl.sum('total_price').alias('monetary'))
        
        # Recência
        today = date.today()
        last_dates = rec['last_date'].to_list()
        recency_list = [(today - d).days for d in last_dates]
        rec = rec.with_columns(pl.Series('recency_days', recency_list))
        cm = freq.join(rec, on='customer_id').join(mon, on='customer_id')

        # Métricas de loja: vendas e perdidos
        sold = inv.group_by('store_id').agg(pl.sum('quantity').alias('total_sold'))
        lost_amount = [(row['quantity'] - row['stock_after']) for row in inv.to_dicts()]

        lost_df = pl.DataFrame(inv.to_dicts()).with_columns(pl.Series('lost_sales', lost_amount))
        lost = lost_df.group_by('store_id').agg(pl.sum('lost_sales').alias('lost_sales'))
        sm = sold.join(lost, on='store_id')
        
        return {
            'customer_metrics': cm, 
            'store_metrics': sm
        }

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
        products, suppliers=self.generate_products()
        transactions = self.generate_transactions(customers, stores, products, suppliers)
        inventory = self.simulate_inventory(transactions, suppliers)
        
        return {
            'customers':customers,
            'stores':stores,
            'products':products,
            'suppliers': suppliers,
            'transactions':transactions,
            'inventory': inventory,
        }

    def run(self) -> Dict[str,pl.DataFrame]:
        try:
            logger.info("🛒 Generating retail data...")
            retail_data = self.generate_retail_data()
        except Exception as e:
            logger.error(f"❌ Error generating retail data: {e}")
            raise

        try:
            logger.info("📊 Generating retail metrics...")
            metrics = self.compute_metrics(retail_data)
        except Exception as e:
            logger.error(f"❌ Error computing retail metrics: {e}")
            raise

        try:
            logger.info("🧩 Updating data with metrics...")
            retail_data.update(metrics)
        except Exception as e:
            logger.error(f"❌ Error updating data with metrics: {e}")
            raise

        try:
            logger.info("📦 Populating data sink...")
            self.write(self.spec.destination, retail_data)
        except Exception as e:
            logger.error(f"❌ Error populating data sink: {e}")
            raise

        logger.info("✅ Synthetic retail data generation complete!")
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
        "--destination", type=str, default='duckdb://data/retail.db', 
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
        num_stores=args.num_stores,
        seed=args.seed,
        destination=args.destination or RetailDataSpec().destination,
        geo_clusters=geo_clusters
    )

    # Run generator
    gen = RetailDataGenerator(spec)
    gen.run()



if __name__ == '__main__':
    main()
