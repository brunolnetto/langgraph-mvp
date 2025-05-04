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
from concurrent.futures import ThreadPoolExecutor

import polars as pl
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
class RetailDataSpec:
    num_customers: int = 100
    num_products: int = 50
    num_transactions: int = 500
    num_stores: int = 5
    reg_date_start: str = '-2y'
    data_end: str = 'today'
    seed: Optional[int] = None
    destination: str = field(default_factory=lambda: os.getenv('RETAIL_DESTINATION', './data'))
    parquet_compression: Optional[str] = None
    suppliers_per_product: int = 2
    return_prob: float = 0.05
    disruption_prob: float = 0.1
    disruption_duration_days: Tuple[int, int] = (3, 10)
    initial_stock_range: Tuple[int, int] = (20, 200)
    reorder_point: int = 10
    reorder_quantity: int = 100
    reorder_lead_time_days: Tuple[int, int] = (1, 5)

    def __post_init__(self):
        assert self.num_customers > 0
        assert self.num_products > 0
        assert self.num_transactions > 0
        assert self.num_stores > 0

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

    # Utility
    @staticmethod
    def parse_date(rel: str) -> date:
        if rel == 'today': return date.today()
        if rel.endswith('y'):
            return date.today() + relativedelta(years=int(rel[:-1]))
        if rel.endswith('d'):
            return date.today() + timedelta(days=int(rel[:-1]))
        raise ValueError("Invalid date str")

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R=6371; dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
        a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return R*2*math.asin(math.sqrt(a))

    def gen_customers(self) -> pl.DataFrame:
        s, e = self.parse_date(self.spec.reg_date_start), self.parse_date(self.spec.data_end)
        data=[]
        for _ in range(self.spec.num_customers):
            reg=self.fake.date_between(s,e)
            lat=random.uniform(*self.spec.initial_stock_range)
            data.append({
                'customer_id': self.fake.uuid4(),
                'registration_date': reg,
                'age': random.randint(18,80),
                'income': random.choice(['low','medium','high']),
                'channel_pref': random.choice(['online','in_store','marketplace']),
                'latitude': round(random.uniform(*self.spec.initial_stock_range),6),
                'longitude': round(random.uniform(*self.spec.initial_stock_range),6)
            })
        return pl.DataFrame(data)

    def gen_stores(self) -> pl.DataFrame:
        data=[]
        for _ in range(self.spec.num_stores):
            data.append({'store_id': self.fake.uuid4(),
                         'latitude': round(random.uniform(*self.spec.initial_stock_range),6),
                         'longitude': round(random.uniform(*self.spec.initial_stock_range),6)})
        return pl.DataFrame(data)

    def gen_products(self) -> Tuple[pl.DataFrame, Dict[str,List[Dict]]]:
        prods=[]; suppliers={}
        for _ in range(self.spec.num_products):
            pid=self.fake.uuid4()
            prods.append({'product_id':pid,'price':round(random.uniform(5,1000),2)})
            sups=[]
            for _ in range(self.spec.suppliers_per_product):
                sups.append({'sla': random.randint(*self.spec.reorder_lead_time_days)})
            suppliers[pid]=sups
        return pl.DataFrame(prods), suppliers

    def gen_transactions(
        self, 
        customers: pl.DataFrame, 
        stores: pl.DataFrame, 
        products: pl.DataFrame, 
        suppliers: Dict[str, List[Dict]]
    ) -> pl.DataFrame:
        custs = customers.to_dicts()
        stores_map = {s['store_id']: s for s in stores.to_dicts()}
        prods = products.to_dicts()
        store_ids = list(stores_map.keys())
        rec = []
        end = self.parse_date(self.spec.data_end)
        for c in custs:
            for _ in range(random.randint(1, 8)):
                pid = random.choice(prods)['product_id']
                sid = random.choice(store_ids)
                txn_date = self.fake.date_between(c['registration_date'], end)
                qty = random.randint(1, 5)
                price = next(p['price'] for p in prods if p['product_id'] == pid) * qty
                dist = self.haversine(
                    c['latitude'], c['longitude'],
                    stores_map[sid]['latitude'], stores_map[sid]['longitude']
                )
                txn = {
                    'customer_id': c['customer_id'],
                    'product_id': pid,
                    'store_id': sid,
                    'quantity': qty,
                    'total_price': price,
                    'transaction_date': txn_date,
                    'distance_km': round(dist, 2)
                }
                # returns
                if random.random() < self.spec.return_prob:
                    return_date = txn_date + timedelta(days=random.randint(1, 30))
                    txn['returned'] = True
                    txn['return_date'] = return_date
                else:
                    txn['returned'] = False
                    txn['return_date'] = None
                rec.append(txn)
        return pl.DataFrame(rec)

    def simulate_inventory(
        self, 
        transactions: pl.DataFrame, 
        suppliers: Dict[str, List[Dict]]
    ) -> pl.DataFrame:
        stock = {}
        metrics = []
        # inicializa estoque por loja-produto
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
        cust=self.gen_customers(); 
        stores=self.gen_stores(); 
        prods, sup=self.gen_products()
        tx = self.gen_transactions(cust, stores, prods, sup)
        inv = self.simulate_inventory(tx, sup)
        
        return {
            'customers':cust,
            'stores':stores,
            'products':prods,
            'transactions':tx,
            'inventory':inv
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

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic retail data and metrics.")
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
        "--destination", type=str,  default='duckdb://data/retail.db', 
        help="Data sink URI (e.g. duckdb://data.db or ./data)"
    )
    args = parser.parse_args()

    # Build spec from CLI args
    spec = RetailDataSpec(
        num_customers=args.num_customers,
        num_products=args.num_products,
        num_transactions=args.num_transactions,
        num_stores=args.num_stores,
        seed=args.seed,
        destination=args.destination or RetailDataSpec().destination
    )

    # Run generator
    gen = RetailDataGenerator(spec)
    gen.run()

if __name__ == '__main__':
    main()
