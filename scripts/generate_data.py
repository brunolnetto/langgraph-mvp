import os
import random
import logging
from datetime import timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Protocol

import polars as pl
import duckdb
from sqlalchemy import create_engine
from faker import Faker

logger = logging.getLogger("retail_gen")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)


# ----------------------------
# Config & Validation
# ----------------------------
@dataclass
class RetailDataSpec:
    num_customers: int = 100
    num_products: int = 50
    num_transactions: int = 500
    num_stores: int = 5
    reg_date_start: str = '-2y'
    data_end: str = 'today'
    output_dir: str = './data'
    seed: Optional[int] = None
    duckdb_path: Optional[str] = field(default_factory=lambda: os.getenv('DUCKDB_PATH'))

    categories: List[str] = field(default_factory=lambda: [
        'Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports'
    ])
    subcategories: Dict[str, List[str]] = field(default_factory=lambda: {
        'Electronics': ['Mobile', 'TV', 'Audio', 'Computer'],
        'Clothing': ['Men', 'Women', 'Kids'],
        'Home & Kitchen': ['Furniture', 'Cookware', 'Decor'],
        'Books': ['Fiction', 'Non-Fiction', 'Academic'],
        'Sports': ['Outdoor', 'Gym', 'Team Sports']
    })
    brands: List[str] = field(default_factory=lambda: [
        'Acme', 'Globex', 'Soylent', 'Initech', 'Umbrella'
    ])
    channels: List[str] = field(default_factory=lambda: ['online', 'in_store', 'marketplace'])
    store_lat_range: Tuple[float, float] = (-23.7, -23.5)
    store_lng_range: Tuple[float, float] = (-46.7, -46.5)
    customer_jitter: float = 0.02
    markup_range: Tuple[float, float] = (1.2, 2.0)
    initial_stock_range: Tuple[int, int] = (20, 200)
    reorder_point: int = 10
    reorder_quantity: int = 100
    reorder_lead_time_days: Tuple[int, int] = (1, 5)
    parquet_compression: Optional[str] = None


# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

def fake_instance(seed: Optional[int]) -> Faker:
    return Faker()

# ----------------------------
# Data Generators
# ----------------------------
def gen_customers(spec: RetailDataSpec, fake: Faker) -> pl.DataFrame:
    lat_min, lat_max = spec.store_lat_range
    lng_min, lng_max = spec.store_lng_range
    jitter = spec.customer_jitter
    data = [{
        'customer_id': fake.uuid4(),
        'name': fake.name(),
        'email': fake.email(),
        'phone': fake.phone_number(),
        'address': fake.address().replace("\n", ", "),
        'registration_date': fake.date_between(spec.reg_date_start, spec.data_end),
        'latitude': round(random.uniform(lat_min, lat_max) + random.uniform(-jitter, jitter), 6),
        'longitude': round(random.uniform(lng_min, lng_max) + random.uniform(-jitter, jitter), 6)
    } for _ in range(spec.num_customers)]
    return pl.DataFrame(data)

def gen_stores(spec: RetailDataSpec, fake: Faker) -> pl.DataFrame:
    lat_min, lat_max = spec.store_lat_range
    lng_min, lng_max = spec.store_lng_range
    data = [{
        'store_id': fake.uuid4(),
        'name': f"{fake.city()} {fake.company_suffix()}",
        'address': fake.address().replace("\n", ", "),
        'latitude': round(random.uniform(lat_min, lat_max), 6),
        'longitude': round(random.uniform(lng_min, lng_max), 6)
    } for _ in range(spec.num_stores)]
    return pl.DataFrame(data)

def gen_products(spec: RetailDataSpec, fake: Faker) -> pl.DataFrame:
    cats = spec.categories
    subs = spec.subcategories
    brands = spec.brands
    mmin, mmax = spec.markup_range
    data = []
    for _ in range(spec.num_products):
        cat = random.choice(cats)
        price = round(random.uniform(5.0, 1000.0), 2)
        cost = round(price / random.uniform(mmin, mmax), 2)
        data.append({
            'product_id': fake.uuid4(),
            'category': cat,
            'subcategory': random.choice(subs[cat]),
            'brand': random.choice(brands),
            'price': price,
            'cost_price': cost
        })
    return pl.DataFrame(data)

def gen_transactions(spec: RetailDataSpec, customers, stores, products, fake: Faker) -> pl.DataFrame:
    cust_list, store_list, prod_list = customers.to_dicts(), stores.to_dicts(), products.to_dicts()
    chans = spec.channels
    data = []
    for _ in range(spec.num_transactions):
        c, s, p = random.choice(cust_list), random.choice(store_list), random.choice(prod_list)
        q = random.randint(1, 10)
        tp, ct = round(p['price'] * q, 2), round(p['cost_price'] * q, 2)
        prof = round(tp - ct, 2)
        ch = random.choice(chans)
        ship = round(q * random.uniform(1.0, 5.0), 2) if ch == 'online' else 0.0
        data.append({
            'transaction_id': fake.uuid4(),
            'customer_id': c['customer_id'],
            'store_id': s['store_id'],
            'product_id': p['product_id'],
            'quantity': q,
            'total_price': tp,
            'cost_total': ct,
            'profit': prof,
            'channel': ch,
            'shipping_cost': ship,
            'transaction_date': fake.date_between(c['registration_date'], spec.data_end)
        })
    return pl.DataFrame(data)

def simulate_inventory(transactions: pl.DataFrame, spec: RetailDataSpec,
                       products: pl.DataFrame, stores: pl.DataFrame) -> pl.DataFrame:
    txns = transactions.to_dicts()
    stock = {(p['product_id'], s['store_id']): random.randint(*spec.initial_stock_range)
             for p in products.to_dicts() for s in stores.to_dicts()}
    pending, enriched = [], []
    for ev in sorted(txns, key=lambda x: x['transaction_date']):
        key = (ev['product_id'], ev['store_id'])
        date = ev['transaction_date']
        for r in pending[:]:
            if r['arrival_date'] <= date and (r['product_id'], r['store_id']) == key:
                stock[key] += r['quantity']
                pending.remove(r)
        before, after = stock[key], max(stock[key] - ev['quantity'], 0)
        ev.update({'stock_before': before, 'stock_after': after})
        stock[key] = after
        if after <= spec.reorder_point and not any((r['product_id'], r['store_id']) == key for r in pending):
            pending.append({
                'product_id': key[0], 'store_id': key[1],
                'arrival_date': date + timedelta(days=random.randint(*spec.reorder_lead_time_days)),
                'quantity': spec.reorder_quantity
            })
            ev['reorder_placed'] = True
        else:
            ev['reorder_placed'] = False
        enriched.append(ev)
    return pl.DataFrame(enriched)

# ----------------------------
# Storage Layer
# ----------------------------
class DataSink(Protocol):
    def write(self, data: Dict[str, pl.DataFrame]): ...

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
        for name, df in data.items():
            df.to_pandas().to_sql(name, self.engine, if_exists="replace", index=False)
        self.engine.dispose()

class ParquetSink:
    def __init__(self, path: str, compression: Optional[str] = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def write(self, data: Dict[str, pl.DataFrame]):
        for name, df in data.items():
            df.write_parquet(self.path / f"{name}.parquet", compression=self.compression)

# ----------------------------
# Orchestrator
# ----------------------------
def generate_synthetic_retail_data(spec: RetailDataSpec) -> Dict[str, pl.DataFrame]:
    seed_everything(spec.seed)
    fake = fake_instance(spec.seed)

    logger.info("Generating customers...")
    customers = gen_customers(spec, fake)
    logger.info(f"{len(customers)} customers generated.")

    logger.info("Generating stores...")
    stores = gen_stores(spec, fake)
    logger.info(f"{len(stores)} stores generated.")

    logger.info("Generating products...")
    products = gen_products(spec, fake)
    logger.info(f"{len(products)} products generated.")

    logger.info("Generating transactions...")
    transactions = gen_transactions(spec, customers, stores, products, fake)
    logger.info(f"{len(transactions)} transactions generated.")

    logger.info("Simulating inventory...")
    transactions = simulate_inventory(transactions, spec, products, stores)
    logger.info("Inventory simulation complete.")

    data = {'customers': customers, 'stores': stores, 'products': products, 'transactions': transactions}

    db_url = os.getenv('RETAIL_DB_URL', '')
    if db_url.startswith('duckdb://'):
        logger.info(f"Writing data to DuckDB at {db_url}...")
        DuckDBSink(db_url.replace('duckdb://', '')).write(data)
    elif db_url.startswith(('postgresql://', 'mysql://', 'mysql+pymysql://')):
        logger.info(f"Writing data to SQL database at {db_url}...")
        SQLSink(db_url).write(data)
    else:
        logger.info(f"Writing data to Parquet files at {spec.output_dir}...")
        ParquetSink(spec.output_dir, compression=spec.parquet_compression).write(data)

    logger.info("✅ Synthetic retail data generation complete.")
    return data


# ----------------------------
# Main Function
# ----------------------------
if __name__ == '__main__':
    spec = RetailDataSpec(
        num_customers=500,
        num_products=200,
        num_transactions=2000,
        num_stores=10,
        seed=42
    )
    generate_synthetic_retail_data(spec)
