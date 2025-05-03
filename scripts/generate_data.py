import os
import random
import logging
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Protocol, Type, Set
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import duckdb
from sqlalchemy import create_engine
from faker import Faker
from dateutil.relativedelta import relativedelta

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
    seed: Optional[int] = None
    destination: str = field(default_factory=lambda: os.getenv('RETAIL_DESTINATION', './data'))

    parquet_compression: Optional[str] = None

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

    def __post_init__(self):
        assert self.num_customers > 0, "num_customers must be > 0"
        assert self.num_products > 0, "num_products must be > 0"
        assert self.num_transactions > 0, "num_transactions must be > 0"
        assert self.num_stores > 0, "num_stores must be > 0"
        assert self.store_lat_range[0] < self.store_lat_range[1], "Invalid latitude range"
        assert self.store_lng_range[0] < self.store_lng_range[1], "Invalid longitude range"


# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

def fake_instance(seed: Optional[int]) -> Faker:
    return Faker() if seed is None else Faker(seed)

def parse_relative_date(relative_str: str) -> date:
    if relative_str == 'today':
        return date.today()
    elif relative_str.endswith('y'):
        years = int(relative_str[:-1])
        return date.today() + relativedelta(years=years)
    elif relative_str.endswith('d'):
        days = int(relative_str[:-1])
        return date.today() + timedelta(days=days)
    raise ValueError(f"Unsupported relative date format: {relative_str}")

# ----------------------------
# Data Generators
# ----------------------------
def gen_customers(spec: RetailDataSpec, fake: Faker) -> pl.DataFrame:
    lat_min, lat_max = spec.store_lat_range
    lng_min, lng_max = spec.store_lng_range
    jitter = spec.customer_jitter
    start = parse_relative_date(spec.reg_date_start)
    end = parse_relative_date(spec.data_end)
    data = [
        {
            'customer_id': fake.uuid4(),
            'name': fake.name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'address': fake.address().replace("\n", ", "),
            'registration_date': fake.date_between(start, end),
            'latitude': round(random.uniform(lat_min, lat_max) + random.uniform(-jitter, jitter), 6),
            'longitude': round(random.uniform(lng_min, lng_max) + random.uniform(-jitter, jitter), 6)
        } for _ in range(spec.num_customers)
    ]
    return pl.DataFrame(data)

def gen_stores(spec: RetailDataSpec, fake: Faker) -> pl.DataFrame:
    lat_min, lat_max = spec.store_lat_range
    lng_min, lng_max = spec.store_lng_range
    data = [
        {
            'store_id': fake.uuid4(),
            'name': f"{fake.city()} {fake.company_suffix()}",
            'address': fake.address().replace("\n", ", "),
            'latitude': round(random.uniform(lat_min, lat_max), 6),
            'longitude': round(random.uniform(lng_min, lng_max), 6)
        } for _ in range(spec.num_stores)
    ]
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
    end = parse_relative_date(spec.data_end)
    data = []
    def create_txn():
        c, s, p = random.choice(cust_list), random.choice(store_list), random.choice(prod_list)
        q = random.randint(1, 10)
        tp, ct = round(p['price'] * q, 2), round(p['cost_price'] * q, 2)
        prof = round(tp - ct, 2)
        ch = random.choice(chans)
        ship = round(q * random.uniform(1.0, 5.0), 2) if ch == 'online' else 0.0
        return {
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
            'transaction_date': fake.date_between(c['registration_date'], end)
        }
    with ThreadPoolExecutor() as executor:
        data = list(executor.map(lambda _: create_txn(), range(spec.num_transactions)))
    return pl.DataFrame(data)


def simulate_inventory(
    transactions: pl.DataFrame,
    spec: RetailDataSpec,
    products: pl.DataFrame,
    stores: pl.DataFrame
) -> pl.DataFrame:
    
    initial_stock = {
        (p["product_id"], s["store_id"]): random.randint(*spec.initial_stock_range)
        for p in products.to_dicts()
        for s in stores.to_dicts()
    }

    txns = sorted(transactions.to_dicts(), key=lambda x: x["transaction_date"])
    pending_orders: List[Dict] = []
    reorder_index: Set[Tuple[str, str]] = set()

    enriched = []
    stock = initial_stock.copy()

    for txn in txns:
        key = (txn["product_id"], txn["store_id"])
        date = txn["transaction_date"]

        # Processar pedidos pendentes com chegada nesta data ou anterior
        arrivals_today = [r for r in pending_orders if r["arrival_date"] <= date and (r["product_id"], r["store_id"]) == key]
        for r in arrivals_today:
            stock[key] += r["quantity"]
            pending_orders.remove(r)
            reorder_index.discard((r["product_id"], r["store_id"]))

        before_stock = stock[key]
        after_stock = max(before_stock - txn["quantity"], 0)

        txn["stock_before"] = before_stock
        txn["stock_after"] = after_stock
        stock[key] = after_stock

        # Gatilho de reposição
        if after_stock <= spec.reorder_point and key not in reorder_index:
            lead_time = random.randint(*spec.reorder_lead_time_days)
            arrival_date = date + timedelta(days=lead_time)
            reorder = {
                "product_id": key[0],
                "store_id": key[1],
                "arrival_date": arrival_date,
                "quantity": spec.reorder_quantity
            }
            pending_orders.append(reorder)
            reorder_index.add(key)
            txn["reorder_placed"] = True
        else:
            txn["reorder_placed"] = False

        enriched.append(txn)

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
        with self.engine.begin() as conn:
            for name, df in data.items():
                df.write_database(table_name=name, connection=conn, if_exists="replace")

class ParquetSink:
    def __init__(self, path: str, compression: Optional[str] = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def write(self, data: Dict[str, pl.DataFrame]):
        for name, df in data.items():
            df.write_parquet(self.path / f"{name}.parquet", compression=self.compression)

SINK_REGISTRY: Dict[str, Type[DataSink]] = {
    'duckdb': DuckDBSink,
    'parquet': ParquetSink,
    'sql': SQLSink
}

# ----------------------------
# Orchestrator
# ----------------------------
def generate_retail_data(spec: RetailDataSpec) -> Dict[str, pl.DataFrame]:
    seed_everything(spec.seed)
    fake = fake_instance(spec.seed)
    customers = gen_customers(spec, fake)
    stores = gen_stores(spec, fake)
    products = gen_products(spec, fake)
    transactions = gen_transactions(spec, customers, stores, products, fake)
    inventory = simulate_inventory(transactions, spec, products, stores)
    return {
        'customers': customers,
        'stores': stores,
        'products': products,
        'transactions': transactions,
        'inventory': inventory
    }

def populate_retail_sinks(spec: RetailDataSpec, data: Dict[str, pl.DataFrame]) -> None:
    dest = spec.destination

    if dest.startswith('duckdb://'):
        logger.info(f"🦆 Writing data to DuckDB at {dest}...")
        SINK_REGISTRY['duckdb'](dest.replace('duckdb://', '')).write(data)
    elif dest.startswith(('postgresql://', 'mysql://', 'mysql+pymysql://')):
        logger.info(f"🛢️ Writing data to SQL database at {dest}...")
        SINK_REGISTRY['sql'](dest).write(data)
    else:
        logger.info(f"📁 Writing data to Parquet files at {dest}...")
        SINK_REGISTRY['parquet'](dest, compression=spec.parquet_compression).write(data)

def synthetic_retail_data_pipeline(spec: RetailDataSpec) -> Dict[str, pl.DataFrame]:
    logger.info("🛒 Generating synthetic retail data...")
    data = generate_retail_data(spec)
    
    db_url = os.getenv('RETAIL_DB_URL', '')
    logger.info(f"📦 Populating data sink {db_url}...")
    populate_retail_sinks(spec, data)
    
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
    synthetic_retail_data_pipeline(spec)
