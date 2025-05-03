import random
from faker import Faker
from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Tuple, List
from datetime import timedelta

import pandas as pd
import duckdb

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
    duckdb_path: Optional[str] = None

    # hierarchy & channels & geo
    categories: List[str] = field(default_factory=lambda: [
        'Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports'
    ])
    subcategories: Dict[str,List[str]] = field(default_factory=lambda: {
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
    store_lat_range: Tuple[float,float] = (-23.7, -23.5)
    store_lng_range: Tuple[float,float] = (-46.7, -46.5)
    customer_jitter: float   = 0.02

    # profit & inventory
    markup_range: Tuple[float,float]       = (1.2, 2.0)
    initial_stock_range: Tuple[int,int]   = (20, 200)
    reorder_point: int                    = 10
    reorder_quantity: int                 = 100
    reorder_lead_time_days: Tuple[int,int]= (1, 5)

    # parquet
    parquet_compression: Optional[str]    = None  # None = no compression


def seed_everything(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

def gen_customers(spec: RetailDataSpec) -> pd.DataFrame:
    fake = Faker()
    lat_min, lat_max = spec.store_lat_range
    lng_min, lng_max = spec.store_lng_range
    jitter = spec.customer_jitter
    rows = []
    for _ in range(spec.num_customers):
        base_lat = random.uniform(lat_min, lat_max)
        base_lng = random.uniform(lng_min, lng_max)
        rows.append({
            'customer_id':    fake.uuid4(),
            'name':           fake.name(),
            'email':          fake.email(),
            'phone':          fake.phone_number(),
            'address':        fake.address().replace("\n", ", "),
            'registration_date':
                              fake.date_between(spec.reg_date_start, spec.data_end),
            'latitude':       round(base_lat  + random.uniform(-jitter, jitter), 6),
            'longitude':      round(base_lng  + random.uniform(-jitter, jitter), 6),
        })
    return pd.DataFrame(rows)

def gen_stores(spec: RetailDataSpec) -> pd.DataFrame:
    fake = Faker()
    lat_min, lat_max = spec.store_lat_range
    lng_min, lng_max = spec.store_lng_range
    rows = []
    for _ in range(spec.num_stores):
        rows.append({
            'store_id':  fake.uuid4(),
            'name':      f"{fake.city()} {fake.company_suffix()}",
            'address':   fake.address().replace("\n", ", "),
            'latitude':  round(random.uniform(lat_min, lat_max), 6),
            'longitude': round(random.uniform(lng_min, lng_max), 6),
        })
    return pd.DataFrame(rows)

def gen_products(spec: RetailDataSpec) -> pd.DataFrame:
    fake = Faker()
    cats  = spec.categories
    subs  = spec.subcategories
    brands= spec.brands
    mmin, mmax = spec.markup_range
    rows = []
    for _ in range(spec.num_products):
        cat = random.choice(cats)
        price = round(random.uniform(5.0, 1000.0), 2)
        cost  = round(price / random.uniform(mmin, mmax), 2)
        rows.append({
            'product_id':  fake.uuid4(),
            'category':    cat,
            'subcategory': random.choice(subs[cat]),
            'brand':       random.choice(brands),
            'price':       price,
            'cost_price':  cost
        })
    return pd.DataFrame(rows)

def gen_transactions(spec: RetailDataSpec,
                     customers: pd.DataFrame,
                     stores: pd.DataFrame,
                     products: pd.DataFrame) -> pd.DataFrame:
    fake = Faker()
    cust_list = customers.to_dict('records')
    store_list= stores.to_dict('records')
    prod_list = products.to_dict('records')
    chans     = spec.channels
    rows = []
    for _ in range(spec.num_transactions):
        c = random.choice(cust_list)
        s = random.choice(store_list)
        p = random.choice(prod_list)
        q = random.randint(1, 10)
        tp = round(p['price'] * q, 2)
        ct = round(p['cost_price'] * q, 2)
        prof = round(tp - ct, 2)
        ch = random.choice(chans)
        ship = round(q * random.uniform(1.0, 5.0), 2) if ch == 'online' else 0.0
        rows.append({
            'transaction_id': fake.uuid4(),
            'customer_id':    c['customer_id'],
            'store_id':       s['store_id'],
            'product_id':     p['product_id'],
            'quantity':       q,
            'total_price':    tp,
            'cost_total':     ct,
            'profit':         prof,
            'channel':        ch,
            'shipping_cost':  ship,
            'transaction_date':
                              fake.date_between(c['registration_date'], spec.data_end)
        })
    return pd.DataFrame(rows)

def simulate_inventory(transactions: pd.DataFrame,
                       spec: RetailDataSpec,
                       products: pd.DataFrame,
                       stores: pd.DataFrame) -> pd.DataFrame:

    # init stock dict
    stock = {
        (prod.product_id, store.store_id):
            random.randint(*spec.initial_stock_range)
        for prod in products.itertuples()
        for store in stores.itertuples()
    }
    pending = []
    events  = sorted(transactions.to_dict('records'),
                     key=lambda x: x['transaction_date'])
    enriched = []
    rp       = spec.reorder_point
    rq       = spec.reorder_quantity
    lt_min, lt_max = spec.reorder_lead_time_days

    for ev in events:
        key  = (ev['product_id'], ev['store_id'])
        date = ev['transaction_date']
        # arrivals by index
        i = 0
        while i < len(pending):
            r = pending[i]
            if r['arrival_date'] <= date and (r['product_id'],r['store_id'])==key:
                stock[key] += r['quantity']
                pending.pop(i)
            else:
                i += 1

        before = stock[key]
        after  = max(before - ev['quantity'], 0)
        ev['stock_before'] = before
        ev['stock_after']  = after
        stock[key] = after

        # reorder?
        if after <= rp and not any((r['product_id'],r['store_id'])==key for r in pending):
            lead = random.randint(lt_min, lt_max)
            pending.append({
                'product_id': key[0],
                'store_id':   key[1],
                'arrival_date': date + timedelta(days=lead),
                'quantity':   rq
            })
            ev['reorder_placed'] = True
        else:
            ev['reorder_placed'] = False

        enriched.append(ev)

    return pd.DataFrame(enriched)

def generate_synthetic_retail_data(spec: RetailDataSpec) -> Dict[str, pd.DataFrame]:
    seed_everything(spec.seed)
    logging.basicConfig(level=logging.INFO)
    out_path = Path(spec.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate tables
    customers   = gen_customers(spec)
    stores      = gen_stores(spec)
    products    = gen_products(spec)
    transactions= gen_transactions(spec, customers, stores, products)
    transactions= simulate_inventory(transactions, spec, products, stores)

    data = {
        'customers':   customers,
        'stores':      stores,
        'products':    products,
        'transactions':transactions
    }

    # 2. Save Parquet with parallel writes, no compression
    for name, df in data.items():
        df.to_parquet(
            out_path / f"{name}.parquet",
            index=False,
            compression=spec.parquet_compression,
            use_dictionary=False
        )
    logging.info(f"Parquet files saved to {out_path.resolve()}")

    # 3. Optional DuckDB register
    if spec.duckdb_path:
        con = duckdb.connect(spec.duckdb_path)
        for name, df in data.items():
            con.register(name, df)
        logging.info(f"Tables registered in DuckDB at {spec.duckdb_path}")

    return data


if __name__ == '__main__':
    spec = RetailDataSpec(
        num_customers=500,
        num_products=200,
        num_transactions=2000,
        num_stores=10,
        seed=42,
        duckdb_path=':memory:',
        parquet_compression=None
    )
    data = generate_synthetic_retail_data(spec)
    print({k: v.shape for k, v in data.items()})
