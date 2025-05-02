import random
from faker import Faker
import pandas as pd
from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable
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
    categories: list = field(default_factory=lambda: [
        'Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports'
    ])
    parquet_compression: str = 'snappy'


def seed_everything(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)


def gen_customers(spec: RetailDataSpec) -> pd.DataFrame:
    fake = Faker()
    return pd.DataFrame([
        {
            'customer_id': fake.uuid4(),
            'name': fake.name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'address': fake.address().replace("\n", ", "),
            'registration_date': fake.date_between(
                start_date=spec.reg_date_start, end_date=spec.data_end
            )
        }
        for _ in range(spec.num_customers)
    ])


def gen_stores(spec: RetailDataSpec) -> pd.DataFrame:
    fake = Faker()
    return pd.DataFrame([
        {
            'store_id': fake.uuid4(),
            'name': f"{fake.city()} {fake.company_suffix()}",
            'address': fake.address().replace("\n", ", ")
        }
        for _ in range(spec.num_stores)
    ])


def gen_products(spec: RetailDataSpec) -> pd.DataFrame:
    fake = Faker()
    return pd.DataFrame([
        {
            'product_id': fake.uuid4(),
            'name': fake.unique.catch_phrase(),
            'category': random.choice(spec.categories),
            'price': round(random.uniform(5.0, 1000.0), 2)
        }
        for _ in range(spec.num_products)
    ])


def gen_transactions(spec: RetailDataSpec, customers: pd.DataFrame, 
                     stores: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    fake = Faker()
    cust_list = customers.to_dict('records')
    store_list = stores.to_dict('records')
    prod_list = products.to_dict('records')
    return pd.DataFrame([
        {
            'transaction_id': fake.uuid4(),
            'customer_id': (c := random.choice(cust_list))['customer_id'],
            'store_id': (s := random.choice(store_list))['store_id'],
            'product_id': (p := random.choice(prod_list))['product_id'],
            'quantity': (q := random.randint(1, 10)),
            'total_price': round(p['price'] * q, 2),
            'transaction_date': fake.date_between(
                start_date=c['registration_date'], end_date=spec.data_end
            )
        }
        for _ in range(spec.num_transactions)
    ])


def generate_synthetic_retail_data(spec: RetailDataSpec) -> Dict[str, pd.DataFrame]:
    # setup
    seed_everything(spec.seed)
    logging.basicConfig(level=logging.INFO)
    out_path = Path(spec.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # map generators
    generators: Dict[str, Callable] = {
        'customers': gen_customers,
        'stores': gen_stores,
        'products': gen_products
    }
    data = {name: fn(spec) for name, fn in generators.items()}
    data['transactions'] = gen_transactions(
        spec, data['customers'], data['stores'], data['products']
    )

    # save parquet and optionally register
    for name, df in data.items():
        path = out_path / f"{name}.parquet"
        df.to_parquet(path, index=False, compression=spec.parquet_compression)
    logging.info(f"Parquet files saved to {out_path.resolve()}")

    if spec.duckdb_path:
        con = duckdb.connect(spec.duckdb_path)
        for name in data:
            con.register(name, data[name])
        logging.info(f"Tables registered in DuckDB at {spec.duckdb_path}")

    return data


if __name__ == '__main__':
    spec = RetailDataSpec(
        num_customers=200,
        num_products=100,
        num_transactions=1000,
        num_stores=10,
        seed=42,
        duckdb_path=':memory:'
    )
    data = generate_synthetic_retail_data(spec)
    print({k: v.shape for k, v in data.items()})
