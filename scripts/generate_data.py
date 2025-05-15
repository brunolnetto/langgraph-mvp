import io
import os
import random
import json
from time import perf_counter
from enum import Enum
import logging
from urllib.parse import urlparse
from uuid import uuid4
import argparse
from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Tuple, List, Protocol, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
import datetime
from dateutil.relativedelta import relativedelta
import math

from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic.functional_validators import AfterValidator
import polars as pl
import pandas as pd
import numpy as np
import duckdb
from boto3 import client as boto_client
from gcsfs import GCSFileSystem
from sqlalchemy import create_engine, text
from faker import Faker


logger = logging.getLogger("retail_gen")
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# --- Enums

class ClusterizationType(str, Enum):
    GEOGRAPHIC_EXPANSION = "geographic"
    ECONOMIC_SEGMENTS = "economic"
    MARKET_MATURITY = "maturity"
    LIFESTYLE_SEGMENTS = "lifestyle" 
    TIME_SENSITIVE = "time"
    AGE_SPECIFIC = "age"
    


# --- Channel Weights

class ChannelWeights(BaseModel):
    online: float = 0.5
    in_store: float = 0.5

    @model_validator(mode="after")
    def validate_sum(self):
        total = self.online + self.in_store
        if not 0.99 <= total <= 1.01:
            raise ValueError("Channel weights must sum to approximately 1.0")
        return self


# --- Income Distribution

class IncomeDistribution(BaseModel):
    low: float
    medium: float
    high: float

    @model_validator(mode="after")
    def validate_sum(self):
        total = self.low + self.medium + self.high
        if not 0.99 <= total <= 1.01:
            raise ValueError("Income distribution must sum to approximately 1.0")
        return self


# --- Demographics

class Demographics(BaseModel):
    age_range: List[int] = Field(default_factory=lambda: [18, 65])
    income_dist: IncomeDistribution

    @field_validator("age_range")
    def validate_age_range(cls, v: List[int]):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("age_range must be [min_age, max_age]")
        return v


# --- Cluster Spec

class GeoClusterSpec(BaseModel):
    name: str
    lat: float
    lon: float
    radius_km: int
    num_stores: int
    weight: float
    channels: ChannelWeights = Field(default_factory=ChannelWeights)
    demographics: Demographics = Field(default_factory=lambda: Demographics(
        income_dist=IncomeDistribution(low=0.3, medium=0.5, high=0.2)
    ))

    @classmethod
    def default_demographics(cls) -> Demographics:
        return cls.model_fields["demographics"].default_factory()


# --- Cluster Config Builder

class ClusterConfigBuilder:
    @staticmethod
    def build_cluster(entry: dict) -> GeoClusterSpec:
        age_range = entry.get("age_range")
        income_dist = entry.get("income_dist")

        if (age_range is not None) != (income_dist is not None):
            raise ValueError(f"Incomplete demographics in cluster '{entry.get('name', 'unknown')}': both age_range and income_dist are required")

        demographics = (
            Demographics(age_range=age_range, income_dist=IncomeDistribution(**income_dist))
            if age_range and income_dist
            else GeoClusterSpec.default_demographics()
        )

        return GeoClusterSpec(
            name=entry["name"],
            lat=entry["lat"],
            lon=entry["lon"],
            radius_km=entry["radius_km"],
            num_stores=entry["num_stores"],
            weight=entry["weight"],
            channels=ChannelWeights(**entry["channels"]) if "channels" in entry else ChannelWeights(),
            demographics=demographics
        )

    @staticmethod
    def build_cluster_set(data: Dict[ClusterizationType, List[dict]]) -> Dict[ClusterizationType, List[GeoClusterSpec]]:
        cluster_set = {
            cluster_type: [ClusterConfigBuilder.build_cluster(entry) for entry in entries]
            for cluster_type, entries in data.items()
        }

        # Validate weights sum ≈ 1.0 for each type
        for cluster_type, clusters in cluster_set.items():
            total_weight = sum(cluster.weight for cluster in clusters)
            if not 0.99 <= total_weight <= 1.01:
                raise ValueError(f"Cluster weights for {cluster_type.value} must sum to 1.0 (got {total_weight:.2f})")

        return cluster_set

# --- Cluster Manager

class GeoClusterManager:
    def __init__(self, geo_clusters: List[GeoClusterSpec]):
        """Initialize with a list of GeoClusterSpec instances."""
        self.geo_clusters = {cluster.name: cluster for cluster in geo_clusters}

    def assign_to_region(self, latitude: float, longitude: float) -> str:
        """Assign a point to the nearest region."""
        closest_region = None
        closest_distance = float("inf")

        for region_name, cluster in self.geo_clusters.items():
            distance = haversine(latitude, longitude, cluster.lat, cluster.lon)
            if distance < closest_distance:
                closest_distance = distance
                closest_region = region_name

        return closest_region


# Cluster Data Template
BASE_CLUSTERS = {
    "LATAM": {
        "lat": -23.5505, "lon": -46.6333, "radius_km": 100, "num_stores": 5
    },
    "EMEA": {
        "lat": 48.8566, "lon": 2.3522, "radius_km": 80, "num_stores": 3
    },
    "APAC": {
        "lat": 35.6895, "lon": 139.6917, "radius_km": 60, "num_stores": 4
    }
}

DEMOGRAPHICS = {
    "low_income": {"low": 0.7, "medium": 0.2, "high": 0.1},
    "middle_income": {"low": 0.3, "medium": 0.6, "high": 0.1},
    "high_income": {"low": 0.1, "medium": 0.3, "high": 0.6},
    "mixed": {"low": 0.4, "medium": 0.4, "high": 0.2}
}

CHANNELS = {
    "online_focused": {"online": 0.8, "in_store": 0.2},  # High online preference
    "balanced": {"online": 0.5, "in_store": 0.5},        # Balanced online vs store preference
    "digital_leaning": {"online": 0.6, "in_store": 0.4},  # Digital-first, but still some in-store
    "traditional": {"online": 0.3, "in_store": 0.7},     # Store-first, low online preference
}

CLUSTER_DATA = {
    ClusterizationType.GEOGRAPHIC_EXPANSION: [
        {**BASE_CLUSTERS["LATAM"], "name": "LATAM", "weight": 0.5, "channels": CHANNELS["online_focused"], "income_dist": DEMOGRAPHICS["mixed"], "age_range": [20, 65]},
        {**BASE_CLUSTERS["EMEA"], "name": "EMEA", "weight": 0.3, "channels": CHANNELS["balanced"], "income_dist": DEMOGRAPHICS["middle_income"], "age_range": [25, 60]},
        {**BASE_CLUSTERS["APAC"], "name": "APAC", "weight": 0.2, "channels": CHANNELS["digital_leaning"], "income_dist": DEMOGRAPHICS["low_income"], "age_range": [18, 55]}
    ],

    ClusterizationType.ECONOMIC_SEGMENTS: [
        {**BASE_CLUSTERS["LATAM"], "name": "LATAM", "weight": 0.5, "channels": CHANNELS["online_focused"], "income_dist": DEMOGRAPHICS["low_income"], "age_range": [20, 65]},
        {**BASE_CLUSTERS["EMEA"], "name": "EMEA", "weight": 0.3, "channels": CHANNELS["balanced"], "income_dist": DEMOGRAPHICS["middle_income"], "age_range": [25, 60]},
        {**BASE_CLUSTERS["APAC"], "name": "APAC", "weight": 0.2, "channels": CHANNELS["digital_leaning"], "income_dist": DEMOGRAPHICS["mixed"], "age_range": [18, 55]}
    ],

    ClusterizationType.MARKET_MATURITY: [
        {**BASE_CLUSTERS["LATAM"], "name": "LATAM", "weight": 0.5, "channels": CHANNELS["traditional"], "income_dist": DEMOGRAPHICS["low_income"], "age_range": [20, 65]},
        {**BASE_CLUSTERS["EMEA"], "name": "EMEA", "weight": 0.3, "channels": CHANNELS["balanced"], "income_dist": DEMOGRAPHICS["middle_income"], "age_range": [25, 60]},
        {**BASE_CLUSTERS["APAC"], "name": "APAC", "weight": 0.2, "channels": CHANNELS["digital_leaning"], "income_dist": DEMOGRAPHICS["mixed"], "age_range": [18, 55]}
    ],

    # Lifestyle Segments
    ClusterizationType.LIFESTYLE_SEGMENTS: [
        {
            "name": "Minimalists",
            "lat": 40.7128, "lon": -74.0060, "radius_km": 20,
            "num_stores": 2, "weight": 0.2,
            "channels": CHANNELS["online_focused"],
            "income_dist": DEMOGRAPHICS["mixed"],
            "age_range": [30, 60]
        },
        {
            "name": "Trend Chasers",
            "lat": 34.0522, "lon": -118.2437, "radius_km": 30,
            "num_stores": 4, "weight": 0.3,
            "channels": CHANNELS["online_focused"],
            "income_dist": DEMOGRAPHICS["low_income"],
            "age_range": [18, 35]
        },
        {
            "name": "Deal Hunters",
            "lat": 41.8781, "lon": -87.6298, "radius_km": 25,
            "num_stores": 3, "weight": 0.25,
            "channels": CHANNELS["balanced"],
            "income_dist": DEMOGRAPHICS["middle_income"],
            "age_range": [25, 50]
        },
        {
            "name": "Eco-Conscious",
            "lat": 37.7749, "lon": -122.4194, "radius_km": 15,
            "num_stores": 2, "weight": 0.15,
            "channels": CHANNELS["online_focused"],
            "income_dist": DEMOGRAPHICS["low_income"],
            "age_range": [20, 45]
        },
        {
            "name": "Omni Explorers",
            "lat": 47.6062, "lon": -122.3321, "radius_km": 10,
            "num_stores": 1, "weight": 0.1,
            "channels": CHANNELS["balanced"],
            "income_dist": DEMOGRAPHICS["middle_income"],
            "age_range": [28, 55]
        }
    ],

    # Time-sensitive Segments (specific times of year when spending spikes)
    ClusterizationType.TIME_SENSITIVE: [
        {
            "name": "Holiday Shoppers",
            "lat": 48.8566, "lon": 2.3522, "radius_km": 60,
            "num_stores": 5, "weight": 0.4,
            "channels": CHANNELS["online_focused"],
            "income_dist": DEMOGRAPHICS["mixed"],
            "age_range": [20, 50]
        },
        {
            "name": "Back-to-School Shoppers",
            "lat": 37.7749, "lon": -122.4194, "radius_km": 80,
            "num_stores": 6, "weight": 0.6,
            "channels": CHANNELS["digital_leaning"],
            "income_dist": DEMOGRAPHICS["low_income"],
            "age_range": [18, 40]
        }
    ],

    # Age-Specific Segments (targeting by age group for product/marketing alignment)
    ClusterizationType.AGE_SPECIFIC: [
        {
            "name": "Gen Z",
            "lat": 34.0522, "lon": -118.2437, "radius_km": 50,
            "num_stores": 3, "weight": 0.15,
            "channels": CHANNELS["online_focused"],
            "income_dist": DEMOGRAPHICS["low_income"],
            "age_range": [18, 24]
        },
        {
            "name": "Millennials",
            "lat": 40.7306, "lon": -73.9352, "radius_km": 40,
            "num_stores": 4, "weight": 0.5,
            "channels": CHANNELS["balanced"],
            "income_dist": DEMOGRAPHICS["middle_income"],
            "age_range": [25, 40]
        },
        {
            "name": "Baby Boomers",
            "lat": 51.5074, "lon": -0.1278, "radius_km": 70,
            "num_stores": 2, "weight": 0.35,
            "channels": CHANNELS["traditional"],
            "income_dist": DEMOGRAPHICS["middle_income"],
            "age_range": [55, 75]
        },
    ]
}

cluster_config = ClusterConfigBuilder.build_cluster_set(CLUSTER_DATA)

# ---------------------------
# Utility functions
# ---------------------------

def random_date_weighted(start_date, end_date):
    """Generate random weight-based dates"""
    weights_by_month = {
        1: 0.5, 2: 0.8, 3: 1.0, 12: 2.0,
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
# ----------------------------# 

# === Protocols ===
class DataSink(Protocol):
    @property
    @abstractmethod
    def supports_format_selection(self) -> bool:
        ...

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        ...

    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        ...

# === Base Context ===

class SinkContext(ABC):
    @abstractmethod
    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        pass

# === File Formats ===

class FileFormatHandler(ABC):
    @abstractmethod
    def write(self, df: pl.DataFrame, path: Path):
        pass
    
    @abstractmethod
    def read(self, path: Path) -> pl.DataFrame:
        pass

class CSVHandler(FileFormatHandler):
    def write(self, df: pl.DataFrame, path: Path):
        df.write_csv(path)
    
    def read(self, path: Path) -> pl.DataFrame:
        return pl.read_csv(path)

class ExcelHandler(FileFormatHandler):
    def write(self, df: pl.DataFrame, path: Path):
        df.to_pandas().to_excel(path, index=False)

    def read(self, path: Path) -> pl.DataFrame:
        return pl.from_pandas(pd.read_excel(path))

class ParquetHandler(FileFormatHandler):
    def write(self, df: pl.DataFrame, path: Path):
        df.write_parquet(path)
    
    def read(self, path: Path) -> pl.DataFrame:
        return pl.read_parquet(path)

class JSONHandler(FileFormatHandler):
    def write(self, df: pl.DataFrame, path: Path):
        # Convert any datetime.date or datetime.datetime to string format
        def date_converter(obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.strftime("%Y-%m-%d")
            raise TypeError("Type not serializable")

        # Convert DataFrame to a list of dictionaries and serialize
        with open(path, 'w') as f:
            json.dump(df.to_dicts(), f, indent=2, default=date_converter)
    
    def read(self, path: Path) -> pl.DataFrame:
        with open(path, 'r') as f:
            data = json.load(f)
        return pl.DataFrame(data)

class JSONLHandler(FileFormatHandler):
    def write(self, df: pl.DataFrame, path: Path):
        df.write_ndjson(path)
    
    def read(self, path: Path) -> pl.DataFrame:
        return pl.read_ndjson(path)

class FeatherHandler(FileFormatHandler):
    def write(self, df: pl.DataFrame, path: Path):
        df.write_ipc(path)
    
    def read(self, path: Path) -> pl.DataFrame:
        return pl.read_ipc(path)

FORMAT_HANDLERS = {
    'csv': CSVHandler(),
    'parquet': ParquetHandler(),
    'jsonl': JSONLHandler(),
    'json': JSONHandler(),
    'feather': FeatherHandler(),
    'xls': ExcelHandler(),
    'xlsx': ExcelHandler(),
}

AVAILABLE_FORMATS = tuple(FORMAT_HANDLERS.keys())

# === Filesystem Sinks ===

SINK_REGISTRY = {}

def register_sink(scheme: str, file_formats: Tuple[str] = ()):
    """
    Registers a sink class for the given scheme and file formats.

    Args:
        scheme (str): The scheme (e.g., 'file', 'cloud', etc.).
        file_formats (tuple): A tuple of supported file formats (e.g., 'csv', 'parquet').
    
    This decorator registers the given class for the specified formats in the SINK_REGISTRY.
    """
    def decorator(cls):
        for file_format in file_formats:
            if (scheme, file_format) not in SINK_REGISTRY:
                SINK_REGISTRY[(scheme, file_format)] = cls
            else:
                raise ValueError(f"Sink already registered for {scheme} and format {file_format}")
        return cls
    return decorator

@register_sink('file', AVAILABLE_FORMATS)
class FileSink(DataSink):
    def __init__(self, base_path: str, file_format: str = "csv"):
        self.base_path = Path(base_path)
        self.file_format = file_format
        self.format_handler = FORMAT_HANDLERS.get(file_format)

        if not self.format_handler:
            raise ValueError(f"Unsupported file format: {file_format}")

    @property
    def supports_format_selection(self) -> bool:
        return True

    def write(self, data: dict, destination: Optional[str] = None):
        for name, df in data.items():
            path = self.base_path / destination / f"{name}.{self.file_format}"
            
            os.makedirs(path.parent, exist_ok=True)
            
            self.format_handler.write(df, path)

    def read(self, source: Optional[str] = None) -> dict:
        result = {}
        path = self.base_path / source if source else self.base_path
        for file in path.glob(f"*.{self.file_format}"):
            name = file.stem
            result[name] = self.format_handler.read(file)
        return result

# === Database Sinks ===
@register_sink('duckdb')
class DuckDBSink(DataSink):
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

    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        raise NotImplementedError("Reading from DuckDB not implemented yet")

@register_sink('mysql')
class MySQLSink(DataSink):
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, future=True)

    def write(self, data: Dict[str, pl.DataFrame], database: Optional[str] = None) -> None:
        with self.engine.begin() as conn:
            if database:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{database}`"))
                conn.execute(text(f"USE `{database}`"))
            for name, df in data.items():
                df.write_database(name, conn, if_exists="replace")

    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        raise NotImplementedError("Reading from MySQL not implemented")

@register_sink('postgresql')
class PostgreSQLSink(DataSink):
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, future=True)

    def write(self, data: Dict[str, pl.DataFrame], schema: Optional[str] = None) -> None:
        with self.engine.begin() as conn:
            if schema:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            for name, df in data.items():
                full_name = f"{schema}.{name}" if schema else name
                df.write_database(full_name, conn, if_exists="replace")

    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        raise NotImplementedError("Reading from PostgreSQL not implemented")

@register_sink('sqlite')
class SQLiteSink(DataSink):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None) -> None:
        engine = self.engine if not destination else create_engine(f"sqlite:///{destination}", future=True)
        with engine.begin() as conn:
            for name, df in data.items():
                df.write_database(name, conn, if_exists="replace")

    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        raise NotImplementedError("Reading from SQLite not implemented")

# === Cloud Sinks ===

class CloudFileSink(DataSink):
    def __init__(self, file_format: str = "parquet", compression: Optional[str] = "snappy"):
        self.file_format = file_format
        self.compression = compression
        # Fetch the appropriate format handler from the registry
        self.format_handler = FORMAT_HANDLERS.get(file_format)
        if not self.format_handler:
            raise ValueError(f"Unsupported format: {file_format}")

    @property
    def supports_format_selection(self) -> bool:
        return True

    def _get_bytes(self, df: pl.DataFrame) -> bytes:
        buf = io.BytesIO()
        # Use the format handler to write data to a byte buffer
        self.format_handler.write(df, buf)
        return buf.getvalue()

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None):
        """
        Write data to the specified destination.
        If no destination is provided, use the default path.
        """
        raise NotImplementedError("This method should be implemented in subclass")

    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        raise NotImplementedError("Reading from cloud sinks not implemented")

@register_sink('s3', AVAILABLE_FORMATS)
class S3Sink(CloudFileSink):
    def __init__(self, bucket: str, prefix: str = "", file_format: str = "parquet", compression: str = "snappy"):
        super().__init__(file_format, compression)
        self.bucket = bucket
        self.default_prefix = prefix.rstrip("/") + "/" if prefix else ''
        self.s3 = boto_client("s3")

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None):
        """
        Write data to S3.
        The destination should include the path, such as 'raw' or 'mart'.
        """
        prefix = destination.rstrip("/") + "/" if destination else self.default_prefix
        for name, df in data.items():
            key = f"{prefix}{name}.{self.file_format}"
            self.s3.upload_fileobj(io.BytesIO(self._get_bytes(df)), self.bucket, key)

    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        raise NotImplementedError("Reading from S3 not implemented")


@register_sink('gs', AVAILABLE_FORMATS)
class GCSSink(CloudFileSink):
    def __init__(self, path: str, file_format: str = "parquet", compression: str = "snappy"):
        super().__init__(file_format, compression)
        self.default_path = path.rstrip("/") + "/" if path else ''
        self.fs = GCSFileSystem()

    def write(self, data: Dict[str, pl.DataFrame], destination: Optional[str] = None):
        """
        Write data to GCS.
        The destination should include the path, such as 'raw' or 'mart'.
        """
        path = destination.rstrip("/") + "/" if destination else self.default_path
        for name, df in data.items():
            with self.fs.open(f"{path}{name}.{self.file_format}", "wb") as f:
                f.write(self._get_bytes(df))

    def read(self, source: Optional[str] = None) -> Dict[str, pl.DataFrame]:
        raise NotImplementedError("Reading from GCS not implemented")

def get_sink(destination: str, file_format: str) -> SinkContext:
    """
    Retrieves the correct sink based on the destination and file format.

    Args:
        destination (str): The URL or path specifying the sink location.
        file_format (str): The desired file format (e.g., 'csv', 'parquet').

    Returns:
        SinkContext: The appropriate sink context for the given destination and file format.

    Raises:
        ValueError: If no sink is registered for the specified scheme and file format.
        NotImplementedError: If the scheme is not yet supported.
    """
    # Parse the destination URI
    uri = urlparse(destination)
    scheme = uri.scheme or "file"  # default to local filesystem if no scheme

    # Check for valid scheme + file_format in the registry
    try:
        sink_class = SINK_REGISTRY[(scheme, file_format)]
    except KeyError:
        raise ValueError(f"No sink registered for scheme '{scheme}' and format '{file_format}'. Available formats for scheme '{scheme}': {', '.join([fmt for _, fmt in SINK_REGISTRY if _ == scheme])}")

    # Scheme-specific instantiation logic
    scheme_handlers = {
        "file": lambda: sink_class(base_path=uri.path, file_format=file_format),
        "s3": lambda: sink_class(
            bucket=uri.netloc, prefix=uri.path.strip("/"), file_format=file_format
        ),
        "gcs": lambda: sink_class(
            path=destination, file_format=file_format
        ),
        "": lambda: sink_class(base_path=uri.path, file_format=file_format),
        "duckdb": lambda: sink_class(db_path=uri.path),
        "mysql": lambda: sink_class(db_url=destination),
        "postgresql": lambda: sink_class(db_url=destination),
        "sqlite": lambda: sink_class(db_path=uri.path),
    }

    # Retrieve the appropriate handler, raise NotImplementedError if the scheme isn't supported
    handler = scheme_handlers.get(scheme)
    if handler:
        return handler()
    else:
        raise NotImplementedError(f"Sink scheme '{scheme}' is not yet supported.")

# ----------------------------
# Business Layer
# ----------------------------

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

@dataclass
class RetailDataSpec:
    num_customers: int = 100
    num_products: int = 50
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
        assert self.geo_clusters, "❌ You must define at least one geo_cluster"

class BaseSpecification(ABC):
    @abstractmethod
    def generate(self) -> RetailDataSpec:
        """Generate a RetailDataSpec instance."""
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate the generated specification."""
        pass

class DataGenerator(ABC):
    def __init__(self, spec: RetailDataSpec, sink: DataSink):
        self.spec = spec
        self.sink = sink
        self.data = {}
    
    def compute_metrics(self, data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        """Compute metrics from the generated data."""
        # Placeholder for metric computation logic
        # This should be implemented in subclasses
        return {}
    
    def run(self) -> Dict[str, pl.DataFrame]:
        """Generate data and return as a dictionary of DataFrames."""
        start_time = perf_counter()

        try:
            logger.info("🛠️ Generating data")
            data_dict = self.generate_data()
        except Exception as e:
            logger.error(f"❌ Error generating retail data: {e}")
            raise

        try:
            logger.info("📊 Computing retail metrics...")
            metrics_dict = self.compute_metrics(data_dict)
        except Exception as e:
            logger.error(f"❌ Error computing metrics: {e}")
            raise

        try:
            logger.info(f"📦 Writing raw data to raw`...")
            self.sink.write(data=data_dict, destination="raw")

            logger.info(f"📦 Writing metrics to mart`...")
            self.sink.write(data=metrics_dict, destination="mart")
        except Exception as e:
            logger.error(f"❌ Error writing to sink: {e}")
            raise

        elapsed = perf_counter() - start_time
        logger.info(f"🏁 Synthetic data generation complete in {elapsed:.2f}s")
        return {
            "raw": data_dict,
            "mart": metrics_dict
        }

class RetailDataGenerator(DataGenerator):
    def __init__(self, spec: RetailDataSpec, sink: DataSink):
        super().__init__(spec, sink)

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

            # Access channel preference from nested ChannelWeights model
            channel = random.choices(
                population=["online", "in_store"],
                weights=[cluster.channels.online, cluster.channels.in_store]
            )[0]

            # Access demographics safely
            if not cluster.demographics:
                raise ValueError(f"❌ Cluster '{cluster.name}' missing demographics.")

            age = random.randint(*cluster.demographics.age_range)
            income = random.choices(
                population=["low", "medium", "high"],
                weights=[
                    cluster.demographics.income_dist.low,
                    cluster.demographics.income_dist.medium,
                    cluster.demographics.income_dist.high
                ]
            )[0]

            customers.append({
                "customer_id": self.fake.uuid4(),
                "registration_date": registration,
                "age": age,
                "income": income,
                "channel_pref": channel,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "region": cluster.name,
            })

        logger.info(f"✅ {len(customers)} customers generated across {len(clusters)} geo clusters.")

        return pl.DataFrame(customers)

    def generate_products(self, suppliers: Union[List[dict], pl.DataFrame]) -> pl.DataFrame:
        categories = self.spec.product_categories
        seasonality = self.spec.product_seasonality

        if not isinstance(suppliers, list):
            suppliers = suppliers.to_dicts()

        products = []
        for i in range(self.spec.num_products):
            category = random.choice(categories)
            seasonality = random.choice(seasonality)
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

            # Values between $5 and $200
            price = round(random.uniform(5.0, 200.0), 2)

            products.append({
                "product_id": product_id,
                "name": self.fake.word().capitalize(),
                "category": category,
                "seasonality": seasonality,
                "price": price,
                "suppliers": json.dumps(supplier_links)
            })

        products_df = pl.DataFrame(products)
        logger.info(f"✅ {len(products_df)} products generated.")
        return products_df

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

    def generate_orders_and_transactions(
        self,
        customers: pl.DataFrame,
        stores: pl.DataFrame,
        products: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:

        custs = customers.to_dicts()
        prods = products.to_dicts()

        price_map = {
            p['product_id']: p.get('price', 1.0) or 1.0
            for p in prods
        }

        stores_map = {s['store_id']: s for s in stores.to_dicts()}
        end = parse_date(self.spec.data_end)

        def generate_for_customer(cust):
            lat, lng = cust['latitude'], cust['longitude']
            region = self.geo_cluster_manager.assign_to_region(lat, lng)
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

                store_lat, store_lng = stores_map[sid]['latitude'], stores_map[sid]['longitude']
                dist = calculate_distance(lat, lng,store_lat, store_lng)

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

        # Extract and normalize suppliers by product
        suppliers = {
            row['product_id']: [
                sup for sup in row.get('suppliers', [])
            ]
            for row in products.to_dicts()
        }

        # Initializes storage based on product-store combinations
        for row in transactions.select(['product_id', 'store_id']).unique().to_dicts():
            stock[(row['product_id'], row['store_id'])] = random.randint(*self.spec.initial_stock_range)

        # Process transactions chronologically
        for txn in sorted(transactions.to_dicts(), key=lambda x: x['transaction_date']):
            key = (txn['product_id'], txn['store_id'])
            before = stock[key]
            sold = min(before, txn['quantity'])
            stock[key] = before - sold

            # Conditional replenishment
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

    def generate_data(self) -> Dict[str, pl.DataFrame]:
        customers=self.generate_customers(); 
        stores=self.generate_stores(); 
        suppliers=self.generate_suppliers()
        products=self.generate_products(suppliers)
        transactions, orders = self.generate_orders_and_transactions(customers, stores, products)
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic retail data and metrics.")

    parser.add_argument("-c", "--num_customers", type=int, 
                        default=5000, help="Number of customers to generate")
    parser.add_argument("-p", "--num_products", type=int, 
                        default=10000, help="Number of products to generate"
    )
    parser.add_argument("-o", "--num_orders", type=int, 
                        default=10000, help="Number of orders to generate")
    parser.add_argument("-s", "--seed", type=int, 
                        default=42, help="Random seed for reproducibility")
    parser.add_argument("-d", "--destination", type=str, 
                        default="./data", help="Data sink URI (e.g. s3://bucket or ./data)")
    parser.add_argument("-f", "--format", type=str, choices=AVAILABLE_FORMATS,
                        default="parquet", help="File format to use for data output")
    parser.add_argument("-t", "--clusterization_type", type=str,
                        choices=[e.name for e in ClusterizationType],
                        default=ClusterizationType.GEOGRAPHIC_EXPANSION.name,
                        help="Specify the clusterization type")

    return parser.parse_args()


def build_retail_spec(args: argparse.Namespace) -> RetailDataSpec:
    cluster_type = ClusterizationType[args.clusterization_type]
    geo_clusters = cluster_config[cluster_type]
    return RetailDataSpec(
        num_customers=args.num_customers,
        num_products=args.num_products,
        seed=args.seed,
        destination=args.destination,
        geo_clusters=geo_clusters
    )

def main():
    args = parse_args()
    
    spec = build_retail_spec(args)
    sink = get_sink(args.destination, args.format)
    
    generator = RetailDataGenerator(spec, sink)
    generator.run()

if __name__ == '__main__':
    main()
