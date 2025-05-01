from pydantic import BaseModel
from typing import List, Optional

class KPI(BaseModel):
    metric: str
    value: float
    unit: Optional[str] = None
    target: Optional[float] = None
    trend: Optional[str] = None
    last_updated: Optional[str] = None


class CustomerProfile(BaseModel):
    id: str
    name: str
    score: int
    last_purchase: str
    email: str
    phone: str
    address: str
    purchase_history: List[dict]