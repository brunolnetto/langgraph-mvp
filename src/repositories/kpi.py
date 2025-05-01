from typing import List
from datetime import datetime
from src.ai.schemas import KPI

class FakeKPIRepository:
    def fetch_all(self) -> List[KPI]:
        return [
            KPI(
                metric="Revenue",
                value=123456.78,
                unit="USD",
                target=150000.00,
                trend="up",
                last_updated=datetime.now().isoformat()
            ),
            KPI(
                metric="New Users",
                value=321,
                unit="users",
                target=500,
                trend="down",
                last_updated=datetime.now().isoformat()
            ),
            KPI(
                metric="Churn Rate",
                value=4.2,
                unit="%",
                target=3.0,
                trend="stable",
                last_updated=datetime.now().isoformat()
            )
        ]

    def get_by_metric(self, metric_name: str) -> KPI:
        all_kpis = self.fetch_all()
        for kpi in all_kpis:
            if kpi.metric.lower() == metric_name.lower():
                return kpi
        raise ValueError(f"KPI '{metric_name}' not found.")
