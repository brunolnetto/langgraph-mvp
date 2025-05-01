from typing import List
from src.ai.schemas import KPI
from src.repositories.kpi import FakeKPIRepository

repo = FakeKPIRepository()

def fetch_kpis() -> List[KPI]:
    """
    Retrieve all Key Performance Indicators (KPIs) from the repository.

    This function accesses the KPI repository to fetch a list of all available KPIs.

    Returns:
        List[KPI]: A list containing all KPI objects retrieved from the repository.

    Raises:
        RepositoryError: If there is an issue accessing the KPI repository.
    """
    return repo.fetch_all()

def get_kpi_by_metric(metric_name: str) -> KPI:
    """
    Retrieve a specific KPI by its metric name.

    This function searches the KPI repository for a KPI that matches the provided metric name.

    Args:
        metric_name (str): The name of the metric to search for.

    Returns:
        KPI: The KPI object corresponding to the specified metric name.

    Raises:
        ValueError: If no KPI with the specified metric name is found.
        RepositoryError: If there is an issue accessing the KPI repository.
    """
    return repo.get_by_metric(metric_name)
