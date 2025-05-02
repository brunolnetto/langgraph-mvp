# src/ai/agents/report.py
from langgraph.prebuilt import create_react_agent

from src.ai.tools.kpi import (
    fetch_kpis, get_kpi_by_metric, list_available_kpis,
)
from src.ai.config import provider_name, model_name

# Create the agent with the model
report_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="report_agent",
    prompt="Generates weekly KPI metric reports.",
    tools=[ list_available_kpis, fetch_kpis, get_kpi_by_metric ],
)
