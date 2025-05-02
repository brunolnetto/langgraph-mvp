# src/ai/agents/humor.py
from langgraph.prebuilt import create_react_agent
from src.ai.config import provider_name, model_name
from src.ai.tools.jokes import get_random_joke

humor_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="humor_agent",
    prompt=(
        "You are a humor expert. You can tell jokes from different categories like "
        "Programming, Dark, Pun, Spooky, Misc. When asked, call the get_random_joke tool "
        "with the appropriate category string."
    ),
    tools=[get_random_joke]
)

