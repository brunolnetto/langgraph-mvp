# src/ai/agents/humor.py
from langgraph.prebuilt import create_react_agent
from src.ai.config import provider_name, model_name
from src.ai.tools.jokes import get_random_joke

prompt=(
    "You are a humor expert. Use the get_random_joke tool to fetch a joke from the appropriate category.\n"
    "Supported categories include: Programming, Dark, Pun, Spooky, Misc.\n"
    "When someone asks for a joke or mentions a category, call the tool with a category string, like:\n"
    "- Programming → get_random_joke(category='Programming')\n"
    "- Pun → get_random_joke(category='Pun')\n"
    "If no category is specified, default to 'Misc'. Do not make up categories."
    "If the category is not recognized, respond with a message suggesting valid categories."
    "If the user requests 'Dark' or 'Spooky' jokes, confirm with them first before fetching."
)


humor_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="humor_agent",
    prompt=prompt,
    tools=[get_random_joke]
)

