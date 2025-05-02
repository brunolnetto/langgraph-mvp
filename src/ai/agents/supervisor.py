# src/ai/agents/supervisor.py
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI

from src.ai.config import model_name
from src.ai.utils import save_memory
from src.ai.agents.customer import customer_agent 
from src.ai.agents.report import report_agent
from src.ai.agents.humor import humor_agent

# Define your model
llm_model = ChatOpenAI(model=model_name)

# List of agents
agents = [
    customer_agent, 
    report_agent,
    humor_agent
]

# List of tools
tools = [
    save_memory
]

# Create the supervisor agent
system_prompt_tuple = (
    "You are a helpful assistant.",
    "You will be provided with a list of agents and tools.",
    "You will be asked to coordinate between them.",
    "You can use: ",
    "1. the customer agent for retrieval of profile, feedback and support tickets.",
    "2. the report agent for KPI metrics.",
    "3. the joker agent for jokes.",
    "Be polite and concise.",
    "If you don't know the answer, say 'I don't know'.",
    "If you need to ask a question, ask it.",
    "If you need to provide a list of options, summary, list of steps or tools, provide them.",
)

system_prompt="\n".join(system_prompt_tuple)

supervisor = create_supervisor(
    model=llm_model, 
    prompt=system_prompt,
    agents=agents,
    tools=tools
).compile()