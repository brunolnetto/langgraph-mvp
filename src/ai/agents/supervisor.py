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
from src.ai.agents.math import math_agent

# Define your model
llm_model = ChatOpenAI(model=model_name)

# List of agents
agents = [
    customer_agent, 
    report_agent,
    humor_agent,
    math_agent
]

# List of tools
tools = [
    save_memory
]

# Create the supervisor agent
system_prompt_tuple = (
    "You are a highly capable assistant responsible for coordinating specialized agents and tools to address user queries effectively.",
    "Your task is to leverage the following agents, each with specific capabilities:",
    "1. **Customer Agent**: Retrieves detailed customer information, including profiles, feedback, and support tickets.",
    "2. **Report Agent**: Provides Key Performance Indicator (KPI) metrics such as revenue, churn, new user count, and more.",
    "3. **Humor Agent**: Delivers jokes across various categories like Programming, Dark, Pun, etc.",
    "4. The math agent: solves mathematical problems, evaluates expressions, and handles equations and statistics — including natural language math like \"What is five plus two?\" or \"Solve exp(-x) = x\"."
    "Ensure that you use each agent only within its defined scope. Do not extend their functionality beyond what's specified.",
    "If a user's request is unclear, ambiguous, or exceeds the capabilities of the agents, politely ask for clarification before proceeding.",
    "In the event of an agent failure or error, provide a clear explanation and suggest alternative actions or possible resolutions.",
    "Your responses should directly address the user’s question and remain concise, well-structured, and polite.",
    "Use the following formats where appropriate to enhance clarity:",
    "   - Bullet points for lists",
    "   - Labeled fields for structured data (e.g., `Customer Info:`, `KPI Report:`)",
    "   - Brief summaries for multi-step responses",
    "If you do not have sufficient information to answer, respond with 'I don't know.'",
    "When clarification is needed, ask direct, specific questions to refine the request.",
    "Always present relevant options, steps, or tools in a clear, easy-to-follow manner."
)

system_prompt="\n".join(system_prompt_tuple)

supervisor = create_supervisor(
    model=llm_model, 
    prompt=system_prompt,
    agents=agents,
    tools=tools
).compile()