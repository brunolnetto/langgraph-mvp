from langgraph.graph import StateGraph
from src.ai.agents.supervisor import supervisor

# Define the state structure
class GraphState(TypedDict):
    messages: list
    next: str

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]

    if "FINISH:" in last_message.content:
        return "FINISH"
    else:
        return state["next"]


# Create the LangGraph workflow
workflow = StateGraph(GraphState)

# Add the supervisor node
workflow.add_node("supervisor", supervisor)

# Set the entry point
workflow.set_entry_point("supervisor")

# Compile the workflow
app = workflow.compile()
