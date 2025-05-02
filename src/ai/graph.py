from typing import TypedDict

from langgraph.graph import StateGraph

from src.ai.agents.supervisor import supervisor

# Define the state structure
class GraphState(TypedDict):
    messages: list
    next: str


# Crie o fluxo de trabalho com StateGraph
workflow = StateGraph(GraphState)

# Adicione o nó do supervisor
workflow.add_node("supervisor", supervisor)

# Defina o ponto de entrada
workflow.set_entry_point("supervisor")

# Compile o fluxo de trabalho
corp_workflow = workflow.compile() 