from typing import Annotated

from langgraph.config import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from typing import Annotated, Sequence, TypedDict


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

def collect_graph_states(graph, inputs, config=None):
    """
    Collects the full state of the graph after each super-step into a list.

    Args:
        graph: The compiled LangGraph instance.
        inputs: The initial input to the graph.
        config: Optional configuration dictionary.

    Returns:
        A list of state dictionaries representing the graph's state after each step.
    """
    states = []
    for state in graph.stream(inputs, config=config, stream_mode="values"):
        states.append(state)
    return states

def save_memory(memory: str, *, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]) -> str:
    '''Save the given memory for the current user.'''
    # This is a **tool** the model can use to save memories to storage
    user_id = config.get("configurable", {}).get("user_id")
    namespace = ("memories", user_id)
    store.put(namespace, f"memory_{len(store.search(namespace))}", {"data": memory})
    return f"Saved memory: {memory}"

def prepare_model_inputs(
    initial_prompt: str,
    state: AgentState, 
    config: RunnableConfig, 
    store: BaseStore
):
    # Retrieve user memories and add them to the system message
    # This function is called **every time** the model is prompted. It converts the state to a prompt
    user_id = config.get("configurable", {}).get("user_id")
    namespace = ("memories", user_id)
    memories = [m.value["data"] for m in store.search(namespace)]
    system_msg = f"{initial_prompt}. User memories: {', '.join(memories)}"
    return [{"role": "system", "content": system_msg}] + state["messages"]
