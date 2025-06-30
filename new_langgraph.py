from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from typing import List

# Assuming your custom LLM class
class MyLLM:
    def __init__(self, **params):
        # Initialization code
        pass

    async def get_response(self, message, thread_id):
        # Custom async call to your model
        return f"Model response to '{message}' in thread '{thread_id}'"

# Example tool definition with @tool decorator from LangChain
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary."""
    return f"Wikipedia result for '{query}'"

# Define state for LangGraph
class GraphState:
    message: str
    thread_id: str
    result: str = ""

# Node: call your custom LLM
async def call_llm(state: GraphState) -> GraphState:
    llm = MyLLM()
    response = await llm.get_response(state.message, state.thread_id)
    state.result = response
    return state

# Node: call your LangChain tool
async def call_tool(state: GraphState) -> GraphState:
    tool_result = search_wikipedia(state.message)
    state.result = tool_result
    return state

# Condition node to decide next step
def decide_node(state: GraphState):
    if state.message.lower().startswith("search"):
        return "call_tool"
    return "call_llm"

# Create LangGraph
workflow = StateGraph(GraphState)
workflow.add_node("call_llm", call_llm)
workflow.add_node("call_tool", call_tool)
workflow.set_conditional_entry_point(decide_node)
workflow.add_edge("call_llm", END)
workflow.add_edge("call_tool", END)

# Compile the graph into an executable app
app = workflow.compile()

# Example usage
async def run_example():
    initial_state = GraphState(message="Search Python programming", thread_id="12345")
    final_state = await app.ainvoke(initial_state)
    print(final_state.result)
