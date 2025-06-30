# jira_agent.py

import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage

# Import your tools and the custom LLM wrapper
import jira_services
from llm_wrapper import InternalThreadedChatModel

# --- 1. Define the state for our graph ---
# This will be the memory of our agent.
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# --- 2. Instantiate Tools and LLM ---
tools = [
    jira_services.get_jira_project_schema,
    jira_services.get_my_jira_issues,
    jira_services.create_jira_issue,
]
tool_executor = ToolExecutor(tools)

# This is where you would pass your application settings
mock_settings = {} 
llm = InternalThreadedChatModel(settings=mock_settings)

# Bind the tools to the LLM. This is a critical step that allows the LLM
# to see the tools in a format it understands, making it model-agnostic.
model = llm.bind_tools(tools)


# --- 3. Define the graph nodes and edges ---

def should_continue(state: AgentState) -> str:
    """Conditional edge: decides whether to call a tool or finish."""
    last_message = state["messages"][-1]
    # If the LLM's last response has no tool calls, we're done.
    if not last_message.tool_calls:
        return "end"
    # Otherwise, we continue by calling the tool.
    return "continue"

def call_model(state: AgentState):
    """Node: invokes the LLM with the current state."""
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a dictionary with the key `messages` to update the state
    return {"messages": [response]}

def call_tool(state: AgentState):
    """Node: executes the tool calls made by the LLM."""
    last_message = state["messages"][-1]
    
    # The ToolExecutor takes a tool call and returns the tool's output
    action = last_message.tool_calls[0]
    response = tool_executor.invoke(action)
    
    # We wrap the tool's output in a ToolMessage to send back to the LLM
    tool_message = ToolMessage(content=str(response), tool_call_id=action['id'])
    
    return {"messages": [tool_message]}

# --- 4. Assemble and compile the graph ---
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Define the entry point and the edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

# Compile the graph into a runnable object. This is our final agent.
app_graph = workflow.compile()
