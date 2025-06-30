# jira_agent.py

import operator
from typing import TypedDict, Annotated

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage

# Import your tools and the custom LLM wrapper
import jira_services
from llm_wrapper import InternalThreadedChatModel # Assuming this is your final wrapper

# --- 1. Define the State for our Graph ---
# This will be the memory of our agent. It's a list of messages.
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# --- 2. Instantiate Tools and LLM ---
tools = [
    jira_services.get_jira_project_schema,
    jira_services.get_my_jira_issues,
    jira_services.create_jira_issue,
]

# This is where you would pass your application settings
mock_settings = {} 
# Use your custom chat model wrapper
llm = InternalThreadedChatModel(settings=mock_settings)

# The critical step: Bind the tools to the LLM.
# This tells the LLM about the tools in a standardized way,
# making it much more likely to use them correctly.
model_with_tools = llm.bind_tools(tools)


# --- 3. Define the Graph Nodes and Edges ---

def should_continue(state: AgentState) -> str:
    """
    Conditional Edge: This function decides what to do next.
    If the LLM's last response has tool calls, we route to the 'action' node.
    Otherwise, we are done and route to END.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

def call_model(state: AgentState):
    """
    Node: This function invokes the LLM with the current state.
    The response from the LLM is added to the list of messages.
    """
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# We use the pre-built ToolNode for executing tools. It's robust and simple.
tool_node = ToolNode(tools)


# --- 4. Assemble and Compile the Graph ---
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entry point for the graph
workflow.set_entry_point("agent")

# Add the conditional logic for routing
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# Add the edge that loops back from the tool execution to the agent
workflow.add_edge("action", "agent")

# Compile the graph into a runnable object. This is our final agent.
app_graph = workflow.compile()
