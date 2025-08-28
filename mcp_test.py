import requests
import json

# --- Configuration ---
BASE_URL = "http://localhost:8000"  # Change to your server's address
MCP_ENDPOINT = f"{BASE_URL}/mcp"

# --- Headers for Streaming Request ---
stream_headers = {
    'Authorization': f'Bearer {BEARER_TOKEN}',
    'Accept': 'text/event-stream'
}

session = requests.Session()

try:
    # 1. Connect and get operations using a POST request
    print(f"--> Connecting to {MCP_ENDPOINT} with POST...")
    initial_payload = {
        "mcp_operation": "get_all_operations"
    }

    # Use POST instead of GET and include the initial payload
    initial_response = session.post(
        MCP_ENDPOINT,
        headers=stream_headers,
        json=initial_payload,
        stream=True  # Keep stream=True to handle the response
    )
    initial_response.raise_for_status() # This should now pass

    print("\n✅ Connection successful. Server Operations:")
    for line in initial_response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data:'):
                data_str = line_str[len('data:'):].strip()
                if data_str:
                    data = json.loads(data_str)
                    print(json.dumps(data, indent=2))

    # 2. Subsequent requests remain the same
    print(f"\n--> Requesting list of tools from {MCP_ENDPOINT}...")
    tool_request_payload = {
        "mcp_operation": "get_all_tools"
    }

    # For a non-streaming POST, standard JSON accept is fine
    post_headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}',
        'Accept': 'application/json'
    }
    tool_response = session.post(MCP_ENDPOINT, headers=post_headers, json=tool_request_payload)
    tool_response.raise_for_status()

    print("\n✅ Successfully retrieved tools:")
    print(json.dumps(tool_response.json(), indent=2))

except requests.exceptions.RequestException as e:
    print(f"\n❌ An error occurred: {e}")


# jira_agent.py

import time
import json
import re
from typing import TypedDict, Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

# --- Import your actual modules ---
import jira_services
from internal.api_models import InternalThreadedChatModel
from internal.llm_suite import settings # Your actual settings

# --- 1. State Definition ---
class GState(TypedDict, total=False):
    user_input: str
    route: str
    project_key: Optional[str]
    schema: Dict[str, Any]
    params: Dict[str, Any]
    result: Any
    final_answer: Optional[str]

# --- 2. Pydantic Models for Structured LLM Output ---
class RoutePick(BaseModel):
    route: Literal["find_issues", "create_issue"] = Field(..., description="The single best route to take.")
    project_key: Optional[str] = Field(None, description="The JIRA project key (e.g., 'PROJ1') if mentioned.")

class FindParams(BaseModel):
    issue_type: Optional[str] = Field(None, description="The type of issue (e.g., 'Story', 'Defect').")
    status_category: Optional[List[str]] = Field(None, description="A list of status categories (e.g., ['To Do', 'In Progress']).")
    last_updated_days: Optional[int] = Field(None, description="Filter for issues updated in the last N days.")

# --- 3. LLM and Tool Instantiation ---
llm = InternalThreadedChatModel(settings=settings)

# Helper function for structured LLM calls
async def llm_structured(prompt: str, pydantic_model: BaseModel, config: dict) -> BaseModel:
    """Calls the LLM and parses the output into a Pydantic model."""
    full_prompt = f"{prompt}\n\nReturn ONLY a valid JSON object matching the following schema:\n{pydantic_model.schema_json()}"
    
    # We need to wrap the string prompt in a list of messages for our chat model
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content=full_prompt)]

    # The _astream method returns an async generator of AIMessageChunk
    response_chunks = [chunk.content async for chunk in llm._astream(messages, configurable=config)]
    response_str = "".join(response_chunks)
    
    response_str = re.sub(r"```json\n?([\s\S]*?)\n?```", r"\1", response_str).strip()
    response_json = json.loads(response_str)
    return pydantic_model.parse_obj(response_json)

# --- 4. Graph Nodes ---
async def route_node(state: GState, config: dict) -> GState:
    """The first node in the graph. Decides which path to take."""
    prompt = f"You are a JIRA assistant. Choose exactly one route for the user request below.\nUser: {state['user_input']}\n\nRoutes:\n- find_issues: For searching or listing issues.\n- create_issue: For creating a new issue."
    decision = await llm_structured(prompt, RoutePick, config)
    state["route"] = decision.route
    state["project_key"] = decision.project_key or state.get("project_key")
    print(f"--- Router decided route: {decision.route} for project: {state['project_key']} ---")
    return state

async def ensure_schema_node(state: GState) -> GState:
    """Shared node: Fetches and caches the JIRA project schema."""
    pk = state.get("project_key") or "GAUSS"
    cache = state.get("schema", {})
    entry = cache.get(pk)
    is_stale = (not entry) or (time.time() - entry.get("_ts", 0) > 6 * 60 * 60)
    if is_stale:
        print(f"--- Schema for '{pk}' is stale. Fetching... ---")
        schema_data = jira_services.get_jira_project_schema.run({"project_key": pk})
        schema_data["_ts"] = time.time()
        cache[pk] = schema_data
        state["schema"] = cache
    return state

async def derive_find_params(state: GState, config: dict) -> GState:
    """Normalizes user input into tool parameters."""
    pk = state.get("project_key") or "GAUSS"
    schema = state["schema"][pk]
    allowed_values = {"issue_types": schema.get("issue_types", []), "status_categories": schema.get("statuses", [])}
    prompt = f"Normalize the user's request into canonical JIRA filters using ONLY these allowed values:\n{json.dumps(allowed_values)}\n\nUser Request: {state['user_input']}\n\nIf a value is not mentioned, it should be null."
    params = await llm_structured(prompt, FindParams, config)
    state["params"] = params.dict(exclude_none=True)
    print(f"--- Derived find params: {state['params']} ---")
    return state

async def run_find(state: GState) -> GState:
    """Runs the 'get_my_jira_issues' tool."""
    params_with_project = {"project_key": state.get("project_key") or "GAUSS", **state.get("params", {})}
    result = jira_services.get_my_jira_issues.run(params_with_project)
    state["result"] = result
    return state

async def summarize_result(state: GState, config: dict) -> GState:
    """Summarizes the raw tool output into a friendly response."""
    prompt = f"You are a helpful assistant. Summarize the following JIRA data for the user:\n\n{json.dumps(state['result'], indent=2)}"
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content=prompt)]
    response_chunks = [chunk.content async for chunk in llm._astream(messages, configurable=config)]
    state["final_answer"] = "".join(response_chunks)
    return state

# --- 5. Graph Assembly ---
workflow = StateGraph(GState)
workflow.add_node("router", route_node)
workflow.add_node("ensure_schema", ensure_schema_node)
workflow.add_node("derive_find_params", derive_find_params)
workflow.add_node("run_find", run_find)
workflow.add_node("summarize_result", summarize_result)
workflow.set_entry_point("router")
def pick_route(state: GState) -> str: return state["route"]
workflow.add_conditional_edges("router", pick_route, {"find_issues": "ensure_schema"})
workflow.add_edge("ensure_schema", "derive_find_params")
workflow.add_edge("derive_find_params", "run_find")
workflow.add_edge("run_find", "summarize_result")
workflow.add_edge("summarize_result", END)
app_graph = workflow.compile()


finally:
    session.close()
    print("\nSession closed.")
