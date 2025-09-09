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
    pk = state.get("project_key") or "PROJ1"
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
    pk = state.get("project_key") or "PROJ1"
    schema = state["schema"][pk]
    allowed_values = {"issue_types": schema.get("issue_types", []), "status_categories": schema.get("statuses", [])}
    prompt = f"Normalize the user's request into canonical JIRA filters using ONLY these allowed values:\n{json.dumps(allowed_values)}\n\nUser Request: {state['user_input']}\n\nIf a value is not mentioned, it should be null."
    params = await llm_structured(prompt, FindParams, config)
    state["params"] = params.dict(exclude_none=True)
    print(f"--- Derived find params: {state['params']} ---")
    return state

async def run_find(state: GState) -> GState:
    """Runs the 'get_my_jira_issues' tool."""
    params_with_project = {"project_key": state.get("project_key") or "PROJ1", **state.get("params", {})}
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
    pass

# jira_agent.py

import asyncio
import json
import time
from typing import TypedDict, Dict, Any, Optional, List, Literal

from pydantic import BaseModel, Field

# Your existing modules
import jira_services
from internal.api_models import InternalThreadedChatModel # Your simplified LLM Wrapper
from internal.llm_suite import settings # Your actual settings

# --- 1. State Definition ---
# ENHANCEMENT: Added `user_info` to pass user context through the graph.
class GState(TypedDict, total=False):
    user_input: str
    user_info: Dict[str, Any] # e.g., {"user_id": "dmitry.elsakov", "display_name": "Dmitry Elsakov"}
    route: str
    project_key: Optional[str]
    schema: Dict[str, Any]
    params: Dict[str, Any]
    result: Any
    final_answer: Optional[str]
    # For multi-turn interactions, especially in the 'create_issue' flow
    missing_fields: Optional[Dict[str, str]]
    creation_draft: Optional[Dict[str, Any]]

# --- 2. Pydantic Models for Structured LLM Output ---
class JIRARoutePick(BaseModel):
    route: Literal["find_issues", "project_details", "create_issue", "sprint_details"] = Field(..., description="The single best route to take.")
    project_key: Optional[str] = Field(None, description="The JIRA project key (e.g., 'PROJ1') if the user mentioned one.")

class JiraIssuesInput(BaseModel):
    # This remains the same as in your jira_services.py
    project_key: Optional[List[str]] = Field(None, description="A list of JIRA project keys to search within.")
    issue_type: Optional[str] = None
    status: Optional[List[str]] = None
    status_category: Optional[List[str]] = None
    assignee: Optional[str] = Field(None, description="The user's ID, or the special value 'currentUser()' for the current user.")
    last_updated_days: Optional[int] = None

class SprintDetails(BaseModel):
    sprint_id: Optional[int] = None
    sprint_name: Optional[str] = None
    sprint_state: Optional[Literal["active", "future", "closed"]] = None
    
class CreateIssueDraft(BaseModel):
    project_key: str
    issue_type: str
    summary: Optional[str]
    description: Optional[str]
    # Add other fields as needed

# --- 3. LLM and Tools Initialization ---
llm = InternalThreadedChatModel(settings=settings)
# (Your tool definitions are in jira_services.py and are correct)

# --- 4. Graph Nodes ---

async def llm_structured(prompt: str, pydantic_model: BaseModel, config: dict) -> BaseModel:
    """Helper function to get structured output from the LLM."""
    full_prompt = f"{prompt}\n\nReturn ONLY a valid JSON object matching the following Pydantic schema:\n{pydantic_model.schema_json(indent=2)}"
    messages = [HumanMessage(content=full_prompt)]
    response_chunks = [chunk.content async for chunk in llm._astream(messages, configurable=config)]
    response_str = "".join(response_chunks).strip()
    
    # Simple regex to find JSON blob
    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
    if not json_match:
        # Fallback logic can be added here
        raise ValueError(f"Could not find a JSON object in the LLM response: {response_str}")
        
    try:
        response_json = json.loads(json_match.group(0))
        return pydantic_model.model_validate(response_json)
    except (json.JSONDecodeError, ValidationError) as e:
        # Fallback or error handling
        raise ValueError(f"Failed to decode or validate LLM response JSON: {e}")

async def route_node(state: GState, config: dict) -> GState:
    """The first node in the graph. Determines which path to take."""
    # ENHANCEMENT: The router is now aware of the confirmation step.
    if state.get("confirmation_pending"):
        if "yes" in state["user_input"].lower() or "proceed" in state["user_input"].lower() or "ok" in state["user_input"].lower():
             state["route"] = "confirm_create"
             return state
        else:
             state["route"] = "cancel"
             return state

    prompt = f"""You are an expert JIRA routing assistant. Based on the user's request, choose exactly one route from the available options.

User Request: "{state['user_input']}"

Available Routes:
- find_issues: For searching, finding, or listing tickets/issues.
- project_details: For questions about a project's metadata (components, versions).
- sprint_details: For questions about sprints.
- create_issue: For requests to create a new ticket, story, or bug.
- cancel: If the user wants to stop or cancel the current operation.
"""
    decision = await llm_structured(prompt, JIRARoutePick, config)
    state["route"] = decision.route
    state["project_key"] = decision.project_key
    print(f"--- Router decided route: '{decision.route}' for project: '{decision.project_key}' ---")
    return state

async def ensure_schema_node(state: GState, config: dict) -> GState:
    """Shared node: Fetches and caches the JIRA project schema if needed."""
    project_key = state.get("project_key") or "PROJ1" # Default project
    schema_cache = state.get("schema", {})
    
    if project_key not in schema_cache: # Simple check, can be enhanced with TTL
        print(f"--- Schema for '{project_key}' not in cache. Fetching... ---")
        # Assuming get_jira_project_schema can take a single key
        schema_data = jira_services.get_jira_project_schema.run({"project_key": project_key})
        schema_cache[project_key] = schema_data
        state["schema"] = schema_cache
    return state

# --- Nodes for the "find_issues" Subgraph ---
async def derive_find_params(state: GState, config: dict) -> GState:
    """Uses the LLM to translate natural language into structured search parameters."""
    project_key = state.get("project_key") or "PROJ1"
    schema = state["schema"][project_key]
    user_info = state["user_info"]

    # ENHANCEMENT: A much more powerful prompt for parameter extraction.
    prompt = f"""You are a JIRA query assistant. Your job is to convert a user's natural language request into a structured JSON object for searching JIRA.

Use the following schema information for project '{project_key}':
- Available Issue Types: {[it['name'] for it in schema.get('issue_types', [])]}
- Available Statuses: {[s['name'] for s in schema.get('statuses', [])]}
- Available Status Categories: {[sc['name'] for sc in schema.get('status_categories', [])]}

Current user information:
- User ID: {user_info['user_id']}
- Display Name: {user_info['display_name']}

Analyze the user's request below and fill in the parameters.
- If the user says "me", "my", or "I", set 'assignee' to "currentUser()".
- Translate terms like "open", "in progress", "active" to the appropriate 'status_category' values.
- Translate timeframes like "last week" or "this month" into 'last_updated_days'.

User Request: "{state['user_input']}"
"""
    params = await llm_structured(prompt, JiraIssuesInput, config)
    state["params"] = params.model_dump(exclude_none=True)
    print(f"--- Derived find params: {state['params']} ---")
    return state

async def run_search_issues(state: GState, config: dict) -> GState:
    """Executes the JIRA search tool with the derived parameters."""
    print(f"--- Running search with params: {state['params']} ---")
    result = jira_services.get_my_jira_issues.run(state["params"])
    state["result"] = result
    return state

async def summarize_result(state: GState, config: dict) -> GState:
    """Summarizes the raw tool output into a friendly response for the user."""
    prompt = f"""You are a helpful assistant. The user asked: "{state['user_input']}"
We ran a search and got the following JSON result:
{json.dumps(state['result'], indent=2)}

Based on this data, summarize the answer in a clear, user-friendly, and concise way.
If there are no results, state that clearly.
"""
    messages = [HumanMessage(content=prompt)]
    response_chunks = [chunk.content async for chunk in llm._astream(messages, configurable=config)]
    state["final_answer"] = "".join(response_chunks)
    return state

# --- Nodes for the "create_issue" Subgraph (MODIFIED) ---
async def collect_create_fields(state: GState, config: dict) -> GState:
    """Gathers information to create an issue, asking the user if fields are missing."""
    print("--- Collecting fields for new issue... ---")
    draft = {"project_key": "PROJ1", "issue_type": "Bug", "summary": state["user_input"]}
    state["creation_draft"] = draft
    state["missing_fields"] = None 
    return state

# NEW NODE: This node prepares the confirmation message and pauses the graph.
async def prepare_confirmation_node(state: GState, config: dict) -> GState:
    """Formats the confirmation prompt for the user."""
    draft = state["creation_draft"]
    # Create a nicely formatted summary of the draft
    draft_summary = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in draft.items()])
    
    confirmation_message = f"I am ready to create the following JIRA issue:\n{draft_summary}\n\nShall I proceed? (yes/no)"
    
    state["final_answer"] = confirmation_message
    state["confirmation_pending"] = True # Set the flag
    print("--- Prepared confirmation prompt. Pausing for user input. ---")
    return state

async def run_create_issue(state: GState, config: dict) -> GState:
    """Runs the create issue tool after receiving confirmation."""
    print(f"--- User confirmed. Running create issue with draft: {state['creation_draft']} ---")
    result = jira_services.create_jira_issue.run(state["creation_draft"])
    state["result"] = result
    # Clear the flags so we don't get stuck in a loop
    state["confirmation_pending"] = False
    state["creation_draft"] = None
    return state

# NEW NODE: Handles the user cancelling the operation.
async def cancel_node(state: GState, config: dict) -> GState:
    """Handles the user cancelling the operation."""
    state["final_answer"] = "Okay, I have cancelled the operation."
    state["confirmation_pending"] = False
    state["creation_draft"] = None
    print("--- User cancelled operation. ---")
    return state

# --- 5. Graph Assembly ---
workflow = StateGraph(GState)

# Add Nodes
workflow.add_node("router", route_node)
workflow.add_node("ensure_schema", ensure_schema_node)
workflow.add_node("derive_find_params", derive_find_params)
workflow.add_node("run_search_issues", run_search_issues)
workflow.add_node("summarize_result", summarize_result)
workflow.add_node("collect_create_fields", collect_create_fields)
# NEW NODES for confirmation flow
workflow.add_node("prepare_confirmation", prepare_confirmation_node)
workflow.add_node("run_create_issue", run_create_issue)
workflow.add_node("cancel_operation", cancel_node)


# Define Edges
workflow.set_entry_point("router")

def pick_route(state: GState) -> str:
    return state["route"]

workflow.add_conditional_edges("router", pick_route, {
    "find_issues": "ensure_schema",
    "create_issue": "ensure_schema",
    # NEW: Route directly to the create tool if confirmation is given
    "confirm_create": "run_create_issue",
    "cancel": "cancel_operation",
    # Add other routes here
    "project_details": "summarize_result",
    "sprint_details": "summarize_result",
})

# Edges for 'find_issues' subgraph (Unchanged)
workflow.add_conditional_edges("ensure_schema", lambda s: "find_issues" if s["route"] == "find_issues" else "create_issue", {
    "find_issues": "derive_find_params",
    "create_issue": "collect_create_fields"
})
workflow.add_edge("derive_find_params", "run_search_issues")
workflow.add_edge("run_search_issues", "summarize_result")

# MODIFIED Edges for 'create_issue' subgraph
workflow.add_edge("collect_create_fields", "prepare_confirmation")
# After preparing confirmation, we end the graph to wait for user input.
workflow.add_edge("prepare_confirmation", END) 
# The actual creation happens after the user confirms and the router sends us to 'run_create_issue'
workflow.add_edge("run_create_issue", "summarize_result")
# If the user cancels, the graph ends.
workflow.add_edge("cancel_operation", END)

workflow.add_edge("summarize_result", END)

# Compile the graph
app_graph = workflow.compile()



    
    session.close()
    print("\nSession closed.")

# --- 2. Pydantic Models for Structured LLM Output ---
class JIRARoutePick(BaseModel):
    route: Literal["find_issues", "project_details", "create_issue", "sprint_details", "confirm_create", "cancel"] = Field(..., description="The single best route to take.")
    project_key: Optional[str] = Field(None, description="The JIRA project key (e.g., 'GAUSS') if the user mentioned one.")

# NEW/UPDATED Pydantic models for the create flow
class CreateIssueDraft(BaseModel):
    """Represents the data extracted to create a JIRA issue."""
    project_key: str
    issue_type: str = Field(..., description="The type of the issue, e.g., 'Bug', 'Story'.")
    summary: Optional[str] = Field(None, description="The summary or title of the issue.")
    description: Optional[str] = Field(None, description="The detailed description of the issue.")
    # This will hold any other custom fields extracted by the LLM
    other_fields: Dict[str, Any] = Field(default_factory=dict)

class CreateStepResult(BaseModel):
    """The structured output from the LLM after analyzing the user's creation request."""
    draft: CreateIssueDraft
    ask_user: Optional[Dict[str, str]] = Field(None, description="A dictionary where keys are the names of REQUIRED fields that are still missing, and values are user-friendly questions to ask for that information.")

async def route_node(state: GState, config: dict) -> GState:
    # ... (Your existing route_node is fine)
    if state.get("confirmation_pending"):
        if any(word in state["user_input"].lower() for word in ["yes", "proceed", "ok", "confirm", "do it"]):
             state["route"] = "confirm_create"
        else:
             state["route"] = "cancel"
        return state
    # ... your routing logic
    return state # fallback

async def ensure_schema_node(state: GState, config: dict) -> GState:
    # ... (Your existing ensure_schema_node is fine)
    return state

# --- REDESIGNED Node for the "create_issue" Subgraph ---
async def collect_create_fields(state: GState, config: dict) -> GState:
    """
    Analyzes user input against the JIRA project schema to create a draft
    and identify any missing required information.
    """
    print("--- Collecting fields for new issue... ---")
    project_key = state.get("project_key") or "GAUSS" # Default project

    # 1. Fetch the dynamic field schema for the project
    project_fields_schema = await asyncio.to_thread(
        jira_services.get_fields_list_by_project, project=project_key
    )

    # 2. Prepare a clean summary of the schema for the LLM prompt
    field_summary_for_prompt = {}
    for issue_type, fields in project_fields_schema.items():
        field_summary_for_prompt[issue_type] = {
            field_name: {
                "required": details.get("required", False),
                "type": details.get("type", "string"),
                "allowed_values": details.get("allowedValues"),
            }
            for field_name, details in fields.items()
        }

    # 3. Construct the detailed prompt for the LLM
    prompt = f"""You are an intelligent JIRA assistant responsible for creating new issues.
Your task is to analyze the user's request to fill out the necessary fields for creating a JIRA ticket.

**User's Request:**
"{state['user_input']}"

**Available Fields Schema for Project '{project_key}':**
```json
{json.dumps(field_summary_for_prompt, indent=2)}
```

**Instructions:**
1.  Read the user's request carefully.
2.  Determine the most appropriate `issue_type`. If not specified, default to 'Bug'.
3.  Fill in the `summary`, `description`, and any `other_fields` based on the user's request.
4.  After filling the draft, check if any fields marked as `"required": true` are still empty.
5.  If any required fields are missing, formulate a user-friendly question for each one and add it to the `ask_user` dictionary.
6.  Return a JSON object with the final `draft` and the `ask_user` questions.
"""
    
    # 4. Call the LLM to get the structured draft and follow-up questions
    step_result = await llm_structured(prompt, CreateStepResult, config)

    # 5. Update the graph's state
    # We combine 'other_fields' with the main draft fields
    draft_dict = step_result.draft.model_dump(exclude_none=True)
    other_fields = draft_dict.pop("other_fields", {})
    draft_dict.update(other_fields)
    
    state["creation_payload"] = draft_dict
    state["missing_fields"] = step_result.ask_user
    
    print(f"--- Collected Draft: {state['creation_payload']} ---")
    if state["missing_fields"]:
        print(f"--- Missing Fields: {state['missing_fields']} ---")
        
    return state


async def ask_user_for_info_node(state: GState, config: dict) -> GState:
    """
    If fields are missing, this node formats them into a single message to the user
    and pauses the graph.
    """
    questions = state["missing_fields"]
    formatted_questions = "\n".join(f"- {q}" for q in questions.values())
    
    state["final_answer"] = f"I need a bit more information to create the issue:\n{formatted_questions}"
    # We set a flag or route to indicate we are waiting for the user's answers
    state["route"] = "awaiting_user_details" # The router will need to handle this
    print("--- Asking user for more information. Pausing graph. ---")
    return state


# ... (The rest of your nodes: prepare_confirmation_node, run_create_issue, etc. are fine) ...
async def prepare_confirmation_node(state: GState, config: dict) -> GState:
    # ...
    return state
async def run_create_issue(state: GState, config: dict) -> GState:
    # ...
    return state
async def cancel_node(state: GState, config: dict) -> GState:
    # ...
    return state
async def summarize_result(state: GState, config: dict) -> GState:
    # ...
    return state


# --- 5. Graph Assembly ---
workflow = StateGraph(GState)

# Add Nodes
workflow.add_node("router", route_node)
workflow.add_node("ensure_schema", ensure_schema_node)
# workflow.add_node("derive_find_params", derive_find_params)
# workflow.add_node("run_search_issues", run_search_issues)
workflow.add_node("summarize_result", summarize_result)
# Updated create flow nodes
workflow.add_node("collect_create_fields", collect_create_fields)
workflow.add_node("ask_user_for_info", ask_user_for_info_node)
workflow.add_node("prepare_confirmation", prepare_confirmation_node)
workflow.add_node("run_create_issue", run_create_issue)
workflow.add_node("cancel_operation", cancel_node)


# Define Edges
workflow.set_entry_point("router")

def pick_route(state: GState) -> str:
    # Your router logic is here
    return state.get("route", END)

# ... (Main routing is unchanged, but ensure 'awaiting_user_details' is handled) ...
workflow.add_conditional_edges("router", pick_route, {
    "find_issues": "ensure_schema",
    "create_issue": "collect_create_fields",
    "confirm_create": "run_create_issue",
    "cancel": "cancel_operation",
    # When resuming after providing details, we re-collect/update the fields
    "awaiting_user_details": "collect_create_fields", 
})

# NEW: Conditional edge after collecting fields
def after_collecting_fields(state: GState) -> str:
    """Decides whether to ask the user for more info or proceed to confirmation."""
    if state.get("missing_fields"):
        return "ask_user_for_info"
    else:
        return "prepare_confirmation"

workflow.add_conditional_edges("collect_create_fields", after_collecting_fields, {
    "ask_user_for_info": "ask_user_for_info",
    "prepare_confirmation": "prepare_confirmation",
})

# ... (Other edges are mostly the same) ...
workflow.add_edge("ask_user_for_info", END) # Pause to get user input
workflow.add_edge("prepare_confirmation", END) 
workflow.add_edge("run_create_issue", "summarize_result")
workflow.add_edge("cancel_operation", END)
workflow.add_edge("summarize_result", END)

# You will need to re-add your find_issues flow edges
# workflow.add_edge("ensure_schema", "derive_find_params")
# workflow.add_edge("derive_find_params", "run_search_issues")
# workflow.add_edge("run_search_issues", "summarize_result")


# Compile the graph
checkpointer = MemorySaver()
app_graph = workflow.compile(checkpointer=checkpointer)


 # --- Step 1: Preliminary LLM call to extract search terms ---
    search_term_prompt = f"""You are an information extraction assistant.
Analyze the user's request and extract any text they used to describe the following JIRA fields.
Do not invent information. If they did not mention a field, leave it null.

User Request: "{state['user_input']}"
"""
    search_terms = await llm_structured(search_term_prompt, ExtractedSearchTerms, config)
    print(f"--- Extracted search terms: {search_terms.model_dump(exclude_none=True)} ---")

    # --- Step 2: Run fuzzy searches based on extracted terms ---
    field_suggestions = {}
    if search_terms.components:
        top_components = await asyncio.to_thread(fuzzy_search_components, search_terms.components, project_key, limit=5)
        if top_components: field_suggestions['components'] = top_components
    
    if search_terms.fixVersions:
        top_versions = await asyncio.to_thread(fuzzy_search_versions, search_terms.fixVersions, project_key, limit=5)
        if top_versions: field_suggestions['fixVersions'] = top_versions

    if search_terms.sprint:
        top_sprints = await asyncio.to_thread(fuzzy_search_sprint, search_terms.sprint, project_key, limit=5)
        if top_sprints: field_suggestions['sprint'] = top_sprints

    print(f"--- Fuzzy search top suggestions: {field_suggestions} ---")

    # --- Step 3: Final LLM call with refined context ---
    project_fields_schema = await asyncio.to_thread(jira_services.get_fields_list_by_project, project=project_key)
    # ... (Prepare your field_summary_for_prompt as before) ...
    field_summary_for_prompt = {} # Your logic to build this summary
    
    final_prompt = f"""You are an intelligent JIRA assistant responsible for creating new issues.
Your task is to analyze the user's request and fill out a structured JSON draft.

**User's Request:**
"{state['user_input']}"

**Available Fields Schema for Project '{project_key}':**
(Your full schema summary here)

**IMPORTANT: Here are some suggested values based on a preliminary search. If any of these are a good match for the user's request, you MUST use them.**
```json
{json.dumps(field_suggestions, indent=2)}
