from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from typing import Dict, Any, Optional, List

class MyLLM:
    def __init__(self, **params):
        pass

    async def get_response(self, message, thread_id):
        if "get issues" in message.lower():
            return {"tool": "get_jira_issues_chain", "params": {"project_keys": ["ABC", "XYZ"], "status": "Open"}}
        elif "details" in message.lower():
            issue_key = message.split()[-1]
            return {"tool": "get_jira_issue_details", "params": {"issue_key": issue_key}}
        return {"tool": "unknown", "params": {}}

@tool
def get_jira_project_schema(project_keys: List[str]) -> Dict[str, Any]:
    return {key: f"Schema for {key}" for key in project_keys}

@tool
def get_jira_issues(schema: Dict[str, Any], status: str) -> str:
    schema_info = ', '.join([f"{k}: {v}" for k, v in schema.items()])
    return f"Issues for projects with schemas ({schema_info}) and status {status}"

@tool
def get_jira_issue_details(issue_key: str) -> str:
    return f"Details for issue key: {issue_key}"

class GraphState:
    message: str
    thread_id: str
    llm_decision: Optional[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None
    result: Optional[str] = None

async def call_llm_and_decide(state: GraphState) -> GraphState:
    llm = MyLLM()
    decision = await llm.get_response(state.message, state.thread_id)
    state.llm_decision = decision
    return state

async def fetch_schema(state: GraphState) -> GraphState:
    params = state.llm_decision["params"]
    schema = get_jira_project_schema(params["project_keys"])
    state.schema = schema
    return state

async def fetch_issues(state: GraphState) -> GraphState:
    issues = get_jira_issues(schema=state.schema, status=state.llm_decision["params"]["status"])
    state.result = issues
    return state

async def fetch_issue_details(state: GraphState) -> GraphState:
    details = get_jira_issue_details(**state.llm_decision["params"])
    state.result = details
    return state

def decide_next_step(state: GraphState):
    tool = state.llm_decision["tool"]
    if tool == "get_jira_issues_chain":
        return "fetch_schema"
    elif tool == "get_jira_issue_details":
        return "fetch_issue_details"
    return END

workflow = StateGraph(GraphState)
workflow.add_node("call_llm_and_decide", call_llm_and_decide)
workflow.add_node("fetch_schema", fetch_schema)
workflow.add_node("fetch_issues", fetch_issues)
workflow.add_node("fetch_issue_details", fetch_issue_details)
workflow.set_entry_point("call_llm_and_decide")
workflow.add_conditional_edges("call_llm_and_decide", decide_next_step, {
    "fetch_schema": "fetch_schema",
    "fetch_issue_details": "fetch_issue_details",
    END: END
})
workflow.add_edge("fetch_schema", "fetch_issues")
workflow.add_edge("fetch_issues", END)
workflow.add_edge("fetch_issue_details", END)

app = workflow.compile()

# Example usage
async def run_example():
    state = GraphState(message="Get issues for ABC and XYZ", thread_id="12345")
    final_state = await app.ainvoke(state)
    print(final_state.result)

    state = GraphState(message="Details ISSUE-123", thread_id="12345")
    final_state = await app.ainvoke(state)
    print(final_state.result)
