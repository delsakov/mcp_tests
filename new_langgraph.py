from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from typing import Dict, Any, Optional

class MyLLM:
    def __init__(self, **params):
        pass

    async def get_response(self, message, thread_id):
        # Example simplified parsing logic using model inference
        if "get issues" in message.lower():
            return {"tool": "get_jira_issues", "params": {"project_key": "ABC", "status": "Open"}}
        elif "details" in message.lower():
            issue_key = message.split()[-1]
            return {"tool": "get_jira_issue_details", "params": {"issue_key": issue_key}}
        return {"tool": "unknown", "params": {}}

@tool
def get_jira_issues(project_key: str, status: str) -> str:
    return f"Issues from project {project_key} with status {status}"

@tool
def get_jira_issue_details(issue_key: str) -> str:
    return f"Details for issue key: {issue_key}"

class GraphState:
    message: str
    thread_id: str
    llm_decision: Optional[Dict[str, Any]] = None
    result: Optional[str] = None

async def call_llm_and_decide(state: GraphState) -> GraphState:
    llm = MyLLM()
    decision = await llm.get_response(state.message, state.thread_id)
    state.llm_decision = decision
    return state

async def execute_tool(state: GraphState) -> GraphState:
    decision = state.llm_decision
    if decision["tool"] == "get_jira_issues":
        state.result = get_jira_issues(**decision["params"])
    elif decision["tool"] == "get_jira_issue_details":
        state.result = get_jira_issue_details(**decision["params"])
    else:
        state.result = "Unable to determine appropriate action."
    return state

# Workflow definition
workflow = StateGraph(GraphState)
workflow.add_node("call_llm_and_decide", call_llm_and_decide)
workflow.add_node("execute_tool", execute_tool)
workflow.set_entry_point("call_llm_and_decide")
workflow.add_edge("call_llm_and_decide", "execute_tool")
workflow.add_edge("execute_tool", END)

app = workflow.compile()

# Example usage
async def run_example():
    state = GraphState(message="Get issues from ABC project", thread_id="12345")
    final_state = await app.ainvoke(state)
    print(final_state.result)

    state = GraphState(message="Details ISSUE-123", thread_id="12345")
    final_state = await app.ainvoke(state)
    print(final_state.result)
