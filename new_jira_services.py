# jira_services.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List, Union

# --- Placeholder for your actual JIRA API call functions ---
# In a real app, these would live here and be called by the tools.
def _internal_get_project_options(project_key: str):
    print(f"--- SERVICE: Fetching schema for project: {project_key} ---")
    if project_key.upper() == "PROJ1":
        return {"project_key": "PROJ1", "issue_types": ["Story", "Defect"], "statuses": ["New", "In Progress", "Done"]}
    return {"project_key": project_key, "issue_types": ["Bug", "Task"], "statuses": ["To Do", "In Progress", "Resolved"]}

def _internal_get_my_issues(user_id: str, project_key: str, status_exclude: list, issue_type: str):
    print(f"--- SERVICE: Getting issues for {user_id} with filters... ---")
    return {"PROJ1-123": {"summary": "Example issue found"}}

def _internal_create_issue(project_key: str, summary: str, description: str, issue_type: str):
    print(f"--- SERVICE: Creating issue... ---")
    return {"status": "Success", "key": f"{project_key}-555"}
# -------------------------------------------------------------

class JiraProjectOutput(BaseModel):
    # Your Pydantic models for tool outputs
    pass

class JiraIssuesOutput(BaseModel):
    pass

class CreateJiraIssueInput(BaseModel):
    project_key: str = Field(..., description="The key of the JIRA project, e.g., 'PROJ1'.")
    summary: str = Field(..., description="The main title or summary of the JIRA issue.")
    description: str = Field(..., description="The detailed body or description for the JIRA issue.")
    issue_type: str = Field(..., description="The type of issue to create, e.g., 'Defect', 'Story'. You should know the valid types from using the get_jira_project_schema tool.")

@tool
def get_jira_project_schema(project_key: str) -> dict:
    """
    Retrieves the schema information for a specified JIRA project, including valid issue types and statuses.
    ALWAYS use this tool FIRST before searching for issues to ensure you're using valid filter values.
    """
    return _internal_get_project_options(project_key)

@tool
def get_my_jira_issues(
    project_key: Optional[str] = None,
    issue_type: Optional[str] = None,
    status_exclude: Optional[List[str]] = None
) -> dict:
    """
    Retrieves and filters JIRA issues for the current user. You should ALWAYS use the
    get_jira_project_schema tool first to find the valid values for the filter parameters.
    """
    # In a real app, you'd get the user_id from a secure context
    user_id = "current_user"
    return _internal_get_my_issues(user_id, project_key, status_exclude, issue_type)

@tool(args_schema=CreateJiraIssueInput)
def create_jira_issue(project_key: str, summary: str, description: str, issue_type: str) -> dict:
    """
    Creates a new JIRA issue. Use this when the user asks to create, add, log, or make a new ticket.
    You must have all the required information (project_key, summary, description, issue_type) before calling this tool.
    If you are missing any information, you MUST ask the user for it first.
    """
    return _internal_create_issue(project_key, summary, description, issue_type)
