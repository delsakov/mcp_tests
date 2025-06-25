# orchestration.py

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List

# Import your service functions
from . import jira_services

# A placeholder function to get user context
# In the real app, this is set dynamically in the request
def get_current_user_id() -> str:
    return "default_user"

# --- TOOL 1: Get Project Schema ---
@tool
def get_jira_project_schema(project_key: str):
    """
    Retrieves the available issue types and statuses for a specific JIRA project.
    Use this tool FIRST to understand what filter options are available for a project
    before trying to search for issues.
    """
    return jira_services.get_project_options(project_key)

# --- TOOL 2: The Main Issue Search/Filter Tool ---
class JiraIssuesInput(BaseModel):
    project_key: Optional[str] = Field(
        None,
        description="The key of the JIRA project to search within, e.g., 'PROJ1', 'SKYNET'."
    )
    issue_type: Optional[str] = Field(
        None, 
        description="Filter issues by a specific type. You MUST use a value previously obtained from the get_jira_project_schema tool."
    )
    status_exclude: Optional[List[str]] = Field(
        None, 
        description="A list of statuses to EXCLUDE from the results. Use this to find 'open' or 'not closed' issues. You MUST use values previously obtained from the get_jira_project_schema tool."
    )

@tool(args_schema=JiraIssuesInput)
def get_my_jira_issues(
    project_key: Optional[str] = None,
    issue_type: Optional[str] = None,
    status_exclude: Optional[List[str]] = None
):
    """
    Retrieves JIRA issues for the current user. It can be filtered by project,
    issue type, or by excluding certain statuses. ALWAYS use the get_jira_project_schema
    tool first to find the valid values for the filter parameters.
    """
    user_id = get_current_user_id()
    return jira_services.get_my_issues(
        user_id=user_id,
        project_key=project_key,
        issue_type=issue_type,
        status_exclude=status_exclude,
    )


# orchestration.py (add these new definitions)

class CreateJiraIssueInput(BaseModel):
    project_key: str = Field(..., description="The key of the JIRA project, e.g., 'PROJ1'.")
    summary: str = Field(..., description="The main title or summary of the JIRA issue.")
    description: str = Field(..., description="The detailed body or description for the JIRA issue.")
    issue_type: str = Field(..., description="The type of issue to create, e.g., 'Defect', 'Story', 'Task'. You should know the valid types from using the get_jira_project_schema tool.")

@tool(args_schema=CreateJiraIssueInput)
def create_jira_issue(project_key: str, summary: str, description: str, issue_type: str):
    """
    Creates a new JIRA issue. Use this when the user asks to create, add, log, or make a new ticket, defect, story, etc.
    You must have all the required information (project_key, summary, description, issue_type) before calling this tool.
    If you are missing any information, you MUST ask the user for it first.
    """
    return jira_services.create_issue(
        project_key=project_key,
        summary=summary,
        description=description,
        issue_type=issue_type
    )
