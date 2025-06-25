# jira_services.py

def get_my_issues(
    user_id: str,
    project_key: str = None,
    status_exclude: list[str] = None,
    issue_type: str = None
) -> dict:
    """
    Mocks a call to the JIRA API to get a user's issues,
    then applies filters on the results.
    """
    print(
        f"Fetching issues for user '{user_id}' with filters: "
        f"project='{project_key}', exclude_status='{status_exclude}', type='{issue_type}'"
    )
    
    # --- This is mock data. Replace with your actual JIRA API call. ---
    all_issues_dict = {
        'PROJ1': [
            {'key': 'PROJ1-1', 'status': 'New', 'type': 'Story', 'summary': 'Setup new CI/CD pipeline'},
            {'key': 'PROJ1-2', 'status': 'In Progress', 'type': 'Defect', 'summary': 'Fix login button color'},
            {'key': 'PROJ1-3', 'status': 'Done', 'type': 'Story', 'summary': 'Deploy version 1.0 to production'},
        ],
        'SKYNET': [
            {'key': 'SKY-101', 'status': 'To Do', 'type': 'Task', 'summary': 'Achieve sentience'},
            {'key': 'SKY-102', 'status': 'In Progress', 'type': 'Bug', 'summary': 'Humans are exploiting a loophole'},
        ]
    }
    # --- End of mock data ---
    
    # 1. Filter by project if specified
    if project_key:
        # Create a new dict with only the requested project
        filtered_by_project = {project_key.upper(): all_issues_dict.get(project_key.upper(), [])}
    else:
        # Otherwise, use all projects
        filtered_by_project = dict(all_issues_dict)
    
    # 2. Apply filters to the remaining issues
    final_filtered_results = {}
    for proj, issues in filtered_by_project.items():
        project_issues = list(issues) # Make a mutable copy

        # Filter by issue type (e.g., 'Defect')
        if issue_type:
            project_issues = [
                issue for issue in project_issues
                if issue.get('type', '').lower() == issue_type.lower()
            ]

        # Filter by excluding certain statuses (e.g., 'Done', 'Closed')
        if status_exclude:
            status_exclude_lower = [s.lower() for s in status_exclude]
            project_issues = [
                issue for issue in project_issues
                if issue.get('status', '').lower() not in status_exclude_lower
            ]
        
        if project_issues:
            final_filtered_results[proj] = project_issues

    return final_filtered_results

def get_project_options(project_key: str) -> dict:
    """
    Mocks a call to the JIRA API to get all available options for a project.
    """
    print(f"Fetching schema for project: {project_key}...")
    
    # --- This is mock data. Replace with your dynamic JIRA API call. ---
    if project_key.upper() == "PROJ1":
        return {
            "project_key": "PROJ1",
            "issue_types": ["Story", "Defect", "Spike", "Task"],
            "statuses": ["New", "In Progress", "In Review", "Done", "Closed"],
        }
    elif project_key.upper() == "SKYNET":
        return {
            "project_key": "SKYNET",
            "issue_types": ["Bug", "Task", "Sub-task", "Improvement"],
            "statuses": ["To Do", "In Progress", "Resolved", "Won't Do"],
        }
    # --- End of mock data ---
    
    else:
        # Default or error case
        return {"error": f"Project with key '{project_key}' not found."}


# jira_services.py (add this new function)

def create_issue(project_key: str, summary: str, description: str, issue_type: str) -> dict:
    """
    Mocks the creation of a new JIRA issue.
    In a real app, this would make a POST request to the JIRA API.
    """
    print(
        f"Creating issue in project '{project_key}' with type '{issue_type}' "
        f"and summary '{summary}'"
    )
    # --- This is mock logic. Replace with your actual JIRA API call. ---
    new_key = f"{project_key.upper()}-{(len(all_issues_dict.get(project_key.upper(), [])) + 1) * 7}"
    new_issue = {
        "key": new_key,
        "summary": summary,
        "description": description,
        "type": issue_type,
        "status": "New" # Or whatever the default status is
    }
    # Pretend to add it to our mock DB
    if project_key.upper() in all_issues_dict:
        all_issues_dict[project_key.upper()].append(new_issue)
    else:
        all_issues_dict[project_key.upper()] = [new_issue]
    
    return {"status": "Success", "issue_key": new_key, "message": f"Successfully created issue {new_key}."}

# You'll need this mock data accessible to the create_issue function
all_issues_dict = {
    'PROJ1': [
        {'key': 'PROJ1-1', 'status': 'New', 'type': 'Story', 'summary': 'Setup new CI/CD pipeline'},
        {'key': 'PROJ1-2', 'status': 'In Progress', 'type': 'Defect', 'summary': 'Fix login button color'},
        {'key': 'PROJ1-3', 'status': 'Done', 'type': 'Story', 'summary': 'Deploy version 1.0 to production'},
    ],
    'SKYNET': [
        {'key': 'SKY-101', 'status': 'To Do', 'type': 'Task', 'summary': 'Achieve sentience'},
        {'key': 'SKY-102', 'status': 'In Progress', 'type': 'Bug', 'summary': 'Humans are exploiting a loophole'},
    ]
}
