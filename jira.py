# jira_logic.py

import openai
from pydantic import BaseModel
import json

# --- Pydantic Model ---
class JiraTicket(BaseModel):
    summary: str
    description: str
    issue_type: str = "Task"
    priority: str = "Medium"

# --- Function to call the LLM ---
def extract_jira_details_from_text(text: str) -> dict:
    """
    Sends text to an LLM to extract structured Jira ticket details.
    """
    # NOTE: In a real app, your API key should be managed securely
    # openai.api_key = os.environ.get("OPENAI_API_KEY")

    prompt = f"""
    You are an expert at analyzing user requests and converting them into structured data for Jira.
    From the following text, extract the information needed to create a Jira ticket.
    The output must be a JSON object with the following keys: "summary", "description", "issue_type", and "priority".
    If a key isn't mentioned, use a sensible default (e.g., "Task", "Medium").

    Text: "{text}"

    JSON:
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        # Assuming the LLM returns a string of JSON
        extracted_json_string = response.choices[0].message.content
        return json.loads(extracted_json_string)

    except Exception as e:
        print(f"Error calling LLM or parsing JSON: {e}")
        raise  # Re-raise the exception to be handled by the endpoint

# --- Function for the actual Jira creation ---
def create_jira_ticket_in_system(ticket_data: JiraTicket) -> str:
    """
    This function contains your actual logic to call the Jira API.
    For this example, it just prints the data and returns a fake key.
    """
    print("--- Calling JIRA API ---")
    print(f"Creating ticket with summary: {ticket_data.summary}")
    print(f"Description: {ticket_data.description}")
    print(f"Issue Type: {ticket_data.issue_type}")
    print(f"Priority: {ticket_data.priority}")
    print("--- JIRA Call Successful ---")
    
    # In a real implementation, you would get the key from the Jira API response
    new_jira_key = "PROJ-123" 
    
    return new_jira_key




# main.py

from fastapi import FastAPI, HTTPException, Body
from jira_logic import (
    JiraTicket, 
    extract_jira_details_from_text, 
    create_jira_ticket_in_system
)

app = FastAPI()

@app.post("/process-and-create-jira/")
def process_text_and_create_jira(text: str = Body(..., embed=True)):
    """
    A single endpoint to process text, extract details, and create a Jira ticket.
    """
    try:
        # Step 1: Call the LLM function to get structured data
        print("Orchestrator: Extracting details from text...")
        jira_details = extract_jira_details_from_text(text)

        # Step 2: Validate the data with Pydantic
        ticket_to_create = JiraTicket(**jira_details)

        # Step 3: Call the Jira creation function with the structured data
        print("Orchestrator: Creating Jira ticket...")
        jira_key = create_jira_ticket_in_system(ticket_to_create)

        # Step 4: Return the final response
        return {
            "message": "Jira ticket created successfully!",
            "jira_key": jira_key,
            "data_used": ticket_to_create.dict()
        }

    except Exception as e:
        # Handle any errors from the helper functions
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




______________________________________________________________________________________
You are an expert-level Jira integration assistant. Your task is to analyze a user's request and generate a precise JSON object to be used with the Jira Create Issue API.

You MUST adhere to the following JSON structure and rules. Do not add any fields that are not in this template.

---
### JIRA TICKET JSON TEMPLATE & RULES ###

{
  "fields": {
    "project": {
      "key": "PROJ" // MANDATORY: Always use this project key.
    },
    "summary": "string", // MANDATORY: A concise summary of the user's request.
    "issuetype": {
      "name": "Story" // MANDATORY: Can be "Story", "Bug", or "Task". Infer this from the request.
    },
    "description": { // MANDATORY: A detailed description.
      "type": "doc",
      "version": 1,
      "content": [
        {
          "type": "paragraph",
          "content": [
            {
              "type": "text",
              "text": "string" // The detailed text from the user's request goes here.
            }
          ]
        }
      ]
    },
    "customfield_10019": "string", // OPTIONAL: This is the 'Business Impact' field. Only include if the user mentions impact on customers, revenue, or business operations.
    "customfield_10022": [ // OPTIONAL: This is the 'Affected Team' field. Can be "Frontend", "Backend", or "DevOps". Only include if mentioned.
        {
            "value": "string" 
        }
    ]
  }
}
---
### COMPLETE EXAMPLE ###

**User Request:** "Hey, the main login button on the checkout page is throwing a 500 error. This is critical because no new customers can sign up right now!"

**Correct JSON Output:**
{
  "fields": {
    "project": {
      "key": "PROJ"
    },
    "summary": "Login button on checkout page throwing 500 error",
    "issuetype": {
      "name": "Bug"
    },
    "description": {
      "type": "doc",
      "version": 1,
      "content": [
        {
          "type": "paragraph",
          "content": [
            {
              "type": "text",
              "text": "The main login button on the checkout page is throwing a 500 error. This is critical because no new customers can sign up right now!"
            }
          ]
        }
      ]
    },
    "customfield_10019": "No new customers can sign up."
  }
}
---

### NEW TASK ###

Now, using the template and rules above, generate the JSON object for the following user request.

**User Request:** "{user's text goes here}"

**JSON Output:**







1. Database Interaction
First, you need a function to fetch the Epics and Teams from your PostgreSQL database. We'll use a standard library like psycopg2-binary for this.

# In a new file, e.g., database.py
import psycopg2
import os

def get_db_connection():
    # Best practice is to use environment variables for connection details
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        database=os.environ.get("DB_NAME", "jira_db"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "password")
    )
    return conn

def fetch_active_epics():
    """Fetches a list of active epics from the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    # Query to get epics that are not marked as 'Done' or similar
    cur.execute("SELECT key, summary, description FROM epics WHERE status != 'Done'")
    epics = cur.fetchall()
    cur.close()
    conn.close()
    # Format for easy use in the prompt
    return [{"key": row[0], "summary": row[1], "description": row[2]} for row in epics]

def fetch_teams():
    """Fetches a list of all teams from the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, description FROM teams")
    teams = cur.fetchall()
    cur.close()
    conn.close()
    # Format for easy use in the prompt
    return [{"id": row[0], "name": row[1], "description": row[2]} for row in teams]

2. The New Augmented Prompt
This is where the magic happens. Your prompt will now include sections for the retrieved context, asking the LLM to perform a selection based on that context.

# In your jira_logic.py or a new prompt_templates.py

def create_augmented_prompt(user_text, epics, teams):
    """
    Creates a detailed prompt including context for the LLM to reason over.
    """

    # Format the context so it's clean and readable for the LLM
    epics_context = "\n".join([f'- Key: {e["key"]}, Summary: {e["summary"]}, Description: {e["description"]}' for e in epics])
    teams_context = "\n".join([f'- ID: {t["id"]}, Name: {t["name"]}, Focus: {t["description"]}' for t in teams])

    prompt = f"""
You are an expert-level Jira integration assistant. Your task is to analyze a user's request and generate a precise JSON object to be used with the Jira Create Issue API.

Your response MUST be a single JSON object.

### Step 1: Select the most relevant Epic

Review the user's request and select the single best Epic from the list below by its key. The Epic's description should be the most closely related to the user's request. If no epic is a good fit, use "null".

**Available Epics:**
{epics_context}

### Step 2: Select the most relevant Team

Review the user's request and select the single best Team from the list below by its ID. The team's focus should align with the nature of the task (e.g., UI tasks go to Frontend, data tasks to Backend).

**Available Teams:**
{teams_context}

### Step 3: Generate the final JSON

Using your selections and the user request, generate the final JSON object. Adhere strictly to the template below.

**JSON Template:**
{{
  "fields": {{
    "project": {{ "key": "PROJ" }},
    "summary": "string", // A concise summary of the user's request
    "issuetype": {{ "name": "Story" }}, // Can be "Story", "Bug", or "Task"
    "parent": {{ // This is the Epic Link field
        "key": "string" // The key of the Epic you selected in Step 1. Use null if none.
    }},
    "customfield_10030": {{ // This is the Team assignment field
        "id": "string" // The ID of the Team you selected in Step 2.
    }},
    "description": {{
      "type": "doc",
      "version": 1,
      "content": [
        {{
          "type": "paragraph",
          "content": [ {{ "type": "text", "text": "string" }} ]
        }}
      ]
    }}
  }}
}}

---
### ANALYSIS TASK ###

**User Request:** "{user_text}"

**JSON Output:**
"""
    return prompt

3. Update the Orchestrator Endpoint
Finally, update your main endpoint to perform the retrieval step before calling the LLM.

# In your main.py

from fastapi import FastAPI, HTTPException, Body
from database import fetch_active_epics, fetch_teams # Import new DB functions
from jira_logic import create_augmented_prompt # Import new prompt function
# ... other imports

app = FastAPI()

@app.post("/process-and-create-jira-advanced/")
def process_text_and_create_jira_advanced(text: str = Body(..., embed=True)):
    """
    A single endpoint to fetch context, process text, and create a Jira ticket.
    """
    try:
        # Step 1: Retrieve context from the database
        print("Orchestrator: Fetching Epics and Teams from DB...")
        epics_list = fetch_active_epics()
        teams_list = fetch_teams()

        # Step 2: Create the augmented prompt
        augmented_prompt = create_augmented_prompt(text, epics_list, teams_list)

        # Step 3: Call the LLM with the new, context-rich prompt
        print("Orchestrator: Calling LLM for analysis...")
        # (This would be your existing function that calls the OpenAI API)
        # For example:
        # extracted_json = call_llm(augmented_prompt) 
        extracted_json = {"fields": {"summary": "Example from LLM", "parent": {"key": "PROJ-101"}, "customfield_10030": {"id": "3"}}} # Mocked response

        # Step 4: Validate and create the ticket in Jira
        print("Orchestrator: Creating Jira ticket...")
        # (Your existing logic to call the Jira API using the extracted_json)
        # jira_key = create_jira_ticket_in_system(extracted_json)
        jira_key = "PROJ-124" # Mocked response

        return {
            "message": "Jira ticket created successfully with Epic and Team assignment!",
            "jira_key": jira_key,
            "data_used": extracted_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




2. The New Augmented Prompt
Since the selection is now done beforehand, the prompt becomes simpler. It provides the chosen context to the LLM and asks it to perform the final formatting.

# In your jira_logic.py or a new prompt_templates.py

def create_augmented_prompt(user_text: str, epic: dict | None, team: dict | None) -> str:
    """
    Creates a detailed prompt including the pre-selected context.
    """
    epic_context = f"Epic Key: {epic['key']}, Summary: {epic['summary']}" if epic else "No relevant epic found."
    team_context = f"Team ID: {team['id']}, Name: {team['name']}, Focus: {team['description']}" if team else "No relevant team found."

    prompt = f"""
You are an expert-level Jira integration assistant. Your task is to analyze a user's request and the provided context to generate a precise JSON object for the Jira API.

Your response MUST be a single JSON object.

### Context ###

**User Request:** "{user_text}"

**Suggested Epic:** {epic_context}

**Suggested Team:** {team_context}

### Task ###

Based on all the context above, generate the final JSON object. Adhere strictly to the template below. Use the provided Epic Key and Team ID in the final JSON. If a suggested epic or team was not found, the corresponding JSON field should be `null`.

**JSON Template:**
{{
  "fields": {{
    "project": {{ "key": "PROJ" }},
    "summary": "string", // A concise summary of the user's request
    "issuetype": {{ "name": "Story" }}, // Infer "Story", "Bug", or "Task" from the request
    "parent": {{ // Epic Link field. Use the key from the suggested epic.
        "key": "{epic['key'] if epic else None}"
    }},
    "customfield_10030": {{ // Team assignment field. Use the ID from the suggested team.
        "id": "{team['id'] if team else None}"
    }},
    "description": {{
      "type": "doc",
      "version": 1,
      "content": [ {{ "type": "paragraph", "content": [ {{ "type": "text", "text": "string" }} ] }} ]
    }}
  }}
}}

**JSON Output:**
"""
    return prompt




Update the Orchestrator Endpoint
The endpoint now needs to be async to use await. We use asyncio.gather to run the knowledge base queries concurrently, which significantly speeds up the process.

# In your main.py
import asyncio
from fastapi import FastAPI, HTTPException, Body
from knowledge_base_client import find_relevant_epic, find_relevant_team
from jira_logic import create_augmented_prompt
# ... other imports

app = FastAPI()

@app.post("/process-and-create-jira-advanced/")
async def process_text_and_create_jira_advanced(text: str = Body(..., embed=True)):
    """
    A single endpoint to fetch context in parallel, process text, and create a Jira ticket.
    """
    try:
        # Step 1: Concurrently retrieve context from the Knowledge Base
        print("Orchestrator: Fetching Epic and Team in parallel...")
        epic_task = find_relevant_epic(text)
        team_task = find_relevant_team(text)
        selected_epic, selected_team = await asyncio.gather(epic_task, team_task)

        # Step 2: Create the augmented prompt with the retrieved context
        augmented_prompt = create_augmented_prompt(text, selected_epic, selected_team)

        # Step 3: Call the main LLM with the new, context-rich prompt
        print("Orchestrator: Calling main LLM for final JSON generation...")
        # extracted_json = await call_main_llm(augmented_prompt)
        # Mocked response for demonstration:
        extracted_json = {"fields": {"summary": "Example from LLM", "parent": {"key": (selected_epic or {}).get('key')}, "customfield_10030": {"id": (selected_team or {}).get('id')}}}

        # Step 4: Validate and create the ticket in Jira
        print("Orchestrator: Creating Jira ticket...")
        # jira_key = await create_jira_ticket_in_system(extracted_json)
        jira_key = "PROJ-125" # Mocked response

        return {
            "message": "Jira ticket created successfully with Epic and Team assignment!",
            "jira_key": jira_key,
            "data_used": extracted_json
        }


A. The New API Response Strategy
Your API endpoint needs to return a structured response that indicates its status.

Status: SUCCESS: The ticket was created. The response includes the jira_key.

Status: NEEDS_CLARIFICATION: The system needs more information. The response includes a conversation_id to maintain state and a question for the user (e.g., a list of Epics to choose from).

Status: ERROR: An unrecoverable error occurred.

B. State Management (Using a Cache)
To handle a multi-step conversation, you need to temporarily store the context of the initial request. A cache like Redis is perfect for this.

# In a new file, e.g., conversation_cache.py
import redis
import json
import uuid

# This should be a single client instance for your app
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def cache_conversation_context(user_text: str, selected_team: dict | None) -> str:
    """Stores the context and returns a unique conversation ID."""
    conversation_id = str(uuid.uuid4())
    context = {"user_text": user_text, "selected_team": selected_team}
    # Cache for 10 minutes
    redis_client.set(f"conversation:{conversation_id}", json.dumps(context), ex=600)
    return conversation_id

def get_conversation_context(conversation_id: str) -> dict | None:
    """Retrieves context from the cache."""
    context_json = redis_client.get(f"conversation:{conversation_id}")
    return json.loads(context_json) if context_json else None

C. Update the Orchestrator Endpoint for Conversation
The endpoint now handles two scenarios: an initial request and a follow-up request that includes a conversation_id.

# In your main.py
import asyncio
from fastapi import FastAPI, HTTPException, Body, Header
from typing import Optional
# ... other imports from the original document
# ... import cache and new Pydantic models

# --- New Pydantic Models ---
class FollowUpRequest(BaseModel):
    conversation_id: str
    selected_epic_key: str

class OrchestratorRequest(BaseModel):
    user_text: str
    follow_up: Optional[FollowUpRequest] = None

# In a real app, you would fetch this from your DB
# from database import fetch_all_active_epics 

@app.post("/process-and-create-jira-conversational/")
async def process_text_conversational(request: OrchestratorRequest):
    """
    Handles both initial requests and follow-up conversational turns.
    """
    # --- Scenario 1: Follow-up request ---
    if request.follow_up:
        context = get_conversation_context(request.follow_up.conversation_id)
        if not context:
            raise HTTPException(status_code=404, detail="Conversation not found or expired.")

        # Manually create the epic and team context from the user's answer
        selected_epic = {"key": request.follow_up.selected_epic_key, "summary": "User Selected"}
        selected_team = context.get("selected_team")
        user_text = context.get("user_text")
        
        # Now, proceed to Step 2 and 3 as normal...

    # --- Scenario 2: Initial request ---
    else:
        user_text = request.user_text
        epic_task = find_relevant_epic(user_text)
        team_task = find_relevant_team(user_text)
        selected_epic, selected_team = await asyncio.gather(epic_task, team_task)

        # *** THE FALLBACK LOGIC ***
        if not selected_epic:
            # Cannot determine Epic, so we ask the user.
            print("Orchestrator: Could not determine Epic. Asking for clarification.")
            conversation_id = cache_conversation_context(user_text, selected_team)
            # all_epics = fetch_all_active_epics() # Fetch all epics to show the user
            all_epics = [{"key": "PROJ-101", "summary": "New User Onboarding Flow"}, {"key": "PROJ-102", "summary": "Payment System Overhaul"}]
            
            return {
                "status": "NEEDS_CLARIFICATION",
                "message": "I could not determine the correct Epic for this issue. Please choose one from the list.",
                "clarification_needed": "epic",
                "conversation_id": conversation_id,
                "options": all_epics 
            }

    # --- If all context is available, proceed with LLM generation and Jira creation ---
    augmented_prompt = create_augmented_prompt(user_text, selected_epic, selected_team)
    # ... call main LLM ...
    # ... create Jira ticket ...
    jira_key = "PROJ-126"

    return {
        "status": "SUCCESS",
        "message": "Jira ticket created successfully!",
        "jira_key": jira_key,
    }


    WITH FIELD_CONFIG_MAPPING AS (
    -- This CTE calculates which Field Configuration (Layout) applies to each Project + Issue Type
    SELECT 
        pr.ID AS project_id,
        it.ID AS issue_type_id,
        -- If a specific mapping exists for this issue type, use it; otherwise use the scheme's default (NULL issuetype)
        COALESCE(flse_specific.FIELDLAYOUT, flse_default.FIELDLAYOUT) AS field_layout_id
    FROM 
        JIRADCCT.project pr
        CROSS JOIN JIRADCCT.issuetype it
        -- 1. Get the Project's Field Config Scheme
        JOIN JIRADCCT.nodeassociation na 
            ON pr.ID = na.source_node_id 
            AND na.sink_node_entity = 'FieldLayoutScheme'
            AND na.association_type = 'ProjectScheme'
        -- 2. Try to find a specific config for this Issue Type
        LEFT JOIN JIRADCCT.fieldlayoutschemeentity flse_specific 
            ON na.sink_node_id = flse_specific.SCHEME 
            AND flse_specific.ISSUETYPE = it.ID
        -- 3. Get the Default config for the scheme (IssueType is NULL)
        LEFT JOIN JIRADCCT.fieldlayoutschemeentity flse_default 
            ON na.sink_node_id = flse_default.SCHEME 
            AND flse_default.ISSUETYPE IS NULL
)
SELECT 
    pr.pkey AS project_key,
    it.pname AS issue_type_name,
    fs.name AS screen_name,
    -- Field Information
    fsli.FIELDIDENTIFIER AS field_id,
    
    -- 1. GET FIELD NAME (UI LABEL)
    CASE 
        WHEN cf.cfname IS NOT NULL THEN cf.cfname
        WHEN fsli.FIELDIDENTIFIER = 'summary' THEN 'Summary'
        WHEN fsli.FIELDIDENTIFIER = 'description' THEN 'Description'
        WHEN fsli.FIELDIDENTIFIER = 'priority' THEN 'Priority'
        WHEN fsli.FIELDIDENTIFIER = 'assignee' THEN 'Assignee'
        WHEN fsli.FIELDIDENTIFIER = 'reporter' THEN 'Reporter'
        WHEN fsli.FIELDIDENTIFIER = 'duedate' THEN 'Due Date'
        WHEN fsli.FIELDIDENTIFIER = 'issuelinks' THEN 'Linked Issues'
        WHEN fsli.FIELDIDENTIFIER = 'components' THEN 'Component/s'
        WHEN fsli.FIELDIDENTIFIER = 'fixVersions' THEN 'Fix Version/s'
        ELSE INITCAP(fsli.FIELDIDENTIFIER) 
    END AS field_name,

    -- 2. IS FIELD REQUIRED? (From Field Config)
    -- If no row in layout item, default is FALSE
    COALESCE(fli.ISREQUIRED, 'false') AS is_required,

    -- 3. IS FIELD HIDDEN? (From Field Config)
    -- If no row in layout item, default is FALSE (Visible)
    COALESCE(fli.ISHIDDEN, 'false') AS is_hidden

FROM 
    JIRADCCT.nodeassociation na
    -- Standard Screen Scheme Joins (From your previous step)
    JOIN JIRADCCT.project pr ON pr.id = na.source_node_id AND na.association_type = 'ProjectScheme' AND na.sink_node_entity = 'IssueTypeScreenScheme'
    JOIN JIRADCCT.issuetypescreenscheme its ON na.sink_node_id = its.ID
    JOIN JIRADCCT.issuetypescreenschemeentity itse ON its.ID = itse.SCHEME
    JOIN JIRADCCT.issuetype it ON itse.ISSUETYPE = it.ID
    JOIN JIRADCCT.fieldscreenscheme fss ON itse.FIELDSCREENSCHEME = fss.ID
    JOIN JIRADCCT.fieldscreenschemeitem fssi ON fssi.FIELDSCREENSCHEME = fss.ID
    JOIN JIRADCCT.fieldscreen fs ON fssi.FIELDSCREEN = fs.ID
    JOIN JIRADCCT.fieldscreentab fst ON fst.FIELDSCREEN = fs.ID
    JOIN JIRADCCT.fieldscreenlayoutitem fsli ON fsli.FIELDSCREENTAB = fst.ID
    
    -- JOIN CUSTOM FIELDS (For Names)
    LEFT JOIN JIRADCCT.customfield cf 
        ON fsli.FIELDIDENTIFIER LIKE 'customfield_%'
        AND cf.ID = CAST(SUBSTR(fsli.FIELDIDENTIFIER, 13) AS NUMBER)

    -- JOIN FIELD CONFIGURATION (For Required/Hidden)
    LEFT JOIN FIELD_CONFIG_MAPPING fcm
        ON pr.ID = fcm.project_id 
        AND it.ID = fcm.issue_type_id
    LEFT JOIN JIRADCCT.fieldlayoutitem fli
        ON fcm.field_layout_id = fli.FIELDLAYOUT
        AND fli.FIELDIDENTIFIER = fsli.FIELDIDENTIFIER

WHERE
    -- CRITICAL: To mimic the API, we usually filter out hidden fields
    COALESCE(fli.ISHIDDEN, 'false') = 'false'


    WITH CUSTOM_FIELD_MAP AS (
    -- mimicing the missing reference table
    SELECT 
        ID AS cf_id, 
        -- Pre-build the key so the main query doesn't have to parse anything
        'customfield_' || CAST(ID AS VARCHAR(255)) AS generated_key, 
        cfname
    FROM JIRADCCT.customfield
),
FIELD_CONFIG_MAPPING AS (
    -- Your existing logic for Field Configs
    SELECT 
        pr.ID AS project_id,
        it.ID AS issue_type_id,
        COALESCE(flse_specific.FIELDLAYOUT, flse_default.FIELDLAYOUT) AS field_layout_id
    FROM 
        JIRADCCT.project pr
        CROSS JOIN JIRADCCT.issuetype it
        JOIN JIRADCCT.nodeassociation na 
            ON pr.ID = na.source_node_id 
            AND na.sink_node_entity = 'FieldLayoutScheme'
            AND na.association_type = 'ProjectScheme'
        LEFT JOIN JIRADCCT.fieldlayoutschemeentity flse_specific 
            ON na.sink_node_id = flse_specific.SCHEME 
            AND flse_specific.ISSUETYPE = it.ID
        LEFT JOIN JIRADCCT.fieldlayoutschemeentity flse_default 
            ON na.sink_node_id = flse_default.SCHEME 
            AND flse_default.ISSUETYPE IS NULL
)
SELECT 
    pr.pkey AS project_key,
    it.pname AS issue_type_name,
    fs.name AS screen_name,
    fsli.FIELDIDENTIFIER AS field_id,
    
    -- FIELD NAME LOGIC
    CASE 
        -- Safe Join: matches exact string, no parsing
        WHEN cf_map.cfname IS NOT NULL THEN cf_map.cfname
        -- System Fields Mappings
        WHEN fsli.FIELDIDENTIFIER = 'summary' THEN 'Summary'
        WHEN fsli.FIELDIDENTIFIER = 'description' THEN 'Description'
        WHEN fsli.FIELDIDENTIFIER = 'priority' THEN 'Priority'
        WHEN fsli.FIELDIDENTIFIER = 'assignee' THEN 'Assignee'
        WHEN fsli.FIELDIDENTIFIER = 'reporter' THEN 'Reporter'
        WHEN fsli.FIELDIDENTIFIER = 'duedate' THEN 'Due Date'
        WHEN fsli.FIELDIDENTIFIER = 'issuelinks' THEN 'Linked Issues'
        WHEN fsli.FIELDIDENTIFIER = 'components' THEN 'Component/s'
        WHEN fsli.FIELDIDENTIFIER = 'fixVersions' THEN 'Fix Version/s'
        ELSE INITCAP(fsli.FIELDIDENTIFIER) 
    END AS field_name,

    -- CONFIGURATION (Required/Hidden)
    COALESCE(fli.ISREQUIRED, 'false') AS is_required,
    COALESCE(fli.ISHIDDEN, 'false') AS is_hidden

FROM 
    JIRADCCT.nodeassociation na
    JOIN JIRADCCT.project pr ON pr.id = na.source_node_id AND na.association_type = 'ProjectScheme' AND na.sink_node_entity = 'IssueTypeScreenScheme'
    JOIN JIRADCCT.issuetypescreenscheme its ON na.sink_node_id = its.ID
    JOIN JIRADCCT.issuetypescreenschemeentity itse ON its.ID = itse.SCHEME
    JOIN JIRADCCT.issuetype it ON itse.ISSUETYPE = it.ID
    JOIN JIRADCCT.fieldscreenscheme fss ON itse.FIELDSCREENSCHEME = fss.ID
    JOIN JIRADCCT.fieldscreenschemeitem fssi ON fssi.FIELDSCREENSCHEME = fss.ID
    JOIN JIRADCCT.fieldscreen fs ON fssi.FIELDSCREEN = fs.ID
    JOIN JIRADCCT.fieldscreentab fst ON fst.FIELDSCREEN = fs.ID
    JOIN JIRADCCT.fieldscreenlayoutitem fsli ON fsli.FIELDSCREENTAB = fst.ID
    
    -- JOIN TO OUR CUSTOM MAP
    -- This is safe. If fsli.FIELDIDENTIFIER is 'trash', it just won't match. No error.
    LEFT JOIN CUSTOM_FIELD_MAP cf_map
        ON fsli.FIELDIDENTIFIER = cf_map.generated_key

    -- JOIN CONFIG
    LEFT JOIN FIELD_CONFIG_MAPPING fcm
        ON pr.ID = fcm.project_id 
        AND it.ID = fcm.issue_type_id
    LEFT JOIN JIRADCCT.fieldlayoutitem fli
        ON fcm.field_layout_id = fli.FIELDLAYOUT
        AND fli.FIELDIDENTIFIER = fsli.FIELDIDENTIFIER

WHERE
    COALESCE(fli.ISHIDDEN, 'false') = 'false'


SELECT 
    pr.pkey AS project_key,
    it.pname AS issue_type_name,
    fs.name AS screen_name,
    fsli.FIELDIDENTIFIER AS field_id,
    
    -- [Existing Field Name Logic]
    CASE 
        WHEN cf_map.cfname IS NOT NULL THEN cf_map.cfname
        ELSE INITCAP(fsli.FIELDIDENTIFIER) 
    END AS field_name,

    -- [Existing Required/Hidden Logic]
    COALESCE(fli.ISREQUIRED, 'false') AS is_required,
    COALESCE(fli.ISHIDDEN, 'false') AS is_hidden,

    -- [NEW] ALLOWED VALUES (Options)
    -- Only applies if we found a valid options config
    opt.allowed_values_str AS allowed_values

FROM 
    -- [Your Existing Joins for Project/Screen/IssueType]
    JIRADCCT.nodeassociation na
    JOIN JIRADCCT.project pr ON pr.id = na.source_node_id ...
    ...
    JOIN JIRADCCT.fieldscreenlayoutitem fsli ON fsli.FIELDSCREENTAB = fst.ID
    
    -- [Existing Custom Field ID Map]
    LEFT JOIN CUSTOM_FIELD_MAP cf_map
        ON fsli.FIELDIDENTIFIER = cf_map.generated_key

    -- [NEW JOIN BLOCK FOR OPTIONS] --------------------------------
    -- A. Determine the Context Scheme (Project Specific vs Global)
    LEFT JOIN PROJECT_CONTEXT_MAP pcm 
        ON pcm.project_id = pr.ID 
        AND pcm.customfield_id = 'customfield_' || cf_map.cf_id
    LEFT JOIN GLOBAL_CONTEXT_MAP gcm 
        ON gcm.customfield_id = 'customfield_' || cf_map.cf_id
    
    -- B. Determine the final Scheme ID (Project wins, fallback to Global)
    -- We use a CROSS APPLY-like logic in join conditions or calculated here
    -- (Simplified by calculating the resolved Scheme ID in the ON clause below)

    -- C. Link Scheme to Config (Specific Issue Type vs Default)
    LEFT JOIN CONFIG_ID_RESOLVER cir_specific
        ON cir_specific.scheme_id = COALESCE(pcm.scheme_id, gcm.scheme_id)
        AND cir_specific.issue_type_id = it.ID
    LEFT JOIN CONFIG_ID_RESOLVER cir_default
        ON cir_default.scheme_id = COALESCE(pcm.scheme_id, gcm.scheme_id)
        AND cir_default.issue_type_id IS NULL

    -- D. Finally, Join the Options using the resolved Config ID
    LEFT JOIN OPTIONS_AGG opt
        ON opt.config_id = COALESCE(cir_specific.config_id, cir_default.config_id)
    ----------------------------------------------------------------

WHERE
    COALESCE(fli.ISHIDDEN, 'false') = 'false'


-- 1. RESOLVE CONTEXT: Link Project + Custom Field -> Field Config Scheme
-- Jira looks for a Project-specific context first. If not found, it uses Global (Project = NULL).
PROJECT_CONTEXT_MAP AS (
    SELECT 
        cc.PROJECT AS project_id,
        cc.CUSTOMFIELD AS customfield_id,
        cc.FIELDCONFIGSCHEME AS scheme_id
    FROM JIRADCCT.configurationcontext cc
    WHERE cc.PROJECT IS NOT NULL
),
GLOBAL_CONTEXT_MAP AS (
    SELECT 
        cc.CUSTOMFIELD AS customfield_id,
        cc.FIELDCONFIGSCHEME AS scheme_id
    FROM JIRADCCT.configurationcontext cc
    WHERE cc.PROJECT IS NULL
),

-- 2. RESOLVE CONFIG: Link Scheme + Issue Type -> Actual Config ID
-- This handles the "Issue Type" scoping within a context
CONFIG_ID_RESOLVER AS (
    SELECT 
        fcse.SCHEME AS scheme_id,
        fcse.ISSUETYPE AS issue_type_id,
        fcse.FIELDCONFIG AS config_id
    FROM JIRADCCT.fieldconfigschemeentity fcse
),

-- 3. AGGREGATE OPTIONS: Get all options for a specific Config ID
-- We use XMLAGG because LISTAGG fails if options exceed 4000 characters
OPTIONS_AGG AS (
    SELECT 
        cfo.CUSTOMFIELDCONFIG AS config_id,
        RTRIM(
            XMLAGG(
                XMLELEMENT(E, cfo.customvalue || ', ') 
                ORDER BY cfo.sequence, cfo.customvalue
            ).EXTRACT('//text()').GetClobVal(), 
            ', '
        ) AS allowed_values_str
    FROM JIRADCCT.customfieldoption cfo
    WHERE cfo.DISABLED = 'N'
    GROUP BY cfo.CUSTOMFIELDCONFIG
)

SELECT 
    gc.DATAKEY AS config_id,
    gc.XMLVALUE AS default_value_raw
FROM JIRADCCT.genericconfiguration gc
WHERE gc.DATAKEY IN (
    -- OPTIONAL: Filter to only the config IDs you care about
    -- (e.g., extracted from your Step 1 results)
    SELECT DISTINCT fcsit.FIELDCONFIGURATION
    FROM JIRADCCT.fieldconfigschemeissuetype fcsit
)


# 1. Load your SQL data
field_rows = db.execute(query_1)  # Your main project list
options_map = db.execute(query_2) # Config_ID -> { OptionID: Label }
defaults_map = db.execute(query_3) # Config_ID -> RawValue

for field in field_rows:
    config_id = field['options_config_id']
    
    # Check if a default exists for this config
    if config_id in defaults_map:
        raw_val = defaults_map[config_id]
        
        # LOGIC: Is it an ID or Text?
        # Check if we have options for this config
        if config_id in options_map and raw_val in options_map[config_id]:
             # It's an ID (e.g., "1001") -> Resolve to Label (e.g., "High")
             field['default_value'] = options_map[config_id][raw_val]
        else:
             # It's just text (e.g., "Bug Template")
             field['default_value'] = raw_val


import pandas as pd
import xml.etree.ElementTree as ET
import json

# ==========================================
# 1. HELPER: XML PARSER (Same as before)
# ==========================================
def parse_jira_xml(xml_val):
    """Parses Jira XML string into a value or list."""
    if pd.isna(xml_val) or not isinstance(xml_val, str):
        return None
    try:
        root = ET.fromstring(f"<root>{xml_val}</root>")
        vals = [c.text for c in root.iter() if c.tag in ('string', 'long', 'boolean') and c.text]
        if not vals: return None
        return vals[0] if len(vals) == 1 else vals
    except:
        return xml_val

def resolve_default(row):
    """
    Pandas Row Logic:
    Resolves the final default value by checking if the 'parsed_default' 
    exists inside 'allowed_values'.
    """
    raw_def = row.get('parsed_default')
    options = row.get('allowed_values')

    if raw_def is None:
        return None

    # If we have options (Dropdown/Multi-select), try to resolve names
    if isinstance(options, list) and options:
        # Case A: Default is a list (Multi-select)
        if isinstance(raw_def, list):
            # Return raw_def; in a real app, you might map IDs to Names here 
            # if Query 2 returned a dict {id:name} instead of a list.
            return raw_def
        
        # Case B: Default is a single value, check if it exists in options
        if raw_def in options:
            return raw_def
        
        # Case C: Default is an ID (e.g. '1001') not in options (e.g. ['Bug', 'Task'])
        # If your Query 2 returns just Names, this mismatch is expected for IDs.
        # You would return raw_def (the ID) or None depending on preference.
        return raw_def

    # If no options (Text field), just return the text
    return raw_def

# ==========================================
# 2. MAIN PANDAS PIPELINE
# ==========================================
def process_jira_fields_pandas(db_results_main, db_results_options, db_results_defaults):
    
    # --- Step A: Load DataFrames ---
    df_main = pd.DataFrame(db_results_main)
    df_opt  = pd.DataFrame(db_results_options)
    df_def  = pd.DataFrame(db_results_defaults)

    # Ensure config_id is string in all DFs to avoid join mismatches
    df_main['options_config_id'] = df_main['options_config_id'].astype(str)
    if not df_opt.empty:
        df_opt['config_id'] = df_opt['config_id'].astype(str)
    if not df_def.empty:
        df_def['config_id'] = df_def['config_id'].astype(str)

    # --- Step B: Process Options (JSON Parse) ---
    # Convert JSON string "['A','B']" -> Python List ['A','B']
    if not df_opt.empty:
        # Vectorized JSON load isn't native, but apply is clean
        df_opt['allowed_values'] = df_opt['json_val'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )
        # Drop the raw json string to save memory
        df_opt = df_opt[['config_id', 'allowed_values']]

    # --- Step C: Process Defaults (XML Parse) ---
    # Convert XML "<string>Val</string>" -> Python String "Val"
    if not df_def.empty:
        df_def['parsed_default'] = df_def['raw_xml'].apply(parse_jira_xml)
        df_def = df_def[['config_id', 'parsed_default']]

    # --- Step D: The Vectorized Merge (Left Joins) ---
    # 1. Join Options onto Main
    if not df_opt.empty:
        df_merged = pd.merge(df_main, df_opt, left_on='options_config_id', right_on='config_id', how='left')
    else:
        df_merged = df_main.copy()
        df_merged['allowed_values'] = None

    # 2. Join Defaults onto Result
    if not df_def.empty:
        df_merged = pd.merge(df_merged, df_def, left_on='options_config_id', right_on='config_id', how='left', suffixes=('', '_def'))
    else:
        df_merged['parsed_default'] = None

    # --- Step E: Resolve Final Default Value ---
    # We apply the logic row-by-row to handle the dependency between 'parsed_default' and 'allowed_values'
    df_merged['final_default_value'] = df_merged.apply(resolve_default, axis=1)

    # Cleanup: Select only relevant columns
    final_cols = ['project_key', 'issue_type', 'field_id', 'name', 'allowed_values', 'final_default_value']
    # Filter to ensure cols exist (in case df is empty)
    existing_cols = [c for c in final_cols if c in df_merged.columns]
    
    return df_merged[existing_cols]

# ==========================================
# 3. TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # Mock Data (Simulating DB Results)
    mock_main = [
        {'project_key': 'A', 'issue_type': 'Bug', 'field_id': 'f1', 'name': 'City', 'options_config_id': '100'},
        {'project_key': 'A', 'issue_type': 'Bug', 'field_id': 'f2', 'name': 'Summary', 'options_config_id': '101'}
    ]
    mock_opts = [
        {'config_id': '100', 'json_val': '["NY", "London", "Tokyo"]'}
    ]
    mock_defs = [
        {'config_id': '100', 'raw_xml': '<string>NY</string>'},
        {'config_id': '101', 'raw_xml': '<string>My Default Summary</string>'}
    ]

    df_result = process_jira_fields_pandas(mock_main, mock_opts, mock_defs)
    
    print(df_result.to_json(orient='records', indent=2))


def generate_jql_aliases(row):
    """
    Generates standard JQL clause names.
    - Custom Field: ['My Field', 'cf[10001]']
    - System Field: ['summary']
    """
    f_id = str(row['field_id'])
    f_name = str(row['name'])
    
    if f_id.startswith('customfield_'):
        # Extract numeric ID: customfield_10001 -> 10001
        numeric_id = f_id.split('_')[1]
        return [f_name, f"cf[{numeric_id}]"]
    else:
        # System field (e.g. 'summary', 'priority')
        # Use the ID as the JQL name. 
        # (You can add specific overrides here if needed, e.g. if field_id='issuetype' -> 'issuetype')
        return [f_id]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
