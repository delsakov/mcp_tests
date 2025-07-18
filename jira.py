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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
