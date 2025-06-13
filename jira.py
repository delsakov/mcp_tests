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
