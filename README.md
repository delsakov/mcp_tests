Integrating LLMs with Your JIRA APIsThe process can be broken down into three main components:Natural Language Understanding (NLU): A component that can read unstructured text (like an email or a chat message) and extract the necessary information to create a Jira ticket. This is where the LLM comes in.Orchestration Logic: A process that takes the user's message, sends it to the LLM for processing, and then uses the extracted information to call your existing Jira API.Tool-Enabled APIs: Your existing FastAPI endpoints, made available to the LLM via fastapi-mcp.Visualizing the WorkflowHere’s a high-level look at how the system will work:+----------------+      +-----------------+      +--------------------+      +----------------+
|  User Message  |  ->  |  Orchestrator   |  ->  |        LLM         |  ->  |  Orchestrator  |
|   (or Email)   |      |  (FastAPI App)  |      |  (Extracts Data)   |      | (Has Jira Data)|
+----------------+      +-----------------+      +--------------------+      +----------------+
                                                                                    |
                                                                                    |
                                                                                    v
+----------------+      +-----------------+
| Your Jira APIs |  <-  |  fastapi-mcp    |
| (Create, etc.) |      |  Endpoint       |
+----------------+      +-----------------+
Step-by-Step ImplementationHere’s a more detailed breakdown of the steps involved, with code examples.1. Exposing Your APIs with fastapi-mcpFirst, ensure your existing FastAPI application has the fastapi-mcp library correctly configured. If you have an endpoint to create a Jira ticket, it might look something like this:from fastapi import FastAPI
from pydantic import BaseModel
from fastapi_mcp import FastApiMCP

app = FastAPI(title="Jira API")

class JiraTicket(BaseModel):
    summary: str
    description: str
    issue_type: str = "Task"
    priority: str = "Medium"

@app.post("/create-jira")
def create_jira(ticket: JiraTicket):
    # Your existing logic to create a Jira ticket
    print(f"Creating Jira ticket: {ticket.summary}")
    return {"message": "Jira ticket created successfully", "ticket": ticket.dict()}

# Expose the endpoints via MCP
mcp = FastApiMCP(app, name="Jira Tools", description="A set of tools for managing Jira tickets.")
mcp.mount()
With this setup, your /create-jira endpoint is now a "tool" that an LLM can use.2. From Natural Language to Structured DataThis is the core of the new functionality. You'll create a new endpoint that takes unstructured text and uses an LLM to convert it into the JiraTicket model.Here's how you can do it using the OpenAI API (the principle is the same for other LLMs like Claude or Gemini):import openai
from fastapi import Body
import os

# It's recommended to use environment variables for your API key
# openai.api_key = os.environ.get("OPENAI_API_KEY")

# For demonstration, we'll hardcode it, but don't do this in production!
openai.api_key = "YOUR_OPENAI_API_KEY"

@app.post("/create-jira-from-text")
async def create_jira_from_text(text: str = Body(..., embed=True)):
    """
    Takes a string of text and uses an LLM to create a Jira ticket.
    """
    prompt = f"""
    You are an expert at analyzing user requests and converting them into structured data for Jira.
    From the following text, extract the information needed to create a Jira ticket.
    The output must be a JSON object with the following keys: "summary", "description", "issue_type", and "priority".
    If any information is missing, use sensible defaults.

    Text: "{text}"

    JSON:
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        extracted_data = response.choices[0].message.content
        
        # Now, take the extracted data and call your existing Jira creation logic
        # In a real-world app, you'd likely call your /create-jira endpoint here
        # For simplicity, we'll just print it
        
        print("Extracted Jira Data:", extracted_data)
        
        # You would then parse the JSON and call your actual Jira creation function.
        # import json
        # ticket_data = json.loads(extracted_data)
        # create_jira(JiraTicket(**ticket_data))

        return {"status": "success", "extracted_data": extracted_data}

    except Exception as e:
        return {"status": "error", "message": str(e)}

3. Handling EmailsTo parse emails, you can use Python's built-in imaplib and email libraries. You would set up a script that runs periodically to:Connect to your email server.Fetch new, unread emails.Parse the email content to get the subject and body.Send the email content to your /create-jira-from-text endpoint.Here's a simplified example of how you might fetch an email:import imaplib
import email

def fetch_latest_email():
    # Note: You'll need to enable "less secure app access" for this to work with Gmail,
    # or use app-specific passwords.
    IMAP_SERVER = "imap.gmail.com"
    EMAIL_ACCOUNT = "your_email@gmail.com"
    PASSWORD = "your_password"

    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, PASSWORD)
    mail.select("inbox")

    status, data = mail.search(None, "ALL")
    mail_ids = data[0]
    id_list = mail_ids.split()
    latest_email_id = id_list[-1]

    status, data = mail.fetch(latest_email_id, "(RFC822)")
    raw_email = data[0][1]
    msg = email.message_from_bytes(raw_email)

    subject = msg["subject"]
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get("Content-Disposition"))

            if ctype == "text/plain" and "attachment" not in cdispo:
                body = part.get_payload(decode=True).decode()
                break
    else:
        body = msg.get_payload(decode=True).decode()

    return {"subject": subject, "body": body}
You can then take the subject and body from the returned dictionary and send it to your new FastAPI endpoint.Next Steps and ConsiderationsError Handling: What if the LLM can't extract the required information? You should add logic to handle cases where the returned JSON is malformed or missing key fields.User Confirmation: For a more robust system, instead of creating the ticket directly, you could have the LLM respond to the user (e.g., by replying to the email) with the extracted information and ask for confirmation before creating the ticket.Advanced Logic: You can enhance the prompt to extract more complex information, like assigning the ticket to a specific person based on keywords in the email or setting a due date.Security: Be mindful of security when parsing emails. Sanitize the input to prevent any injection attacks.By following these steps, you can create a powerful and intuitive system for creating Jira tickets from natural language, leveraging the APIs you've already built and the power of LLMs.
