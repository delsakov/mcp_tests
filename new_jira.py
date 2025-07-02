# jira.py

import os
import uuid
from typing import Optional, List
from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import StreamingResponse

# Import the manual agent loop function from our agent definition file
from jira_agent import run_agent_loop

# This is a simple in-memory store for chat history.
# In production, you would use Redis or a database for persistence.
CHAT_HISTORY_STORE: dict[str, List[str]] = {}

# Disable LangSmith tracing if you don't have external network access
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

router = APIRouter(tags=["JIRA"])

# --- Simplified User/Session Management ---
async def get_user_id_from_request(request: Request) -> str:
    """Placeholder for getting a user ID from a validated auth token."""
    return "user_123"

# --- API Endpoint ---
@router.post("/jira/stream")
async def jira_stream(
    msg: str = Form(..., description="The JIRA-related question or request."),
    thread_id: Optional[str] = Form(None, description="Conversation thread ID for your custom LLM."),
    model_name: str = Form("o3-mini-2025-01-31", description="Model name to use."),
    user_id: str = Depends(get_user_id_from_request)
):
    """
    Stream a response for JIRA-related requests by invoking the manual ReAct agent loop.
    """
    
    session_id = user_id
    
    # Retrieve and format chat history for the prompt
    if session_id not in CHAT_HISTORY_STORE:
        CHAT_HISTORY_STORE[session_id] = []
    
    history_list = CHAT_HISTORY_STORE[session_id]
    chat_history_str = "\n".join(history_list)

    # The `configurable` dictionary is how we pass request-specific data
    # down to the underlying LLM wrapper.
    config = {
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4()),
            "model_name": model_name,
        }
    }

    async def event_stream():
        # This is the crucial connection. We call the run_agent_loop function
        # and stream its results directly to the client.
        final_answer = ""
        async for chunk in run_agent_loop(msg, chat_history_str, config):
            # In this manual setup, the loop yields the final answer at the end.
            # For a better UX, you could also yield intermediate "Thought" steps.
            final_answer = chunk
        
        # Update chat history after the loop is complete
        history_list.append(f"Human: {msg}")
        history_list.append(f"AI: {final_answer}")
        # Keep history from getting too long
        if len(history_list) > 10:
            CHAT_HISTORY_STORE[session_id] = history_list[-10:]

        yield final_answer

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")



from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException

class LenientStructuredChatOutputParser(StructuredChatOutputParser):
    """
    An output parser for structured chat agents that is lenient
    about 'null' action_input. It converts 'null' to an empty dict.
    """
    def parse(self, text: str) -> AgentAction | AgentFinish:
        # The agent's default prompt format includes ```json ... ```
        try:
            # Extract the JSON block from the model's text output
            response_json_str = text.strip().split("```json")[1].split("```")[0].strip()
            response_obj = json.loads(response_json_str)

            # --- THE FIX ---
            # If action_input exists and is null, replace it with an empty dictionary
            if "action_input" in response_obj and response_obj.get("action_input") is None:
                response_obj["action_input"] = {}
            # --- END OF FIX ---

            action = response_obj.get("action")
            action_input = response_obj.get("action_input")

            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)

        except Exception as e:
            # Maintain compatibility with how the agent handles parsing errors
            raise OutputParserException(f"Could not parse LLM output: {text}") from e
