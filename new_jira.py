# jira.py

import uuid
from typing import Optional
from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import StreamingResponse

# LangChain imports for history and message formats
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage

# Import the AgentExecutor from our agent definition file
from jira_agent import app_executor

router = APIRouter(tags=["JIRA"])

# --- Session and History Management ---
# This simple in-memory store is for demonstration.
# For production, you could use RedisChatMessageHistory or another persistent store.
message_history_store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    """Factory function to get the chat history for a given session."""
    if session_id not in message_history_store:
        message_history_store[session_id] = ChatMessageHistory()
    return message_history_store[session_id]

async def get_user_id_from_request(request: Request) -> str:
    """Placeholder for getting a user ID from a validated auth token."""
    return "user_123"

# Wrap the executor to make it stateful. This is the final, runnable object.
# It automatically manages loading and saving messages.
agent_with_chat_history = RunnableWithMessageHistory(
    app_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history", # This key must match the one in the ReAct prompt
)

# --- API Endpoint ---
@router.post("/jira/stream")
async def jira_stream(
    msg: str = Form(..., description="The JIRA-related question or request."),
    thread_id: Optional[str] = Form(None, description="Conversation thread ID for your custom LLM."),
    model_name: str = Form("o3-mini-2025-01-31", description="Model name to use."),
    user_id: str = Depends(get_user_id_from_request)
):
    """Stream a response for JIRA-related requests using a ReAct agent."""
    
    # The session_id for chat history can be the user_id or another session identifier.
    # This ensures that each user has their own separate conversation memory.
    session_id = user_id
    
    # The `configurable` dictionary is how we pass request-specific data
    # down to the agent and the underlying LLM wrapper.
    config = {
        "configurable": {
            "session_id": session_id,
            "thread_id": thread_id or str(uuid.uuid4()), # For your custom LLM
            "model_name": model_name, # For your custom LLM
        }
    }

    # The input to the agent executor is a dictionary.
    # The key "input" must match the `input_messages_key` in the wrapper.
    inputs = {"input": msg}
    
    async def event_stream():
        # Use the stateful agent's astream method to get a stream of events.
        async for chunk in agent_with_chat_history.astream(inputs, config=config):
            # The AgentExecutor stream yields dictionaries for each step (agent, tools).
            # We are interested in the final "output" from the agent to send to the user.
            if "output" in chunk:
                yield chunk["output"]

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")
