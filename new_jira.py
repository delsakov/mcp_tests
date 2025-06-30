# jira.py

import uuid
from typing import Optional
from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

# Import the compiled graph from our agent definition file
from jira_agent import app_graph

router = APIRouter(tags=["JIRA"])

# --- Simplified User/Session Management ---
async def get_user_id_from_request(request: Request) -> str:
    # In a real app, this would come from a validated auth token
    return "user_123"

# --- API Endpoint ---
@router.post("/jira/stream")
async def jira_stream(
    msg: str = Form(..., description="The JIRA-related question or request."),
    thread_id: Optional[str] = Form(None, description="Conversation thread ID."),
    model_name: str = Form("o3-mini-2025-01-31", description="Model name to use."),
    user_id: str = Depends(get_user_id_from_request)
):
    """Stream a response for JIRA-related requests using a LangGraph agent."""
    
    # LangGraph manages its own history based on a `thread_id`.
    # We can use the user_id as the session identifier for the conversation.
    session_id = user_id
    
    # The `configurable` dictionary is how we pass request-specific data
    # to the graph and the underlying LLM wrapper.
    config = {
        "configurable": {
            "thread_id": session_id,
            # Pass any other dynamic parameters your LLM needs
            "model_name": model_name,
        }
    }

    # The input to the graph must be a dictionary matching the AgentState
    inputs = {"messages": [HumanMessage(content=msg)]}
    
    # We stream the graph's output
    async def event_stream():
        async for chunk in app_graph.astream(inputs, config=config):
            # The chunk dictionary contains the output of the node that just ran.
            # We are interested in the final output from the 'agent' node when it has no tool calls.
            if "agent" in chunk:
                agent_response = chunk["agent"]["messages"][-1]
                if not agent_response.tool_calls:
                    yield agent_response.content

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")
