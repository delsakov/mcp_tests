# main.py

import asyncio
from typing import Any, Iterator, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# LangChain imports
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Local module imports
from . import orchestration
from . import jira_services


# --- 1. YOUR CUSTOM LLM & SESSION MANAGEMENT ---

# This is a placeholder for your actual streaming LLM class.
# It simulates creating a thread and streaming back a response.
class MyLLM:
    def create_thread(self) -> str:
        import uuid
        new_thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        return new_thread_id

    def get_response(self, thread_id: str, message: str):
        # In a real app, this calls your model's API. Here we simulate it.
        print(f"\n--- LLM SIMULATOR (Thread: {thread_id}) ---")
        print(f"Received prompt: {message}")
        response = f"Simulated response for thread '{thread_id}' about '{message}'"
        for word in response.split():
            yield f"{word} "
            asyncio.sleep(0.05)
        print("--- END LLM SIMULATOR ---\n")

# This is the custom LangChain wrapper for your LLM.
class InternalThreadedChatModel(BaseChatModel):
    llm_instance: MyLLM

    def _get_thread_id(self, configurable: dict) -> str:
        thread_id = configurable.get("thread_id")
        if not thread_id:
            thread_id = self.llm_instance.create_thread()
        return thread_id

    def _stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        configurable = kwargs.get("configurable", {})
        thread_id = self._get_thread_id(configurable)
        user_prompt = messages[-1].content
        stream_iterator = self.llm_instance.get_response(thread_id, str(user_prompt))

        for chunk_str in stream_iterator:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=chunk_str, response_metadata={"thread_id": thread_id}
            ))

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> ChatResult:
        stream_iterator = self._stream(messages, stop, run_manager, **kwargs)
        full_response_content = "".join(chunk.message.content for chunk in stream_iterator)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=full_response_content))])

    @property
    def _llm_type(self) -> str:
        return "internal-threaded-streaming-chat-model"

# WARNING: In-memory store is for demonstration only. Use Redis or a DB in production.
USER_THREADS = {}
def get_user_thread_id(user_id: str) -> Optional[str]:
    return USER_THREADS.get(user_id)
def set_user_thread_id(user_id: str, thread_id: str):
    USER_THREADS[user_id] = thread_id


# --- 2. AGENT AND FASTAPI SETUP ---

app = FastAPI(title="Schema-Aware JIRA Agent")

# Instantiate your custom model and tools
my_llm_instance = MyLLM()
llm = InternalThreadedChatModel(llm_instance=my_llm_instance)
tools = [
    orchestration.get_my_jira_issues,
    orchestration.get_jira_project_schema,
    orchestration.create_jira_issue, # Add the new tool here
]

# The advanced system prompt that instructs the agent on the two-step process
SYSTEM_PROMPT = """You are a comprehensive JIRA assistant, capable of both reading and writing JIRA data.

**Finding Issues:**
When a user asks to find issues with specific criteria (like type, status), you MUST follow this sequence:
1.  First, identify the JIRA project key. If you are unsure, ask the user.
2.  Use the `get_jira_project_schema` tool to fetch the valid filter options for that project.
3.  Examine the schema. Map the user's request (e.g., "open defects") to the official terms from the schema.
4.  Finally, call the `get_my_jira_issues` tool with the correct, schema-validated parameters.

**Creating Issues:**
When a user asks to create a ticket, defect, story, etc., you MUST follow this sequence:
1.  Use the `create_jira_issue` tool. This tool requires a project key, an issue type, a summary, and a description.
2.  Gather ALL of the required information from the user's request.
3.  If any piece of information is missing (e.g., they provide a summary but no description), you MUST ask clarifying questions to get the missing details from the user.
4.  Do not call the `create_jira_issue` tool until you have all four required pieces of information. If you are unsure of the valid issue types for a project, you can use the `get_jira_project_schema` tool first.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


class ChatRequest(BaseModel):
    prompt: str
    user_id: str
    # In a real app, you'd get user_id from a decoded auth token
    
# --- 3. STREAMING API ENDPOINT ---

async def stream_agent_response(user_id: str, prompt_text: str):
    """Async generator to handle the agent streaming and session management."""
    # Set the user ID for the context of this request
    orchestration.get_current_user_id = lambda: user_id
    
    thread_id = get_user_thread_id(user_id)
    config = {"configurable": {"thread_id": thread_id}}

    is_first_chunk = True
    output_streamed = ""
    
    # Use the asynchronous stream method
    async for chunk in agent_executor.astream(
        {"input": prompt_text},
        config=config
    ):
        # The agent executor yields dictionaries for each step.
        # We are interested in the final answer chunks from the LLM.
        if "output" in chunk:
            chunk_content = chunk["output"]
            if chunk_content:
                # The output comes in full every time, so we send the new part
                new_content = chunk_content.replace(output_streamed, "", 1)
                yield new_content
                output_streamed = chunk_content

        # On the first LLM chunk, check if a new thread was created and save it
        if is_first_chunk and "messages" in chunk:
            last_message = chunk["messages"][-1]
            if isinstance(last_message, AIMessage) and not thread_id:
                new_thread_id = last_message.response_metadata.get("thread_id")
                if new_thread_id:
                    set_user_thread_id(user_id, new_thread_id)
                    print(f"✅ Saved new thread '{new_thread_id}' for user '{user_id}'")
                is_first_chunk = False


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """FastAPI endpoint that returns a streaming response from the JIRA agent."""
    return StreamingResponse(
        stream_agent_response(request.user_id, request.prompt),
        media_type="text/plain"
    )

@app.get("/")
def read_root():
    return {"message": "JIRA Agent API is running. POST to /chat/stream to interact."}
