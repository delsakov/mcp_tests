# llm_wrapper.py

import asyncio
import uuid
from typing import Any, List, Optional, AsyncIterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatResult

class LLM:
    def __init__(self, model_name: str, conversation_id: str, **kwargs):
        self.model_name = model_name
        self.conversation_id = conversation_id
        print(f"--- LLM Client Initialized for model: {self.model_name}, thread: {self.conversation_id[:8]} ---")

    async def get_response(self, prompt: str, conversation_id: str) -> AsyncIterator[str]:
        response_words = [f"Response ", f"from ", f"model ", f"'{self.model_name}'... "]
        for word in response_words:
            yield word
            await asyncio.sleep(0.05)
# ------------------------------------

class InternalThreadedChatModel(BaseChatModel):
    """
    A stateless wrapper for the LLM client.
    It receives all dynamic parameters (like model_name and thread_id)
    during the actual call, not during initialization.
    """
    settings: dict # Pass your app settings here during initialization

    def _generate(self, *args, **kwargs): raise NotImplementedError("Sync methods not supported")
    def _stream(self, *args, **kwargs): raise NotImplementedError("Sync methods not supported")
    async def _agenerate(self, *args, **kwargs): raise NotImplementedError("Sync methods not supported")

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Converts a list of messages into a single string for your API."""
        return "\n".join([f"{m.type}: {m.content}" for m in messages])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """The core async streaming method."""
        # Get dynamic parameters from the config passed into the call
        configurable = kwargs.get("configurable", {})
        thread_id = configurable.get("thread_id") or str(uuid.uuid4())
        model_name = configurable.get("model_name", "o3-mini-2025-01-31") # Default model

        prompt = self._convert_messages_to_prompt(messages)

        # Initialize your client here, inside the call, making the wrapper stateless
        llm_instance = LLM(
            model_name=model_name,
            conversation_id=thread_id,
            # Pass other settings from self.settings
        )

        # The only job is to stream the raw response from the LLM
        async for chunk in llm_instance.get_response(prompt, thread_id):
            yield AIMessageChunk(content=chunk)

    # FIX: Add the required _llm_type property
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "internal-llm"


class LLMWrapper(LLM):
    """
    A custom wrapper for our LLM that expects a single string prompt
    and returns a single string response.
    """
    settings: dict 

    @property
    def _llm_type(self) -> str:
        return "llm-string-wrapper"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        """Run the LLM on the given prompt."""
        # This is a synchronous wrapper around the async call for compatibility.
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        """Async run of the LLM on the given prompt."""
        # The agent passes all dynamic config here in `kwargs`
        configurable = kwargs.get("configurable", {})
        thread_id = configurable.get("thread_id")
        model_name = configurable.get("model_name")

        llm_instance = LLM(
            model_name=model_name,
            conversation_id=thread_id,
            settings=self.settings
        )

        response = await llm_instance.get_chat_response(prompt)
        return response
