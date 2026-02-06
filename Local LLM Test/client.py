"""
Python Client for Local LLM API
Example usage and utility classes
"""

import httpx
import asyncio
import json
from typing import Optional, List, Dict, Any, Generator, AsyncGenerator
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """A chat message with optional thinking"""
    role: str
    content: str
    thinking: Optional[str] = None
    thinking_steps: Optional[List[str]] = None


@dataclass
class ChatResponse:
    """Parsed chat response"""
    message: ChatMessage
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: Optional[int] = None
    backend: Optional[str] = None


class LocalLLMClient:
    """
    Synchronous client for the Local LLM API
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8080",
        timeout: float = 120.0
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def close(self):
        self.client.close()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-oss-20b",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        separate_thinking: bool = True,
        response_format: Optional[Dict] = None,
        stop: Optional[List[str]] = None,
        user: Optional[str] = None
    ) -> ChatResponse:
        """
        Send a chat completion request
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            separate_thinking: Extract thinking from response
            response_format: {"type": "text"|"json_object"}
            stop: List of stop sequences
            user: User ID for sticky sessions
        
        Returns:
            ChatResponse object with parsed response
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "separate_thinking": separate_thinking,
            "stream": False
        }
        
        if response_format:
            payload["response_format"] = response_format
        if stop:
            payload["stop"] = stop
        if user:
            payload["user"] = user
        
        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return self._parse_response(data)
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-oss-20b",
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion
        
        Yields content tokens as they arrive
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    
                    if "content" in delta:
                        yield delta["content"]
    
    def list_models(self) -> List[Dict]:
        """Get list of available models"""
        response = self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json().get("data", [])
    
    def get_status(self) -> Dict:
        """Get cluster status"""
        response = self.client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if service is healthy"""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def _parse_response(self, data: Dict) -> ChatResponse:
        """Parse API response into ChatResponse"""
        choice = data.get("choices", [{}])[0]
        message_data = choice.get("message", {})
        usage = data.get("usage", {})
        routing = data.get("_routing", {})
        
        thinking_data = message_data.get("thinking")
        thinking = None
        thinking_steps = None
        if thinking_data:
            thinking = thinking_data.get("raw")
            thinking_steps = thinking_data.get("steps")
        
        message = ChatMessage(
            role=message_data.get("role", "assistant"),
            content=message_data.get("content", ""),
            thinking=thinking,
            thinking_steps=thinking_steps
        )
        
        return ChatResponse(
            message=message,
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            thinking_tokens=usage.get("thinking_tokens"),
            backend=routing.get("backend")
        )


class AsyncLocalLLMClient:
    """
    Asynchronous client for the Local LLM API
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8080",
        timeout: float = 120.0
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    async def close(self):
        await self.client.aclose()
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-oss-20b",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        separate_thinking: bool = True,
        **kwargs
    ) -> ChatResponse:
        """Send an async chat completion request"""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "separate_thinking": separate_thinking,
            "stream": False,
            **kwargs
        }
        
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return self._parse_response(data)
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-oss-20b",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion asynchronously"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    
                    if "content" in delta:
                        yield delta["content"]
    
    async def list_models(self) -> List[Dict]:
        """Get list of available models"""
        response = await self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json().get("data", [])
    
    async def get_status(self) -> Dict:
        """Get cluster status"""
        response = await self.client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> bool:
        """Check if service is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def _parse_response(self, data: Dict) -> ChatResponse:
        """Parse API response into ChatResponse"""
        choice = data.get("choices", [{}])[0]
        message_data = choice.get("message", {})
        usage = data.get("usage", {})
        routing = data.get("_routing", {})
        
        thinking_data = message_data.get("thinking")
        thinking = None
        thinking_steps = None
        if thinking_data:
            thinking = thinking_data.get("raw")
            thinking_steps = thinking_data.get("steps")
        
        message = ChatMessage(
            role=message_data.get("role", "assistant"),
            content=message_data.get("content", ""),
            thinking=thinking,
            thinking_steps=thinking_steps
        )
        
        return ChatResponse(
            message=message,
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            thinking_tokens=usage.get("thinking_tokens"),
            backend=routing.get("backend")
        )


# ============================================================================
# Example Usage
# ============================================================================

def example_sync():
    """Example: Synchronous usage"""
    print("=" * 60)
    print("Synchronous Client Example")
    print("=" * 60)
    
    with LocalLLMClient("http://localhost:8080") as client:
        # Check health
        if not client.health_check():
            print("Service is not healthy!")
            return
        
        # List models
        models = client.list_models()
        print(f"\nAvailable models: {[m['id'] for m in models]}")
        
        # Simple chat
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Think through problems carefully."},
                {"role": "user", "content": "What is the square root of 144? Think step by step."}
            ],
            model="gpt-oss-20b",
            max_tokens=500
        )
        
        print(f"\n--- Response ---")
        print(f"Model: {response.model}")
        print(f"Backend: {response.backend}")
        print(f"Answer: {response.message.content}")
        
        if response.message.thinking:
            print(f"\n--- Thinking ---")
            print(response.message.thinking)
            if response.message.thinking_steps:
                print(f"\nSteps: {response.message.thinking_steps}")
        
        print(f"\n--- Usage ---")
        print(f"Prompt tokens: {response.prompt_tokens}")
        print(f"Completion tokens: {response.completion_tokens}")
        print(f"Total tokens: {response.total_tokens}")
        if response.thinking_tokens:
            print(f"Thinking tokens: {response.thinking_tokens}")


def example_streaming():
    """Example: Streaming response"""
    print("\n" + "=" * 60)
    print("Streaming Example")
    print("=" * 60)
    
    with LocalLLMClient("http://localhost:8080") as client:
        print("\nStreaming response: ", end="", flush=True)
        
        for token in client.chat_stream(
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            model="gpt-oss-20b",
            max_tokens=100
        ):
            print(token, end="", flush=True)
        
        print("\n")


async def example_async():
    """Example: Asynchronous usage"""
    print("=" * 60)
    print("Asynchronous Client Example")
    print("=" * 60)
    
    async with AsyncLocalLLMClient("http://localhost:8080") as client:
        # Multiple concurrent requests
        tasks = [
            client.chat(
                messages=[{"role": "user", "content": f"What is {i} + {i}?"}],
                model="gpt-oss-20b",
                max_tokens=50
            )
            for i in range(1, 4)
        ]
        
        results = await asyncio.gather(*tasks)
        
        print("\nConcurrent results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}: {result.message.content.strip()}")


async def example_async_stream():
    """Example: Async streaming"""
    print("\n" + "=" * 60)
    print("Async Streaming Example")
    print("=" * 60)
    
    async with AsyncLocalLLMClient("http://localhost:8080") as client:
        print("\nAsync streaming: ", end="", flush=True)
        
        async for token in client.chat_stream(
            messages=[{"role": "user", "content": "Say hello in 3 languages."}],
            model="gpt-oss-20b"
        ):
            print(token, end="", flush=True)
        
        print("\n")


def example_json_mode():
    """Example: JSON response format"""
    print("=" * 60)
    print("JSON Mode Example")
    print("=" * 60)
    
    with LocalLLMClient("http://localhost:8080") as client:
        response = client.chat(
            messages=[
                {"role": "system", "content": "Always respond with valid JSON."},
                {"role": "user", "content": "List 3 programming languages with their years of creation as JSON."}
            ],
            model="gpt-oss-20b",
            response_format={"type": "json_object"}
        )
        
        print(f"\nJSON Response:\n{response.message.content}")
        
        # Try to parse it
        try:
            data = json.loads(response.message.content)
            print(f"\nParsed successfully: {json.dumps(data, indent=2)}")
        except json.JSONDecodeError:
            print("\nNote: Response may need extraction from surrounding text")


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("Local LLM Client Examples")
    print("=" * 60)
    print("\nNote: Make sure the LLM service is running before running examples.")
    print("Start with: python model_server.py or use the load balancer.\n")
    
    try:
        # Sync examples
        example_sync()
        example_streaming()
        example_json_mode()
        
        # Async examples
        asyncio.run(example_async())
        asyncio.run(example_async_stream())
        
    except httpx.ConnectError:
        print("\nError: Could not connect to the LLM service.")
        print("Make sure the service is running on http://localhost:8080")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
