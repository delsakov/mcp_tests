"""
GPT-OSS Model Server - OpenAI-Compatible API
Uses tokenizer.apply_chat_template() for prompt formatting
Parses channel-based output (analysis, commentary, final)
"""

import torch
import asyncio
import uuid
import time
import re
import json
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ReasoningLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResponseFormatType(str, Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"


@dataclass
class ModelConfig:
    model_path: str
    model_name: str
    device_map: str = "auto"
    dtype: torch.dtype = torch.float16
    max_context_length: int = 8192


# ============================================================================
# GPT-OSS Response Parser
# ============================================================================

class GPTOSSParser:
    """
    Parser for GPT-OSS model output
    
    The model echoes back everything:
    1. Auto-generated system prompt (ignored)
    2. Developer instructions (user's "system" converted)
    3. Conversation history
    4. NEW assistant response (analysis, commentary, final)
    """
    
    VALID_CHANNELS = ["analysis", "commentary", "final"]
    
    @classmethod
    def parse(cls, text: str) -> Dict[str, Any]:
        """
        Parse full GPT-OSS response
        
        Returns:
            {
                "system_prompt": str or None,
                "developer_instructions": str or None,
                "history": [{"role": str, "content": str, "channel": str or None}, ...],
                "response": {"analysis": str, "commentary": str, "final": str},
                "raw": str
            }
        """
        result = {
            "system_prompt": None,
            "developer_instructions": None,
            "history": [],
            "response": {"analysis": None, "commentary": None, "final": None},
            "raw": text
        }
        
        # Extract all messages in order
        messages = cls._extract_messages(text)
        
        # Find last user message index
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                last_user_idx = i
        
        # Categorize messages
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            channel = msg.get("channel")
            
            if role == "system":
                result["system_prompt"] = content
                
            elif role == "developer":
                # Strip "# Instructions\n" prefix
                if content.startswith("# Instructions\n"):
                    content = content[16:].strip()
                elif content.startswith("# Instructions"):
                    content = content[14:].strip()
                result["developer_instructions"] = content
                
            elif role == "user":
                result["history"].append({"role": "user", "content": content})
                
            elif role == "assistant":
                if i > last_user_idx:
                    # NEW response (after last user message)
                    ch = (channel or "final").lower()
                    if ch in cls.VALID_CHANNELS:
                        result["response"][ch] = content
                else:
                    # History
                    result["history"].append({
                        "role": "assistant", 
                        "content": content,
                        "channel": channel
                    })
        
        return result
    
    @classmethod
    def _extract_messages(cls, text: str) -> List[Dict[str, Any]]:
        """Extract all message blocks from response"""
        messages = []
        
        # Pattern WITH channel: |start|>ROLE<|channel|>CH<|message|>CONTENT<|end|>
        pattern_channel = r'\|start\|>(\w+)<\|channel\|>(\w+)<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)'
        
        # Pattern WITHOUT channel: |start|>ROLE<|message|>CONTENT<|end|>
        pattern_simple = r'\|start\|>(\w+)<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)'
        
        # Find all with positions
        matches_ch = [(m.start(), {"role": m.group(1), "channel": m.group(2), "content": m.group(3).strip()})
                      for m in re.finditer(pattern_channel, text, re.DOTALL)]
        
        matches_simple = [(m.start(), {"role": m.group(1), "channel": None, "content": m.group(2).strip()})
                         for m in re.finditer(pattern_simple, text, re.DOTALL)
                         if "<|channel|>" not in text[m.start():m.start()+50]]
        
        # Sort by position and return
        all_matches = sorted(matches_ch + matches_simple, key=lambda x: x[0])
        return [m[1] for m in all_matches]


# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str
    channel: Optional[str] = None


class ResponseFormat(BaseModel):
    type: ResponseFormatType = ResponseFormatType.TEXT


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    
    # Core parameters
    max_tokens: int = Field(1000, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False
    
    # Sampling parameters
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=100)
    stop: Optional[List[str]] = Field(None, max_length=4)
    
    # Response format
    response_format: Optional[ResponseFormat] = None
    separate_thinking: bool = True
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        return v


class HistoryMessage(BaseModel):
    role: str
    content: str
    channel: Optional[str] = None


class ThinkingContent(BaseModel):
    analysis: Optional[str] = None
    commentary: Optional[str] = None


class ParsedResponse(BaseModel):
    developer_instructions: Optional[str] = None
    history: List[HistoryMessage] = Field(default_factory=list)
    analysis: Optional[str] = None
    commentary: Optional[str] = None
    final: Optional[str] = None


class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str
    thinking: Optional[ThinkingContent] = None
    parsed: Optional[ParsedResponse] = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: Optional[int] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage


class QueueStatus(BaseModel):
    queue_length: int
    active_requests: int
    max_concurrent: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    queue_status: QueueStatus


# ============================================================================
# Request Queue
# ============================================================================

class RequestQueue:
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_count = 0
        self.queue_count = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            self.queue_count += 1
        await self.semaphore.acquire()
        async with self._lock:
            self.queue_count -= 1
            self.active_count += 1
    
    async def release(self):
        async with self._lock:
            self.active_count -= 1
        self.semaphore.release()
    
    def get_status(self) -> QueueStatus:
        return QueueStatus(
            queue_length=self.queue_count,
            active_requests=self.active_count,
            max_concurrent=self.max_concurrent
        )


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self._lock = asyncio.Lock()
    
    async def load(self):
        logger.info(f"Loading model: {self.config.model_path}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, use_fast=False
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            dtype=self.config.dtype,
            device_map=self.config.device_map,
            trust_remote_code=True,
        ).eval()
        
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on device: {self.device}")
    
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        async with self._lock:
            return await self._generate(request)
    
    async def _generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        # Build prompt using tokenizer's chat template
        chat_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generate
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": max(request.temperature, 0.01),
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, **gen_kwargs)
        
        # Decode
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        completion_tokens = len(generated_ids)
        
        # Clean special tokens
        for token in ["<|end|>", "<|return|>", "</s>", "<|endoftext|>"]:
            generated_text = generated_text.replace(token, "")
        
        # Parse response
        parsed_data = GPTOSSParser.parse(generated_text)
        response = parsed_data["response"]
        
        # Build content
        final_content = response["final"] or ""
        
        # Build thinking
        thinking = None
        if request.separate_thinking and (response["analysis"] or response["commentary"]):
            thinking = ThinkingContent(
                analysis=response["analysis"],
                commentary=response["commentary"]
            )
        
        # Build parsed structure
        parsed = ParsedResponse(
            developer_instructions=parsed_data["developer_instructions"],
            history=[HistoryMessage(**h) for h in parsed_data["history"]],
            analysis=response["analysis"],
            commentary=response["commentary"],
            final=response["final"]
        )
        
        # Calculate thinking tokens
        thinking_tokens = 0
        if response["analysis"]:
            thinking_tokens += len(self.tokenizer.encode(response["analysis"]))
        if response["commentary"]:
            thinking_tokens += len(self.tokenizer.encode(response["commentary"]))
        
        # Build response
        message = ChatMessage(
            content=final_content,
            thinking=thinking,
            parsed=parsed if request.separate_thinking else None
        )
        
        choice = Choice(
            message=message,
            finish_reason="length" if completion_tokens >= request.max_tokens else "stop"
        )
        
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            thinking_tokens=thinking_tokens if thinking_tokens > 0 else None
        )
        
        return ChatCompletionResponse(
            model=self.config.model_name,
            choices=[choice],
            usage=usage
        )
    
    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        # Generate full response first (simplified streaming)
        chat_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": max(request.temperature, 0.01),
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, **gen_kwargs)
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # Parse and stream channels
        parsed = GPTOSSParser.parse(full_text)
        response = parsed["response"]
        
        for channel in ["analysis", "commentary", "final"]:
            content = response.get(channel)
            if content:
                # Stream channel marker
                yield self._chunk(response_id, created, {"channel_start": channel})
                
                # Stream content
                words = content.split()
                for i, word in enumerate(words):
                    text = word + (" " if i < len(words) - 1 else "")
                    delta = {"thinking": text} if channel != "final" else {"content": text}
                    yield self._chunk(response_id, created, delta, channel=channel)
                    await asyncio.sleep(0.02)
                
                yield self._chunk(response_id, created, {"channel_end": channel})
        
        yield self._chunk(response_id, created, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"
    
    def _chunk(self, id: str, created: int, delta: dict, channel: str = None, finish_reason: str = None) -> str:
        chunk = {
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.config.model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        }
        if channel:
            chunk["current_channel"] = channel
        return f"data: {json.dumps(chunk)}\n\n"


# ============================================================================
# FastAPI Application
# ============================================================================

model_manager: Optional[ModelManager] = None
request_queue: Optional[RequestQueue] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, request_queue
    
    import os
    
    config = ModelConfig(
        model_path=os.environ.get("MODEL_PATH", "I:/gpt-oss-20b"),
        model_name=os.environ.get("MODEL_NAME", "gpt-oss-20b"),
        device_map=os.environ.get("DEVICE_MAP", "auto"),
    )
    
    model_manager = ModelManager(config)
    request_queue = RequestQueue(max_concurrent=int(os.environ.get("MAX_CONCURRENT", "2")))
    
    await model_manager.load()
    logger.info(f"Server ready - Model: {config.model_name}")
    
    yield
    
    logger.info("Server shutdown")


app = FastAPI(
    title="GPT-OSS API Server",
    description="OpenAI-compatible API for GPT-OSS model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager is not None and model_manager.model is not None,
        model_name=model_manager.config.model_name if model_manager else "none",
        device=str(model_manager.device) if model_manager else "none",
        queue_status=request_queue.get_status() if request_queue else QueueStatus(
            queue_length=0, active_requests=0, max_concurrent=0
        )
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not model_manager or not request_queue:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    await request_queue.acquire()
    try:
        if request.stream:
            return StreamingResponse(
                model_manager.generate_stream(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            response = await model_manager.generate(request)
            return response.model_dump()
    finally:
        await request_queue.release()


@app.get("/queue/status", response_model=QueueStatus)
async def get_queue_status():
    if not request_queue:
        raise HTTPException(status_code=503, detail="Service not ready")
    return request_queue.get_status()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="GPT-OSS API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default="I:/gpt-oss-20b")
    parser.add_argument("--model-name", default="gpt-oss-20b")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-concurrent", type=int, default=2)
    
    args = parser.parse_args()
    
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["DEVICE_MAP"] = args.device_map
    os.environ["MAX_CONCURRENT"] = str(args.max_concurrent)
    
    uvicorn.run(
        "model_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1
    )
