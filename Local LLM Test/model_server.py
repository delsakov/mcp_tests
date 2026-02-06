"""
Model Server Wrapper - OpenAI-Compatible API for Local LLMs
Supports: GPT-OSS (with channels), Gemma3, and other SafeTensor models
Features: Channel-based thinking separation, Queue management, Streaming
"""

import torch
import asyncio
import uuid
import time
import re
import json
import logging
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import Gemma3 specific class
try:
    from transformers import Gemma3ForConditionalGeneration
    HAS_GEMMA3 = True
except ImportError:
    HAS_GEMMA3 = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Enums
# ============================================================================

class ModelType(str, Enum):
    GPT_OSS = "gpt-oss"
    GEMMA3 = "gemma3"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GENERIC = "generic"


class ReasoningLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResponseFormatType(str, Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_path: str
    model_type: ModelType
    model_name: str
    device_map: str = "auto"
    dtype: torch.dtype = torch.float16
    max_context_length: int = 8192
    use_flash_attention: bool = False


# ============================================================================
# GPT-OSS Token Definitions
# ============================================================================

class GPTOSSTokens:
    """
    Token definitions for GPT-OSS model format
    
    The model response includes EVERYTHING:
    1. Auto-generated system prompt (we ignore this)
    2. Developer instructions (converted from user's "system" role)
    3. Full conversation history
    4. New assistant response (analysis, commentary, final)
    
    We parse and separate all of these.
    """
    
    # Message structure tokens
    START = "|start|>"
    END = "<|end|>"
    MESSAGE = "<|message|>"
    CHANNEL = "<|channel|>"
    RETURN = "<|return|>"
    
    # Roles
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    
    # Channels
    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    FINAL = "final"
    
    # Valid channels list
    VALID_CHANNELS = [ANALYSIS, COMMENTARY, FINAL]
    
    @classmethod
    def format_message(cls, role: str, content: str, channel: Optional[str] = None) -> str:
        """Format a message in GPT-OSS format"""
        if channel:
            return f"{cls.START}{role}{cls.CHANNEL}{channel}{cls.MESSAGE}{content}{cls.END}"
        return f"{cls.START}{role}{cls.MESSAGE}{content}{cls.END}"
    
    @classmethod
    def format_system_prompt(
        cls, 
        instructions: str,
        reasoning: ReasoningLevel = ReasoningLevel.MEDIUM,
        knowledge_cutoff: str = "2024-06",
        current_date: Optional[str] = None
    ) -> str:
        """
        Format a system prompt (OPTIONAL - model auto-generates this)
        Only use if you need to override the model's default system prompt.
        """
        if current_date is None:
            current_date = time.strftime("%Y-%m-%d")
        
        system_content = (
            f"You are ChatGPT, a large language model trained by OpenAI.\n"
            f"Knowledge cutoff: {knowledge_cutoff}\n"
            f"Current date: {current_date}\n\n"
            f"Reasoning: {reasoning.value}\n\n"
            f"# Valid channels: {', '.join(cls.VALID_CHANNELS)}. "
            f"Channel must be included for every message."
        )
        
        return cls.format_message(cls.SYSTEM, system_content)
    
    @classmethod
    def format_developer_instructions(cls, instructions: str) -> str:
        """Format developer instructions"""
        return f"{cls.START}{cls.DEVELOPER}{cls.MESSAGE}# Instructions\n{instructions}{cls.END}"
    
    @classmethod
    def parse_full_response(cls, text: str) -> Dict[str, Any]:
        """
        Parse the FULL GPT-OSS response into structured components
        
        Returns:
            {
                "system_prompt": str or None,      # Auto-generated (usually ignored)
                "developer_instructions": str or None,  # User's system->developer
                "history": [                       # Conversation history
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "...", "channel": "final"},
                    ...
                ],
                "response": {                      # NEW assistant response
                    "analysis": str or None,
                    "commentary": str or None,
                    "final": str or None,
                },
                "raw": str                         # Original text
            }
        """
        result = {
            "system_prompt": None,
            "developer_instructions": None,
            "history": [],
            "response": {
                "analysis": None,
                "commentary": None,
                "final": None,
            },
            "raw": text
        }
        
        # Extract all message blocks
        all_messages = cls._extract_all_messages(text)
        
        # Process each message
        history_messages = []
        last_user_index = -1
        
        # Find the last user message index
        for i, msg in enumerate(all_messages):
            if msg["role"] == "user":
                last_user_index = i
        
        for i, msg in enumerate(all_messages):
            role = msg["role"]
            content = msg["content"]
            channel = msg.get("channel")
            
            if role == "system":
                # Auto-generated system prompt - store but typically ignored
                result["system_prompt"] = content
                
            elif role == "developer":
                # Developer instructions (converted from user's "system" role)
                # Remove "# Instructions\n" prefix if present
                if content.startswith("# Instructions\n"):
                    content = content[len("# Instructions\n"):].strip()
                elif content.startswith("# Instructions"):
                    content = content[len("# Instructions"):].strip()
                result["developer_instructions"] = content
                
            elif role == "user":
                # User messages go to history
                history_messages.append({
                    "role": "user",
                    "content": content
                })
                
            elif role == "assistant":
                if i > last_user_index:
                    # This is part of the NEW response (after last user message)
                    if channel and channel.lower() in cls.VALID_CHANNELS:
                        result["response"][channel.lower()] = content
                    else:
                        # No channel specified, treat as final
                        result["response"]["final"] = content
                else:
                    # This is history (before or at last user message)
                    history_messages.append({
                        "role": "assistant",
                        "content": content,
                        "channel": channel
                    })
        
        result["history"] = history_messages
        
        return result
    
    @classmethod
    def _extract_all_messages(cls, text: str) -> List[Dict[str, Any]]:
        """
        Extract all message blocks from the response in order
        
        Returns list of:
            {"role": str, "content": str, "channel": str or None}
        """
        messages = []
        
        # Pattern for messages WITH channel (assistant)
        # |start|>assistant<|channel|>CHANNEL<|message|>CONTENT<|end|>
        pattern_with_channel = (
            r'\|start\|>(\w+)<\|channel\|>(\w+)<\|message\|>'
            r'(.*?)'
            r'(?:<\|end\|>|<\|return\|>)'
        )
        
        # Pattern for messages WITHOUT channel (system, developer, user)
        # |start|>ROLE<|message|>CONTENT<|end|>
        pattern_without_channel = (
            r'\|start\|>(\w+)<\|message\|>'
            r'(.*?)'
            r'(?:<\|end\|>|<\|return\|>)'
        )
        
        # Find all matches with their positions
        matches_with_channel = [
            (m.start(), {
                "role": m.group(1),
                "channel": m.group(2),
                "content": m.group(3).strip()
            })
            for m in re.finditer(pattern_with_channel, text, re.DOTALL)
        ]
        
        matches_without_channel = [
            (m.start(), {
                "role": m.group(1),
                "channel": None,
                "content": m.group(2).strip()
            })
            for m in re.finditer(pattern_without_channel, text, re.DOTALL)
            # Exclude matches that are actually part of channel pattern
            if "<|channel|>" not in text[m.start():m.start()+50]
        ]
        
        # Combine and sort by position
        all_matches = matches_with_channel + matches_without_channel
        all_matches.sort(key=lambda x: x[0])
        
        # Extract just the message dicts
        messages = [m[1] for m in all_matches]
        
        return messages
    
    @classmethod
    def parse_response(cls, text: str) -> Dict[str, Any]:
        """
        Parse GPT-OSS response - returns only the NEW response part
        (for backward compatibility)
        
        Returns:
            {
                "analysis": str or None,
                "commentary": str or None,
                "final": str,
                "raw": str
            }
        """
        full = cls.parse_full_response(text)
        
        return {
            "analysis": full["response"]["analysis"],
            "commentary": full["response"]["commentary"],
            "final": full["response"]["final"] or "",
            "raw": text
        }


# ============================================================================
# Request/Response Models (OpenAI-Compatible)
# ============================================================================

class Message(BaseModel):
    role: str = Field(..., description="Role: system, user, assistant, developer")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the participant")
    channel: Optional[str] = Field(None, description="Channel for assistant messages (analysis, commentary, final)")


class ResponseFormat(BaseModel):
    type: ResponseFormatType = Field(ResponseFormatType.TEXT, description="Response format type")
    json_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for structured output")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with GPT-OSS extensions"""
    messages: List[Message] = Field(..., description="Array of message objects")
    model: Optional[str] = Field(None, description="Model identifier (optional, uses loaded model)")
    
    # Core parameters
    max_tokens: int = Field(1000, ge=1, le=4096, description="Maximum response tokens")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(False, description="Enable streaming response")
    
    # Advanced parameters
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: int = Field(50, ge=0, le=100, description="Top-k sampling")
    stop: Optional[List[str]] = Field(None, max_items=4, description="Stop sequences")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logprobs: bool = Field(False, description="Return token log probabilities")
    
    # Structured output
    response_format: Optional[ResponseFormat] = Field(None, description="Response format specification")
    
    # GPT-OSS specific parameters
    reasoning: ReasoningLevel = Field(ReasoningLevel.MEDIUM, description="Reasoning level: low, medium, high")
    separate_thinking: bool = Field(True, description="Separate thinking channels from final answer")
    include_channels: List[str] = Field(
        default=["analysis", "commentary", "final"],
        description="Which channels to include in response"
    )
    
    # Developer instructions (separate from system prompt)
    developer_instructions: Optional[str] = Field(None, description="Developer-level instructions")
    
    # Request metadata
    user: Optional[str] = Field(None, description="User identifier for tracking")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages array cannot be empty")
        return v
    
    @validator('include_channels')
    def validate_channels(cls, v):
        valid = {"analysis", "commentary", "final"}
        for ch in v:
            if ch not in valid:
                raise ValueError(f"Invalid channel: {ch}. Must be one of {valid}")
        return v


class ChannelContent(BaseModel):
    """Content from a specific channel"""
    channel: str
    content: str


class ThinkingContent(BaseModel):
    """Separated thinking content from analysis and commentary channels"""
    analysis: Optional[str] = Field(None, description="Analysis channel content")
    commentary: Optional[str] = Field(None, description="Commentary channel content")
    raw: str = Field("", description="Combined raw thinking text")


class HistoryMessage(BaseModel):
    """A message from conversation history"""
    role: str
    content: str
    channel: Optional[str] = None


class ParsedResponse(BaseModel):
    """Fully parsed GPT-OSS response"""
    # Auto-generated by model (usually ignored)
    system_prompt: Optional[str] = Field(None, description="Auto-generated system prompt (ignored)")
    # User's system instructions converted to developer
    developer_instructions: Optional[str] = Field(None, description="Developer instructions (from user's system role)")
    # Conversation history echoed back
    history: List[HistoryMessage] = Field(default_factory=list, description="Conversation history")
    # The actual new response
    analysis: Optional[str] = Field(None, description="Analysis/reasoning channel")
    commentary: Optional[str] = Field(None, description="Commentary channel")
    final: Optional[str] = Field(None, description="Final answer")


class ChatMessage(BaseModel):
    """Response message with full parsing"""
    role: str = "assistant"
    content: str = Field(..., description="Final answer content")
    thinking: Optional[ThinkingContent] = Field(None, description="Separated thinking content")
    # Full parsed structure for GPT-OSS
    parsed: Optional[ParsedResponse] = Field(None, description="Full parsed response structure")


class Choice(BaseModel):
    """Response choice"""
    index: int = 0
    message: ChatMessage
    finish_reason: str = Field("stop", description="Reason for stopping: stop, length, content_filter")
    logprobs: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: Optional[int] = None
    channel_tokens: Optional[Dict[str, int]] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None
    reasoning_level: Optional[str] = None


class StreamChoice(BaseModel):
    """Streaming response choice"""
    index: int = 0
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Streaming response chunk"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    channel: Optional[str] = None  # Current channel being streamed


class ModelInfo(BaseModel):
    """Model information response"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"
    model_type: str
    max_context_length: int
    capabilities: Dict[str, bool]
    supported_channels: List[str]
    supported_reasoning_levels: List[str]


class QueueStatus(BaseModel):
    """Queue status response"""
    queue_length: int
    active_requests: int
    max_concurrent: int
    estimated_wait_seconds: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str
    model_type: str
    device: str
    queue_status: QueueStatus


# ============================================================================
# Request Queue Manager
# ============================================================================

@dataclass
class QueuedRequest:
    """A request in the processing queue"""
    request_id: str
    request: ChatCompletionRequest
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)
    priority: int = 0


class RequestQueueManager:
    """Manages concurrent request processing with queue"""
    
    def __init__(self, max_concurrent: int = 2, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.active_count = 0
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def start(self):
        """Start the queue processor"""
        self._processing = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info(f"Queue manager started with max_concurrent={self.max_concurrent}")
        
    async def stop(self):
        """Stop the queue processor"""
        self._processing = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Queue manager stopped")
    
    async def submit(self, request: ChatCompletionRequest) -> asyncio.Future:
        """Submit a request to the queue"""
        if self.queue.qsize() >= self.max_queue_size:
            raise HTTPException(status_code=503, detail="Queue is full, try again later")
        
        future = asyncio.get_event_loop().create_future()
        queued = QueuedRequest(
            request_id=str(uuid.uuid4()),
            request=request,
            future=future
        )
        await self.queue.put(queued)
        logger.debug(f"Request {queued.request_id} queued, queue size: {self.queue.qsize()}")
        return future
    
    async def _process_queue(self):
        """Background task to process queued requests"""
        while self._processing:
            try:
                queued = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                asyncio.create_task(self._handle_request(queued))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
    
    async def _handle_request(self, queued: QueuedRequest):
        """Handle a single queued request"""
        async with self.semaphore:
            async with self._lock:
                self.active_count += 1
            try:
                queued.future.set_result(True)
            except Exception as e:
                queued.future.set_exception(e)
            finally:
                async with self._lock:
                    self.active_count -= 1
    
    def get_status(self) -> QueueStatus:
        """Get current queue status"""
        queue_len = self.queue.qsize()
        estimated_wait = queue_len * 2.0 / max(self.max_concurrent, 1)
        return QueueStatus(
            queue_length=queue_len,
            active_requests=self.active_count,
            max_concurrent=self.max_concurrent,
            estimated_wait_seconds=estimated_wait
        )


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self._lock = asyncio.Lock()
        
    async def load(self):
        """Load the model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_path}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, use_fast=False
            )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on type
        load_kwargs = {
            "torch_dtype": self.config.dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": True,
        }
        
        if self.config.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        try:
            if self.config.model_type == ModelType.GEMMA3 and HAS_GEMMA3:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.config.model_path, **load_kwargs
                ).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path, **load_kwargs
                ).eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on device: {self.device}")
        
    def _build_prompt_gpt_oss(self, request: ChatCompletionRequest) -> str:
        """Build prompt in GPT-OSS format with channels"""
        parts = []
        
        # Add developer instructions if provided (before other messages)
        if request.developer_instructions:
            parts.append(GPTOSSTokens.format_developer_instructions(
                request.developer_instructions
            ))
        
        # Add messages - model auto-generates system prompt, we just format user messages
        for msg in request.messages:
            if msg.role == "system":
                # User-provided system instructions (optional override)
                parts.append(GPTOSSTokens.format_message("system", msg.content))
            elif msg.role == "developer":
                parts.append(GPTOSSTokens.format_developer_instructions(msg.content))
            elif msg.role == "user":
                parts.append(GPTOSSTokens.format_message("user", msg.content))
            elif msg.role == "assistant":
                channel = msg.channel or "final"
                parts.append(GPTOSSTokens.format_message("assistant", msg.content, channel))
        
        # Add generation prompt - start assistant response
        parts.append(f"{GPTOSSTokens.START}assistant{GPTOSSTokens.CHANNEL}")
        
        return "".join(parts)
    
    def _build_prompt_generic(self, messages: List[Message]) -> str:
        """Build prompt using tokenizer's chat template or fallback"""
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                chat_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            prompt = self._fallback_prompt_format(chat_messages)
        
        return prompt
    
    def _fallback_prompt_format(self, messages: List[Dict]) -> str:
        """Fallback prompt formatting"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant: ")
        return "".join(parts)
    
    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        """Build prompt based on model type"""
        if self.config.model_type == ModelType.GPT_OSS:
            return self._build_prompt_gpt_oss(request)
        else:
            return self._build_prompt_generic(request.messages)
    
    def _parse_response(self, text: str, request: ChatCompletionRequest) -> tuple[str, Optional[ThinkingContent], Optional[ParsedResponse]]:
        """Parse response based on model type"""
        
        if self.config.model_type == ModelType.GPT_OSS:
            # Use full parser
            full = GPTOSSTokens.parse_full_response(text)
            
            # Build thinking content from analysis and commentary
            thinking = None
            response = full["response"]
            
            if request.separate_thinking and (response["analysis"] or response["commentary"]):
                raw_parts = []
                if response["analysis"]:
                    raw_parts.append(f"[Analysis]\n{response['analysis']}")
                if response["commentary"]:
                    raw_parts.append(f"[Commentary]\n{response['commentary']}")
                
                thinking = ThinkingContent(
                    analysis=response["analysis"],
                    commentary=response["commentary"],
                    raw="\n\n".join(raw_parts)
                )
            
            # Final content
            final_content = response["final"] or ""
            
            # Build parsed response structure
            parsed = ParsedResponse(
                system_prompt=full["system_prompt"],
                developer_instructions=full["developer_instructions"],
                history=[HistoryMessage(**h) for h in full["history"]],
                analysis=response["analysis"],
                commentary=response["commentary"],
                final=response["final"]
            )
            
            return final_content, thinking, parsed
        
        else:
            # Generic parsing - look for common thinking patterns
            thinking = None
            final_content = text
            
            # Try to find <think>...</think> blocks
            think_pattern = r'<think>(.*?)</think>'
            matches = re.findall(think_pattern, text, re.DOTALL)
            
            if matches:
                thinking_raw = "\n".join(matches)
                final_content = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()
                thinking = ThinkingContent(
                    analysis=thinking_raw,
                    commentary=None,
                    raw=thinking_raw
                )
            
            return final_content, thinking, None
    
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a completion"""
        async with self._lock:
            return await self._generate_internal(request)
    
    async def _generate_internal(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Internal generation logic"""
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(request)
        logger.debug(f"Built prompt ({len(prompt)} chars)")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": max(request.temperature, 0.01),
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add stop sequences
        stop_sequences = request.stop or []
        if self.config.model_type == ModelType.GPT_OSS:
            # Add GPT-OSS specific stop tokens
            stop_sequences.extend([GPTOSSTokens.RETURN, "<|end|>"])
        
        if stop_sequences:
            stop_token_ids = []
            for seq in stop_sequences:
                tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                if tokens:
                    stop_token_ids.extend(tokens)
            if stop_token_ids:
                existing_eos = self.tokenizer.eos_token_id
                if isinstance(existing_eos, int):
                    existing_eos = [existing_eos]
                elif existing_eos is None:
                    existing_eos = []
                gen_kwargs["eos_token_id"] = list(set(existing_eos + stop_token_ids))
        
        # Add repetition penalty
        if request.presence_penalty != 0 or request.frequency_penalty != 0:
            penalty = 1.0 + abs(request.presence_penalty) + abs(request.frequency_penalty)
            gen_kwargs["repetition_penalty"] = penalty
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, **gen_kwargs)
        
        # Decode
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        completion_tokens = len(generated_ids)
        
        # Clean up special tokens at the end
        for token in ["<|end|>", "<|return|>", "</s>", "<|endoftext|>"]:
            generated_text = generated_text.replace(token, "")
        generated_text = generated_text.strip()
        
        logger.debug(f"Generated text: {generated_text[:200]}...")
        
        # Determine finish reason
        finish_reason = "stop"
        if completion_tokens >= request.max_tokens:
            finish_reason = "length"
        
        # Parse response
        final_content, thinking, parsed = self._parse_response(generated_text, request)
        
        # Calculate channel tokens
        channel_tokens = {}
        thinking_tokens_count = 0
        
        if parsed:
            if parsed.analysis:
                ch_token_count = len(self.tokenizer.encode(parsed.analysis))
                channel_tokens["analysis"] = ch_token_count
                thinking_tokens_count += ch_token_count
            if parsed.commentary:
                ch_token_count = len(self.tokenizer.encode(parsed.commentary))
                channel_tokens["commentary"] = ch_token_count
                thinking_tokens_count += ch_token_count
            if parsed.final:
                channel_tokens["final"] = len(self.tokenizer.encode(parsed.final))
        
        # Handle response format
        if request.response_format:
            if request.response_format.type == ResponseFormatType.JSON_OBJECT:
                json_match = re.search(r'\{.*\}', final_content, re.DOTALL)
                if json_match:
                    try:
                        json.loads(json_match.group())
                        final_content = json_match.group()
                    except json.JSONDecodeError:
                        pass
        
        # Build response
        message = ChatMessage(
            role="assistant",
            content=final_content,
            thinking=thinking,
            parsed=parsed if request.separate_thinking else None
        )
        
        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason
        )
        
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            thinking_tokens=thinking_tokens_count if thinking_tokens_count > 0 else None,
            channel_tokens=channel_tokens if channel_tokens else None
        )
        
        return ChatCompletionResponse(
            model=self.config.model_name,
            choices=[choice],
            usage=usage,
            system_fingerprint=f"{self.config.model_type.value}-{self.device}",
            reasoning_level=request.reasoning.value
        )
    
    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Generate a streaming completion"""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        # Build prompt
        prompt = self._build_prompt(request)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": max(request.temperature, 0.01),
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Generate all at once (for simplicity)
        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, **gen_kwargs)
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # Parse into channels (this also cleans the response)
        if self.config.model_type == ModelType.GPT_OSS:
            full = GPTOSSTokens.parse_full_response(full_text)
            response = full["response"]
            
            # Stream each channel separately (only the NEW response, not history)
            for channel in ["analysis", "commentary", "final"]:
                content = response.get(channel)
                if content and channel in request.include_channels:
                    # Send channel start marker
                    yield self._format_stream_chunk(response_id, created, "", channel, channel_start=True)
                    
                    # Stream content word by word
                    words = content.split()
                    for i, word in enumerate(words):
                        chunk_content = word + (" " if i < len(words) - 1 else "")
                        yield self._format_stream_chunk(response_id, created, chunk_content, channel)
                        await asyncio.sleep(0.02)
                    
                    # Send channel end marker
                    yield self._format_stream_chunk(response_id, created, "", channel, channel_end=True)
        else:
            # Generic streaming
            words = full_text.split()
            for i, word in enumerate(words):
                chunk_content = word + (" " if i < len(words) - 1 else "")
                yield self._format_stream_chunk(response_id, created, chunk_content, "content")
                await asyncio.sleep(0.02)
        
        # Final chunk
        yield self._format_stream_chunk(response_id, created, "", None, finish_reason="stop")
        yield "data: [DONE]\n\n"
    
    def _format_stream_chunk(
        self, 
        response_id: str, 
        created: int, 
        content: str, 
        channel: Optional[str],
        finish_reason: Optional[str] = None,
        channel_start: bool = False,
        channel_end: bool = False
    ) -> str:
        """Format a streaming chunk"""
        delta = {}
        
        if channel_start:
            delta["channel_start"] = channel
        elif channel_end:
            delta["channel_end"] = channel
        elif content:
            if channel in ["analysis", "commentary"]:
                delta["thinking"] = content
                delta["thinking_channel"] = channel
            else:
                delta["content"] = content
        
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.config.model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        
        if channel:
            chunk["current_channel"] = channel
        
        return f"data: {json.dumps(chunk)}\n\n"
    
    def get_info(self) -> ModelInfo:
        """Get model information"""
        return ModelInfo(
            id=self.config.model_name,
            created=int(time.time()),
            model_type=self.config.model_type.value,
            max_context_length=self.config.max_context_length,
            capabilities={
                "streaming": True,
                "thinking_separation": True,
                "json_mode": True,
                "channels": self.config.model_type == ModelType.GPT_OSS,
                "reasoning_levels": self.config.model_type == ModelType.GPT_OSS,
                "function_calling": False
            },
            supported_channels=GPTOSSTokens.VALID_CHANNELS if self.config.model_type == ModelType.GPT_OSS else ["final"],
            supported_reasoning_levels=[r.value for r in ReasoningLevel]
        )


# ============================================================================
# FastAPI Application
# ============================================================================

model_manager: Optional[ModelManager] = None
queue_manager: Optional[RequestQueueManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_manager, queue_manager
    
    import os
    
    model_path = os.environ.get("MODEL_PATH", "I:/gpt-oss-20b")
    model_type = ModelType(os.environ.get("MODEL_TYPE", "gpt-oss"))
    model_name = os.environ.get("MODEL_NAME", "gpt-oss-20b")
    device_map = os.environ.get("DEVICE_MAP", "auto")
    max_concurrent = int(os.environ.get("MAX_CONCURRENT", "2"))
    
    config = ModelConfig(
        model_path=model_path,
        model_type=model_type,
        model_name=model_name,
        device_map=device_map,
    )
    
    model_manager = ModelManager(config)
    queue_manager = RequestQueueManager(max_concurrent=max_concurrent)
    
    await model_manager.load()
    await queue_manager.start()
    
    logger.info(f"Application started - Model: {model_name}, Type: {model_type.value}")
    
    yield
    
    await queue_manager.stop()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Local LLM API Server (GPT-OSS Compatible)",
    description="OpenAI-compatible API with GPT-OSS channel support and reasoning levels",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager is not None and model_manager.model is not None,
        model_name=model_manager.config.model_name if model_manager else "none",
        model_type=model_manager.config.model_type.value if model_manager else "none",
        device=str(model_manager.device) if model_manager else "none",
        queue_status=queue_manager.get_status() if queue_manager else QueueStatus(
            queue_length=0, active_requests=0, max_concurrent=0, estimated_wait_seconds=0
        )
    )


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "object": "list",
        "data": [model_manager.get_info().model_dump()]
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model information (OpenAI-compatible)"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if model_id != model_manager.config.model_name:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model_manager.get_info().model_dump()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint (OpenAI-compatible with GPT-OSS extensions)
    
    GPT-OSS specific parameters:
    - reasoning: "low", "medium", "high" - controls reasoning depth
    - include_channels: ["analysis", "commentary", "final"] - which channels to include
    - separate_thinking: true/false - separate thinking from final answer
    - developer_instructions: string - developer-level instructions
    """
    if not model_manager or not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    await queue_manager.submit(request)
    
    if request.stream:
        return StreamingResponse(
            model_manager.generate_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        response = await model_manager.generate(request)
        return response.model_dump()


@app.get("/queue/status", response_model=QueueStatus)
async def get_queue_status():
    """Get current queue status"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    return queue_manager.get_status()


# ============================================================================
# Legacy Endpoints (Backward Compatibility)
# ============================================================================

class LegacyInferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100


class LegacyChatRequest(BaseModel):
    history: List[Dict[str, str]]
    max_new_tokens: int = 100


@app.post("/generate")
async def legacy_generate(request: LegacyInferenceRequest):
    """Legacy generate endpoint"""
    chat_request = ChatCompletionRequest(
        messages=[Message(role="user", content=request.prompt)],
        max_tokens=request.max_new_tokens,
        separate_thinking=False,
        reasoning=ReasoningLevel.MEDIUM
    )
    response = await model_manager.generate(chat_request)
    return {"generated_text": response.choices[0].message.content}


@app.post("/simple_chat")
async def legacy_simple_chat(request: LegacyInferenceRequest):
    """Legacy simple chat endpoint"""
    chat_request = ChatCompletionRequest(
        messages=[Message(role="user", content=request.prompt)],
        max_tokens=request.max_new_tokens,
        reasoning=ReasoningLevel.MEDIUM
    )
    response = await model_manager.generate(chat_request)
    msg = response.choices[0].message
    
    result = {
        "response": msg.content,
        "thinking": msg.thinking.model_dump() if msg.thinking else None,
    }
    
    # Add parsed structure for GPT-OSS
    if msg.parsed:
        result["parsed"] = msg.parsed.model_dump()
        result["history"] = [h.model_dump() for h in msg.parsed.history]
        result["developer_instructions"] = msg.parsed.developer_instructions
        result["analysis"] = msg.parsed.analysis
        result["commentary"] = msg.parsed.commentary
        result["final"] = msg.parsed.final
    
    return result


@app.post("/chat")
async def legacy_chat(request: LegacyChatRequest):
    """Legacy chat endpoint with history"""
    messages = [Message(role=m["role"], content=m["content"]) for m in request.history]
    chat_request = ChatCompletionRequest(
        messages=messages,
        max_tokens=request.max_new_tokens,
        reasoning=ReasoningLevel.MEDIUM
    )
    response = await model_manager.generate(chat_request)
    msg = response.choices[0].message
    
    result = {
        "response": msg.content,
        "thinking": msg.thinking.model_dump() if msg.thinking else None,
    }
    
    # Add parsed structure for GPT-OSS
    if msg.parsed:
        result["parsed"] = msg.parsed.model_dump()
        result["history"] = [h.model_dump() for h in msg.parsed.history]
        result["developer_instructions"] = msg.parsed.developer_instructions
        result["analysis"] = msg.parsed.analysis
        result["commentary"] = msg.parsed.commentary
        result["final"] = msg.parsed.final
    
    return result


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Local LLM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", default="I:/gpt-oss-20b", help="Path to model")
    parser.add_argument("--model-type", default="gpt-oss", 
                        choices=["gpt-oss", "gemma3", "llama", "mistral", "generic"])
    parser.add_argument("--model-name", default="gpt-oss-20b", help="Model name for API")
    parser.add_argument("--device-map", default="auto", help="Device map")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["MODEL_TYPE"] = args.model_type
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
