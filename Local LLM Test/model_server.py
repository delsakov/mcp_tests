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
    """Token definitions for GPT-OSS model format"""
    
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
        """Format the system prompt with reasoning level"""
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
    def parse_response(cls, text: str) -> Dict[str, Any]:
        """
        Parse GPT-OSS response into channels
        
        Returns:
            {
                "analysis": str or None,
                "commentary": str or None,
                "final": str,
                "raw": str
            }
        """
        result = {
            "analysis": None,
            "commentary": None,
            "final": None,
            "raw": text
        }
        
        # Pattern to match channel blocks
        # |start|>assistant<|channel|>CHANNEL_NAME<|message|>CONTENT<|end|>
        pattern = (
            r'\|start\|>assistant<\|channel\|>(\w+)<\|message\|>'
            r'(.*?)'
            r'(?:<\|end\|>|<\|return\|>|\|start\|>|$)'
        )
        
        matches = re.findall(pattern, text, re.DOTALL)
        
        for channel, content in matches:
            channel_lower = channel.lower()
            if channel_lower in cls.VALID_CHANNELS:
                # Clean up the content
                content = content.strip()
                # Remove any trailing tokens
                content = re.sub(r'<\|(?:end|return)\|>.*$', '', content, flags=re.DOTALL).strip()
                result[channel_lower] = content
        
        # If no structured output found, try simpler patterns
        if not any([result["analysis"], result["commentary"], result["final"]]):
            # Try to find just the final answer
            final_pattern = r'final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)'
            final_match = re.search(final_pattern, text, re.DOTALL)
            if final_match:
                result["final"] = final_match.group(1).strip()
            else:
                # Just use the whole text as final
                result["final"] = text.strip()
        
        return result


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


class ChatMessage(BaseModel):
    """Response message with channel separation"""
    role: str = "assistant"
    content: str = Field(..., description="Final answer content")
    thinking: Optional[ThinkingContent] = Field(None, description="Separated thinking content")
    channels: Optional[Dict[str, str]] = Field(None, description="All channel outputs")


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
        
        # Check if there's a system message in the request
        has_system = any(m.role == "system" for m in request.messages)
        
        # Add system prompt with reasoning level if not provided
        if not has_system:
            parts.append(GPTOSSTokens.format_system_prompt(
                instructions="",
                reasoning=request.reasoning
            ))
        
        # Add developer instructions if provided
        if request.developer_instructions:
            parts.append(GPTOSSTokens.format_developer_instructions(
                request.developer_instructions
            ))
        
        # Add messages
        for msg in request.messages:
            if msg.role == "system":
                # Format system with reasoning level
                content = msg.content
                if "Reasoning:" not in content:
                    content = f"{content}\n\nReasoning: {request.reasoning.value}"
                parts.append(GPTOSSTokens.format_message("system", content))
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
    
    def _parse_response(self, text: str, request: ChatCompletionRequest) -> tuple[str, Optional[ThinkingContent], Dict[str, str]]:
        """Parse response based on model type"""
        
        if self.config.model_type == ModelType.GPT_OSS:
            parsed = GPTOSSTokens.parse_response(text)
            
            # Build thinking content from analysis and commentary
            thinking = None
            if request.separate_thinking and (parsed["analysis"] or parsed["commentary"]):
                raw_parts = []
                if parsed["analysis"]:
                    raw_parts.append(f"[Analysis]\n{parsed['analysis']}")
                if parsed["commentary"]:
                    raw_parts.append(f"[Commentary]\n{parsed['commentary']}")
                
                thinking = ThinkingContent(
                    analysis=parsed["analysis"],
                    commentary=parsed["commentary"],
                    raw="\n\n".join(raw_parts)
                )
            
            # Final content
            final_content = parsed["final"] or ""
            
            # All channels
            channels = {
                k: v for k, v in parsed.items() 
                if k in request.include_channels and v is not None
            }
            
            return final_content, thinking, channels
        
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
            
            return final_content, thinking, {"final": final_content}
    
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
        final_content, thinking, channels = self._parse_response(generated_text, request)
        
        # Calculate channel tokens
        channel_tokens = {}
        thinking_tokens_count = 0
        for ch_name, ch_content in channels.items():
            if ch_content:
                ch_token_count = len(self.tokenizer.encode(ch_content))
                channel_tokens[ch_name] = ch_token_count
                if ch_name in ["analysis", "commentary"]:
                    thinking_tokens_count += ch_token_count
        
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
            channels=channels if request.separate_thinking else None
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
        
        # Parse into channels
        if self.config.model_type == ModelType.GPT_OSS:
            parsed = GPTOSSTokens.parse_response(full_text)
            
            # Stream each channel separately
            for channel in ["analysis", "commentary", "final"]:
                content = parsed.get(channel)
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
    return {
        "response": msg.content,
        "thinking": msg.thinking.model_dump() if msg.thinking else None,
        "channels": msg.channels
    }


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
    return {
        "response": msg.content,
        "thinking": msg.thinking.model_dump() if msg.thinking else None,
        "channels": msg.channels
    }


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
