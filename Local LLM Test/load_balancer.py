"""
LLM Load Balancer & Orchestrator
Routes requests to multiple local model backends
Features: Model selection, Health monitoring, Request distribution, Failover
"""

import asyncio
import time
import json
import logging
import random
import hashlib
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Models
# ============================================================================

class BackendStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DRAINING = "draining"  # Not accepting new requests


class LoadBalanceStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"
    STICKY = "sticky"  # Same user goes to same backend


@dataclass
class BackendConfig:
    """Configuration for a model backend"""
    name: str
    url: str
    model_type: str
    models: List[str]  # List of model names this backend serves
    weight: int = 1
    max_connections: int = 10
    timeout: float = 120.0
    health_check_interval: float = 30.0
    enabled: bool = True


@dataclass
class BackendState:
    """Runtime state for a backend"""
    config: BackendConfig
    status: BackendStatus = BackendStatus.UNKNOWN
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_health_check: float = 0
    last_error: Optional[str] = None
    response_times: List[float] = field(default_factory=list)
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times[-100:]) / len(self.response_times[-100:])
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests


# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ResponseFormat(BaseModel):
    type: str = "text"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """Unified chat completion request"""
    messages: List[Message]
    model: str = Field(..., description="Model name to use")
    
    # Core parameters
    max_tokens: int = Field(1000, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False
    
    # Advanced parameters
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=100)
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    
    # Structured output
    response_format: Optional[ResponseFormat] = None
    separate_thinking: bool = True
    
    # Routing hints
    user: Optional[str] = None  # For sticky sessions
    priority: int = Field(0, ge=0, le=10)  # Higher = more important


class BackendInfo(BaseModel):
    """Backend information response"""
    name: str
    url: str
    model_type: str
    models: List[str]
    status: str
    active_connections: int
    total_requests: int
    error_rate: float
    avg_response_time_ms: float


class ClusterStatus(BaseModel):
    """Overall cluster status"""
    healthy_backends: int
    total_backends: int
    total_active_connections: int
    available_models: List[str]
    backends: List[BackendInfo]


class RegisterBackendRequest(BaseModel):
    """Request to register a new backend"""
    name: str
    url: str
    model_type: str
    models: List[str]
    weight: int = 1
    max_connections: int = 10
    timeout: float = 120.0


# ============================================================================
# Load Balancer Core
# ============================================================================

class LoadBalancer:
    """Core load balancer logic"""
    
    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS,
        health_check_interval: float = 30.0
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.backends: Dict[str, BackendState] = {}
        self.model_to_backends: Dict[str, Set[str]] = defaultdict(set)
        self.round_robin_index = 0
        self.sticky_sessions: Dict[str, str] = {}  # user_id -> backend_name
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def start(self):
        """Start the load balancer"""
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Load balancer started with strategy: {self.strategy}")
    
    async def stop(self):
        """Stop the load balancer"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._http_client:
            await self._http_client.aclose()
        
        logger.info("Load balancer stopped")
    
    async def register_backend(self, config: BackendConfig) -> bool:
        """Register a new backend"""
        async with self._lock:
            if config.name in self.backends:
                # Update existing
                self.backends[config.name].config = config
            else:
                self.backends[config.name] = BackendState(config=config)
            
            # Update model mappings
            for model in config.models:
                self.model_to_backends[model].add(config.name)
            
            logger.info(f"Registered backend: {config.name} at {config.url}")
            
            # Immediate health check
            await self._check_backend_health(config.name)
            
            return True
    
    async def unregister_backend(self, name: str) -> bool:
        """Unregister a backend"""
        async with self._lock:
            if name not in self.backends:
                return False
            
            state = self.backends[name]
            
            # Remove from model mappings
            for model in state.config.models:
                self.model_to_backends[model].discard(name)
            
            del self.backends[name]
            logger.info(f"Unregistered backend: {name}")
            return True
    
    async def select_backend(self, model: str, user_id: Optional[str] = None) -> Optional[BackendState]:
        """Select a backend for the given model"""
        async with self._lock:
            # Get backends that serve this model
            backend_names = self.model_to_backends.get(model, set())
            
            if not backend_names:
                # Try to find by model type pattern
                for name, state in self.backends.items():
                    if model in state.config.models or model == state.config.model_type:
                        backend_names.add(name)
            
            if not backend_names:
                return None
            
            # Filter to healthy backends with capacity
            available = []
            for name in backend_names:
                state = self.backends.get(name)
                if state and self._is_available(state):
                    available.append(state)
            
            if not available:
                return None
            
            # Apply strategy
            return self._apply_strategy(available, user_id)
    
    def _is_available(self, state: BackendState) -> bool:
        """Check if a backend is available for requests"""
        return (
            state.config.enabled and
            state.status == BackendStatus.HEALTHY and
            state.active_connections < state.config.max_connections
        )
    
    def _apply_strategy(
        self, 
        available: List[BackendState], 
        user_id: Optional[str]
    ) -> BackendState:
        """Apply load balancing strategy"""
        
        if self.strategy == LoadBalanceStrategy.STICKY and user_id:
            # Check for existing session
            if user_id in self.sticky_sessions:
                backend_name = self.sticky_sessions[user_id]
                for state in available:
                    if state.config.name == backend_name:
                        return state
            
            # Create new sticky session
            selected = self._select_least_connections(available)
            self.sticky_sessions[user_id] = selected.config.name
            return selected
        
        elif self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._select_round_robin(available)
        
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(available)
        
        elif self.strategy == LoadBalanceStrategy.WEIGHTED:
            return self._select_weighted(available)
        
        else:  # RANDOM
            return random.choice(available)
    
    def _select_round_robin(self, available: List[BackendState]) -> BackendState:
        """Round-robin selection"""
        self.round_robin_index = (self.round_robin_index + 1) % len(available)
        return available[self.round_robin_index]
    
    def _select_least_connections(self, available: List[BackendState]) -> BackendState:
        """Select backend with least active connections"""
        return min(available, key=lambda s: s.active_connections)
    
    def _select_weighted(self, available: List[BackendState]) -> BackendState:
        """Weighted random selection"""
        total_weight = sum(s.config.weight for s in available)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for state in available:
            cumulative += state.config.weight
            if r <= cumulative:
                return state
        return available[-1]
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for name in list(self.backends.keys()):
                    await self._check_backend_health(name)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _check_backend_health(self, name: str):
        """Check health of a specific backend"""
        if name not in self.backends:
            return
        
        state = self.backends[name]
        url = f"{state.config.url}/health"
        
        try:
            response = await self._http_client.get(url, timeout=5.0)
            
            if response.status_code == 200:
                state.status = BackendStatus.HEALTHY
                state.last_error = None
            else:
                state.status = BackendStatus.UNHEALTHY
                state.last_error = f"HTTP {response.status_code}"
                
        except Exception as e:
            state.status = BackendStatus.UNHEALTHY
            state.last_error = str(e)
        
        state.last_health_check = time.time()
        logger.debug(f"Health check for {name}: {state.status}")
    
    async def increment_connections(self, name: str):
        """Increment active connections for a backend"""
        async with self._lock:
            if name in self.backends:
                self.backends[name].active_connections += 1
    
    async def decrement_connections(self, name: str, response_time: float, error: bool = False):
        """Decrement active connections and record metrics"""
        async with self._lock:
            if name in self.backends:
                state = self.backends[name]
                state.active_connections = max(0, state.active_connections - 1)
                state.total_requests += 1
                state.response_times.append(response_time)
                
                if error:
                    state.total_errors += 1
    
    def get_status(self) -> ClusterStatus:
        """Get current cluster status"""
        healthy = sum(1 for s in self.backends.values() if s.status == BackendStatus.HEALTHY)
        total_connections = sum(s.active_connections for s in self.backends.values())
        
        all_models = set()
        for state in self.backends.values():
            if state.status == BackendStatus.HEALTHY:
                all_models.update(state.config.models)
        
        backend_infos = [
            BackendInfo(
                name=s.config.name,
                url=s.config.url,
                model_type=s.config.model_type,
                models=s.config.models,
                status=s.status.value,
                active_connections=s.active_connections,
                total_requests=s.total_requests,
                error_rate=s.error_rate,
                avg_response_time_ms=s.avg_response_time * 1000
            )
            for s in self.backends.values()
        ]
        
        return ClusterStatus(
            healthy_backends=healthy,
            total_backends=len(self.backends),
            total_active_connections=total_connections,
            available_models=list(all_models),
            backends=backend_infos
        )


# ============================================================================
# Request Proxy
# ============================================================================

class RequestProxy:
    """Proxies requests to backends"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.lb = load_balancer
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def start(self):
        """Start the proxy"""
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    
    async def stop(self):
        """Stop the proxy"""
        if self._http_client:
            await self._http_client.aclose()
    
    async def proxy_chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> Response:
        """Proxy a chat completion request"""
        
        # Select backend
        backend = await self.lb.select_backend(request.model, request.user)
        
        if not backend:
            raise HTTPException(
                status_code=503,
                detail=f"No healthy backend available for model: {request.model}"
            )
        
        # Prepare request
        url = f"{backend.config.url}/v1/chat/completions"
        payload = request.model_dump(exclude_none=True)
        
        await self.lb.increment_connections(backend.config.name)
        start_time = time.time()
        error = False
        
        try:
            if request.stream:
                return await self._proxy_streaming(backend, url, payload)
            else:
                return await self._proxy_non_streaming(backend, url, payload)
                
        except Exception as e:
            error = True
            logger.error(f"Proxy error to {backend.config.name}: {e}")
            raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")
            
        finally:
            response_time = time.time() - start_time
            await self.lb.decrement_connections(backend.config.name, response_time, error)
    
    async def _proxy_non_streaming(
        self, 
        backend: BackendState, 
        url: str, 
        payload: dict
    ) -> JSONResponse:
        """Proxy non-streaming request"""
        response = await self._http_client.post(
            url,
            json=payload,
            timeout=backend.config.timeout
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )
        
        data = response.json()
        
        # Add routing info
        data["_routing"] = {
            "backend": backend.config.name,
            "model_type": backend.config.model_type
        }
        
        return JSONResponse(content=data)
    
    async def _proxy_streaming(
        self, 
        backend: BackendState, 
        url: str, 
        payload: dict
    ) -> StreamingResponse:
        """Proxy streaming request"""
        
        async def stream_generator():
            async with self._http_client.stream(
                "POST",
                url,
                json=payload,
                timeout=backend.config.timeout
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"data: {json.dumps({'error': error_text.decode()})}\n\n"
                    return
                
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Backend": backend.config.name
            }
        )


# ============================================================================
# FastAPI Application
# ============================================================================

# Global instances
load_balancer: Optional[LoadBalancer] = None
request_proxy: Optional[RequestProxy] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global load_balancer, request_proxy
    
    import os
    
    strategy = LoadBalanceStrategy(os.environ.get("LB_STRATEGY", "least_connections"))
    health_interval = float(os.environ.get("HEALTH_CHECK_INTERVAL", "30"))
    
    load_balancer = LoadBalancer(strategy=strategy, health_check_interval=health_interval)
    request_proxy = RequestProxy(load_balancer)
    
    await load_balancer.start()
    await request_proxy.start()
    
    # Register default backends from environment
    default_backends = os.environ.get("DEFAULT_BACKENDS", "")
    if default_backends:
        for backend_str in default_backends.split(";"):
            try:
                parts = backend_str.split(",")
                if len(parts) >= 4:
                    config = BackendConfig(
                        name=parts[0],
                        url=parts[1],
                        model_type=parts[2],
                        models=parts[3].split("|")
                    )
                    await load_balancer.register_backend(config)
            except Exception as e:
                logger.error(f"Failed to register default backend: {e}")
    
    logger.info("Load balancer application started")
    
    yield
    
    await request_proxy.stop()
    await load_balancer.stop()
    logger.info("Load balancer application stopped")


app = FastAPI(
    title="LLM Load Balancer",
    description="Load balancer and orchestrator for multiple local LLM backends",
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


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = load_balancer.get_status()
    return {
        "status": "healthy" if status.healthy_backends > 0 else "degraded",
        "healthy_backends": status.healthy_backends,
        "total_backends": status.total_backends
    }


@app.get("/status", response_model=ClusterStatus)
async def get_cluster_status():
    """Get detailed cluster status"""
    return load_balancer.get_status()


@app.get("/v1/models")
async def list_models():
    """List all available models across backends"""
    status = load_balancer.get_status()
    
    models = []
    for backend in status.backends:
        if backend.status == "healthy":
            for model in backend.models:
                models.append({
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": backend.name,
                    "backend": backend.name,
                    "model_type": backend.model_type
                })
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint - routes to appropriate backend
    
    The 'model' field determines which backend handles the request.
    """
    return await request_proxy.proxy_chat_completion(request)


# ============================================================================
# Backend Management Endpoints
# ============================================================================

@app.post("/backends/register")
async def register_backend(request: RegisterBackendRequest):
    """Register a new backend"""
    config = BackendConfig(
        name=request.name,
        url=request.url,
        model_type=request.model_type,
        models=request.models,
        weight=request.weight,
        max_connections=request.max_connections,
        timeout=request.timeout
    )
    
    success = await load_balancer.register_backend(config)
    
    if success:
        return {"status": "registered", "backend": request.name}
    else:
        raise HTTPException(status_code=400, detail="Failed to register backend")


@app.delete("/backends/{name}")
async def unregister_backend(name: str):
    """Unregister a backend"""
    success = await load_balancer.unregister_backend(name)
    
    if success:
        return {"status": "unregistered", "backend": name}
    else:
        raise HTTPException(status_code=404, detail="Backend not found")


@app.post("/backends/{name}/drain")
async def drain_backend(name: str):
    """Set backend to draining mode (no new requests)"""
    if name not in load_balancer.backends:
        raise HTTPException(status_code=404, detail="Backend not found")
    
    load_balancer.backends[name].status = BackendStatus.DRAINING
    return {"status": "draining", "backend": name}


@app.post("/backends/{name}/enable")
async def enable_backend(name: str):
    """Enable a disabled backend"""
    if name not in load_balancer.backends:
        raise HTTPException(status_code=404, detail="Backend not found")
    
    load_balancer.backends[name].config.enabled = True
    return {"status": "enabled", "backend": name}


@app.post("/backends/{name}/disable")
async def disable_backend(name: str):
    """Disable a backend"""
    if name not in load_balancer.backends:
        raise HTTPException(status_code=404, detail="Backend not found")
    
    load_balancer.backends[name].config.enabled = False
    return {"status": "disabled", "backend": name}


# ============================================================================
# Convenience Endpoints
# ============================================================================

@app.post("/chat")
async def simple_chat(messages: List[Message], model: str, max_tokens: int = 1000):
    """Simplified chat endpoint"""
    request = ChatCompletionRequest(
        messages=messages,
        model=model,
        max_tokens=max_tokens
    )
    return await request_proxy.proxy_chat_completion(request)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="LLM Load Balancer")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--strategy", default="least_connections", 
                        choices=["round_robin", "least_connections", "random", "weighted", "sticky"])
    parser.add_argument("--health-interval", type=float, default=30.0, help="Health check interval")
    
    args = parser.parse_args()
    
    os.environ["LB_STRATEGY"] = args.strategy
    os.environ["HEALTH_CHECK_INTERVAL"] = str(args.health_interval)
    
    uvicorn.run(
        "load_balancer:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1
    )
