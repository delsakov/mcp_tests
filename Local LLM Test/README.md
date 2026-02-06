# Local LLM API Infrastructure

A complete solution for serving multiple local LLM models with an OpenAI-compatible API, thinking/answer separation, and load balancing.

## Architecture

```
                    ┌─────────────────────┐
                    │   Your Application  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    │   (port 8080)       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  GPT-OSS-20B    │ │  Gemma3-4B      │ │  Other Model    │
    │  (port 8001)    │ │  (port 8002)    │ │  (port 800X)    │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Features

### Model Server (`model_server.py`)
- **OpenAI-compatible API** - Drop-in replacement for OpenAI API
- **Thinking/Answer Separation** - Extracts `<think>...</think>` blocks into separate fields
- **Queue Management** - Handles concurrent requests with configurable limits
- **Streaming Support** - SSE streaming for real-time responses
- **Multiple Response Formats** - text, json_object, json_schema
- **SafeTensor Support** - Works with HuggingFace SafeTensor models

### Load Balancer (`load_balancer.py`)
- **Multiple Strategies** - Round-robin, least-connections, weighted, sticky sessions
- **Health Monitoring** - Automatic health checks with failover
- **Dynamic Backend Management** - Register/unregister backends at runtime
- **Request Routing** - Routes by model name to appropriate backend

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start a Single Model Server

```bash
# Start GPT-OSS-20B server
python model_server.py \
    --model-path "I:/gpt-oss-20b" \
    --model-type gpt-oss \
    --model-name gpt-oss-20b \
    --port 8001

# Or start Gemma3 server
python model_server.py \
    --model-path "I:/google/gemma-3-4b-it" \
    --model-type gemma3 \
    --model-name gemma-3-4b-it \
    --port 8002
```

### 3. Start Load Balancer (Optional)

```bash
python load_balancer.py --port 8080 --strategy least_connections
```

### 4. Register Backends

```bash
# Register GPT-OSS backend
curl -X POST http://localhost:8080/backends/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpt-oss-20b",
    "url": "http://localhost:8001",
    "model_type": "gpt-oss",
    "models": ["gpt-oss-20b", "gpt-oss"]
  }'

# Register Gemma3 backend
curl -X POST http://localhost:8080/backends/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gemma-3-4b-it",
    "url": "http://localhost:8002",
    "model_type": "gemma3",
    "models": ["gemma-3-4b-it", "gemma3"]
  }'
```

## API Usage

### Chat Completion (OpenAI-compatible with GPT-OSS extensions)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "Always answer in just single word."},
      {"role": "user", "content": "What is the capital of the US?"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7,
    "reasoning": "medium",
    "separate_thinking": true,
    "include_channels": ["analysis", "commentary", "final"]
  }'
```

### GPT-OSS Specific Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `reasoning` | `"low"`, `"medium"`, `"high"` | Controls depth of reasoning in analysis channel |
| `include_channels` | `["analysis", "commentary", "final"]` | Which channels to include in response |
| `separate_thinking` | `true`/`false` | Separate thinking (analysis/commentary) from final answer |
| `developer_instructions` | string | Developer-level instructions (separate from system prompt) |

### GPT-OSS Channel Format

The model uses special tokens for structured output:
- `|start|>assistant<|channel|>analysis<|message|>` - Reasoning/thinking
- `|start|>assistant<|channel|>commentary<|message|>` - Additional commentary  
- `|start|>assistant<|channel|>final<|message|>` - Final answer

### Response Format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-oss-20b",
  "reasoning_level": "medium",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Washington",
        "thinking": {
          "analysis": "User asks: \"What's capital of the US?\" They got answer \"Atlanta\" (incorrect). Then user says \"It's not correct...\" We must answer with a single word only. The correct response: \"Washington\" or \"Washington, D.C.\" But a single word: \"Washington\"",
          "commentary": "The policy states that the assistant must be honest and accurate. It is not allowed to fabricate or lie. Provide the correct answer.",
          "raw": "[Analysis]\nUser asks: \"What's capital...\"[Commentary]\nThe policy states..."
        },
        "channels": {
          "analysis": "User asks: \"What's capital of the US?\"...",
          "commentary": "The policy states that the assistant must be honest...",
          "final": "Washington"
        }
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 125,
    "completion_tokens": 250,
    "total_tokens": 375,
    "thinking_tokens": 200,
    "channel_tokens": {
      "analysis": 150,
      "commentary": 50,
      "final": 5
    }
  },
  "_routing": {
    "backend": "gpt-oss-20b",
    "model_type": "gpt-oss"
  }
}
```

### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Streaming response:
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### JSON Mode

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "List 3 colors as JSON array"}],
    "response_format": {"type": "json_object"}
  }'
```

## Python Client Example

```python
import httpx

# Simple client for local LLM API
class LocalLLMClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
    
    def chat(
        self,
        messages: list,
        model: str = "gpt-oss-20b",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        separate_thinking: bool = True
    ):
        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "separate_thinking": separate_thinking
            }
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self):
        response = self.client.get(f"{self.base_url}/v1/models")
        return response.json()
    
    def get_status(self):
        response = self.client.get(f"{self.base_url}/status")
        return response.json()


# Usage
client = LocalLLMClient()

# List available models
models = client.list_models()
print("Available models:", [m["id"] for m in models["data"]])

# Chat with thinking separation
result = client.chat(
    messages=[
        {"role": "system", "content": "Think through problems step by step."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    model="gpt-oss-20b"
)

print("Answer:", result["choices"][0]["message"]["content"])
if result["choices"][0]["message"].get("thinking"):
    print("Thinking:", result["choices"][0]["message"]["thinking"]["raw"])
```

## Async Python Client

```python
import httpx
import asyncio

async def chat_async():
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 100
            }
        )
        return response.json()

# Run
result = asyncio.run(chat_async())
print(result)
```

## Configuration Options

### Model Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8000` | Port to bind to |
| `--model-path` | Required | Path to model directory |
| `--model-type` | `generic` | Model type: gemma3, gpt-oss, llama, mistral, generic |
| `--model-name` | Required | Model name for API |
| `--device-map` | `auto` | Device mapping strategy |
| `--max-concurrent` | `2` | Maximum concurrent requests |

### Load Balancer Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8080` | Port to bind to |
| `--strategy` | `least_connections` | Load balancing strategy |
| `--health-interval` | `30.0` | Health check interval in seconds |

### Load Balancing Strategies

- **round_robin** - Rotate through backends sequentially
- **least_connections** - Route to backend with fewest active requests
- **random** - Random backend selection
- **weighted** - Weighted random based on backend weights
- **sticky** - Same user always goes to same backend

## Thinking Separation

### GPT-OSS Models
GPT-OSS uses channel-based output with these tokens:
- `|start|>assistant<|channel|>analysis<|message|>` - Main reasoning
- `|start|>assistant<|channel|>commentary<|message|>` - Policy/meta commentary
- `|start|>assistant<|channel|>final<|message|>` - Final answer

The `reasoning` parameter controls depth:
- `low` - Minimal analysis, quick responses
- `medium` - Balanced reasoning (default)
- `high` - Deep analysis with detailed thinking

### Other Models
For non-GPT-OSS models, the system looks for `<think>...</think>` blocks.

Enable with `"separate_thinking": true` in requests.

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

### Cluster Status

```bash
curl http://localhost:8080/status
```

### Queue Status (per model server)

```bash
curl http://localhost:8001/queue/status
```

## Troubleshooting

### Model Loading Issues

1. **Out of Memory**: Try `--device-map cpu` or reduce model size
2. **SafeTensor Errors**: Ensure `safetensors` package is installed
3. **Tokenizer Errors**: Some models need `use_fast=False`

### Performance Tips

1. Use `torch.float16` for GPU inference
2. Set appropriate `--max-concurrent` based on VRAM
3. Enable Flash Attention 2 if supported:
   ```bash
   pip install flash-attn
   ```

### Common Errors

- **503 Service Unavailable**: No healthy backends - check model servers
- **504 Gateway Timeout**: Increase timeout in load balancer config
- **Queue Full**: Increase queue size or add more backends
