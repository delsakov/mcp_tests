# GPT-OSS API Server

OpenAI-compatible API wrapper for GPT-OSS local model.

## Architecture

```
┌─────────────────────┐
│   Your Application  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Load Balancer     │────▶│   GPT-OSS Server    │
│   (port 8080)       │     │   (port 8001)       │
└─────────────────────┘     └─────────────────────┘
```

## Quick Start

### 1. Start Model Server

```bash
python model_server.py \
    --model-path "I:/gpt-oss-20b" \
    --model-name gpt-oss-20b \
    --port 8001
```

### 2. Start Load Balancer (optional, for multiple backends)

```bash
python load_balancer.py --port 8080
```

### 3. Register Backend

```bash
curl -X POST http://localhost:8080/backends/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpt-oss-20b",
    "url": "http://localhost:8001",
    "model_type": "gpt-oss",
    "models": ["gpt-oss-20b"]
  }'
```

## API Usage

### Chat Completion

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Always answer in just single word."},
      {"role": "user", "content": "What is the capital of the US?"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7,
    "separate_thinking": true
  }'
```

### Response Structure

The model echoes back everything. The parser separates:
- **developer_instructions**: Your "system" role (converted to developer by model)
- **history**: Conversation messages echoed back
- **analysis**: Reasoning/thinking channel
- **commentary**: Policy/meta commentary channel  
- **final**: The actual answer

```json
{
  "id": "chatcmpl-abc123",
  "model": "gpt-oss-20b",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Washington",
      "thinking": {
        "analysis": "User asks: \"What's capital of the US?\"...",
        "commentary": "The policy states that the assistant must be honest..."
      },
      "parsed": {
        "developer_instructions": "Always answer in just single word.",
        "history": [
          {"role": "user", "content": "What's capital of the US?"}
        ],
        "analysis": "User asks: \"What's capital of the US?\"...",
        "commentary": "The policy states...",
        "final": "Washington"
      }
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 200,
    "total_tokens": 250,
    "thinking_tokens": 180
  }
}
```

## GPT-OSS Token Format

The model uses special tokens:

```
|start|>system<|message|>...<|end|>          # Auto-generated (ignored)
|start|>developer<|message|>...<|end|>       # Your system instructions
|start|>user<|message|>...<|end|>            # User message
|start|>assistant<|channel|>analysis<|message|>...<|end|>    # Reasoning
|start|>assistant<|channel|>commentary<|message|>...<|end|>  # Commentary
|start|>assistant<|channel|>final<|message|>...<|end|>       # Answer
```

## Endpoints

### Model Server (port 8001)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check with queue status |
| `POST /v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `GET /queue/status` | Current queue status |

### Load Balancer (port 8080)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Cluster health |
| `GET /status` | Detailed cluster status |
| `GET /v1/models` | List available models |
| `GET /v1/models/{id}` | Get model info |
| `POST /v1/chat/completions` | Route to backend |
| `POST /backends/register` | Register backend |
| `DELETE /backends/{name}` | Unregister backend |

## Configuration

### Model Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind |
| `--port` | `8000` | Port to bind |
| `--model-path` | `I:/gpt-oss-20b` | Path to model |
| `--model-name` | `gpt-oss-20b` | Model name for API |
| `--device-map` | `auto` | Device mapping |
| `--max-concurrent` | `2` | Max concurrent requests |
