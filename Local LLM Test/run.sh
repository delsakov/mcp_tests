#!/bin/bash
# ============================================================================
# LLM Infrastructure Startup Script
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Configuration
# ============================================================================

# Model Server 1: GPT-OSS-20B
MODEL1_NAME="gpt-oss-20b"
MODEL1_PATH="I:/gpt-oss-20b"
MODEL1_TYPE="gpt-oss"
MODEL1_PORT=8001

# Model Server 2: Gemma3-4B
MODEL2_NAME="gemma-3-4b-it"
MODEL2_PATH="I:/google/gemma-3-4b-it"
MODEL2_TYPE="gemma3"
MODEL2_PORT=8002

# Load Balancer
LB_PORT=8080
LB_STRATEGY="least_connections"

# ============================================================================
# Functions
# ============================================================================

start_model_server() {
    local name=$1
    local path=$2
    local type=$3
    local port=$4
    
    print_status "Starting model server: $name on port $port"
    
    python model_server.py \
        --model-path "$path" \
        --model-type "$type" \
        --model-name "$name" \
        --port "$port" \
        --max-concurrent 2 &
    
    echo $! > "/tmp/${name}.pid"
    print_status "Model server $name started with PID $(cat /tmp/${name}.pid)"
}

start_load_balancer() {
    print_status "Starting load balancer on port $LB_PORT"
    
    python load_balancer.py \
        --port "$LB_PORT" \
        --strategy "$LB_STRATEGY" &
    
    echo $! > "/tmp/load_balancer.pid"
    print_status "Load balancer started with PID $(cat /tmp/load_balancer.pid)"
}

register_backends() {
    print_status "Waiting for servers to start..."
    sleep 10
    
    print_status "Registering backends with load balancer..."
    
    # Register GPT-OSS
    curl -X POST "http://localhost:$LB_PORT/backends/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$MODEL1_NAME\",
            \"url\": \"http://localhost:$MODEL1_PORT\",
            \"model_type\": \"$MODEL1_TYPE\",
            \"models\": [\"$MODEL1_NAME\", \"gpt-oss\"]
        }" 2>/dev/null
    
    # Register Gemma3
    curl -X POST "http://localhost:$LB_PORT/backends/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$MODEL2_NAME\",
            \"url\": \"http://localhost:$MODEL2_PORT\",
            \"model_type\": \"$MODEL2_TYPE\",
            \"models\": [\"$MODEL2_NAME\", \"gemma3\"]
        }" 2>/dev/null
    
    print_status "Backends registered!"
}

stop_all() {
    print_status "Stopping all services..."
    
    for pid_file in /tmp/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                print_status "Stopped process $pid"
            fi
            rm "$pid_file"
        fi
    done
    
    print_status "All services stopped"
}

show_status() {
    echo ""
    echo "============================================"
    echo "          Service Status"
    echo "============================================"
    
    for pid_file in /tmp/*.pid; do
        if [ -f "$pid_file" ]; then
            name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  $name: ${GREEN}Running${NC} (PID: $pid)"
            else
                echo -e "  $name: ${RED}Stopped${NC}"
            fi
        fi
    done
    
    echo ""
    echo "Cluster Status:"
    curl -s "http://localhost:$LB_PORT/status" 2>/dev/null | python -m json.tool 2>/dev/null || echo "  Load balancer not responding"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

case "$1" in
    start)
        print_status "Starting LLM Infrastructure..."
        start_model_server "$MODEL1_NAME" "$MODEL1_PATH" "$MODEL1_TYPE" "$MODEL1_PORT"
        start_model_server "$MODEL2_NAME" "$MODEL2_PATH" "$MODEL2_TYPE" "$MODEL2_PORT"
        start_load_balancer
        register_backends
        show_status
        ;;
    
    start-lb)
        start_load_balancer
        ;;
    
    start-model)
        if [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
            echo "Usage: $0 start-model <name> <path> <type> <port>"
            exit 1
        fi
        start_model_server "$2" "$3" "$4" "$5"
        ;;
    
    stop)
        stop_all
        ;;
    
    status)
        show_status
        ;;
    
    register)
        register_backends
        ;;
    
    *)
        echo "Usage: $0 {start|stop|status|start-lb|start-model|register}"
        echo ""
        echo "Commands:"
        echo "  start        - Start all services (model servers + load balancer)"
        echo "  stop         - Stop all services"
        echo "  status       - Show service status"
        echo "  start-lb     - Start only the load balancer"
        echo "  start-model  - Start a specific model server"
        echo "  register     - Register backends with load balancer"
        exit 1
        ;;
esac
