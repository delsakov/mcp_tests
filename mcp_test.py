import requests
import json

# --- Configuration ---
BASE_URL = "http://localhost:8000"  # Change to your server's address
MCP_ENDPOINT = f"{BASE_URL}/mcp"

# --- Headers for Streaming Request ---
stream_headers = {
    'Authorization': f'Bearer {BEARER_TOKEN}',
    'Accept': 'text/event-stream'
}

session = requests.Session()

try:
    # 1. Connect and get operations using a POST request
    print(f"--> Connecting to {MCP_ENDPOINT} with POST...")
    initial_payload = {
        "mcp_operation": "get_all_operations"
    }

    # Use POST instead of GET and include the initial payload
    initial_response = session.post(
        MCP_ENDPOINT,
        headers=stream_headers,
        json=initial_payload,
        stream=True  # Keep stream=True to handle the response
    )
    initial_response.raise_for_status() # This should now pass

    print("\n✅ Connection successful. Server Operations:")
    for line in initial_response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data:'):
                data_str = line_str[len('data:'):].strip()
                if data_str:
                    data = json.loads(data_str)
                    print(json.dumps(data, indent=2))

    # 2. Subsequent requests remain the same
    print(f"\n--> Requesting list of tools from {MCP_ENDPOINT}...")
    tool_request_payload = {
        "mcp_operation": "get_all_tools"
    }

    # For a non-streaming POST, standard JSON accept is fine
    post_headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}',
        'Accept': 'application/json'
    }
    tool_response = session.post(MCP_ENDPOINT, headers=post_headers, json=tool_request_payload)
    tool_response.raise_for_status()

    print("\n✅ Successfully retrieved tools:")
    print(json.dumps(tool_response.json(), indent=2))

except requests.exceptions.RequestException as e:
    print(f"\n❌ An error occurred: {e}")

finally:
    session.close()
    print("\nSession closed.")
