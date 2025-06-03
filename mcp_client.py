import requests
import json

# Configuration
SERVER_BASE_URL = "http://localhost:8000"  # Change if your server is elsewhere
MCP_HANDSHAKE_PATH = "/mcp"
REQUEST_TIMEOUT = 10  # Seconds for requests to timeout

def get_mcp_manifest_from_server(base_url: str) -> dict | None:
    """
    Performs the MCP handshake to get a session URL via SSE,
    then requests and returns the MCP manifest.
    """
    mcp_handshake_url = f"{base_url}{MCP_HANDSHAKE_PATH}"
    session_path = None

    print(f"Step 1: Connecting to MCP SSE endpoint: {mcp_handshake_url}")
    try:
        # 1. Connect to the /mcp endpoint to initiate SSE and get the session-specific path
        with requests.get(mcp_handshake_url, stream=True, timeout=REQUEST_TIMEOUT) as sse_response:
            sse_response.raise_for_status()  # Check for HTTP errors like 404
            print("   Successfully connected to SSE. Waiting for endpoint data...")

            found_endpoint_event = False
            for line_bytes in sse_response.iter_lines():
                if not line_bytes:  # Skip keep-alive newlines
                    continue
                
                line = line_bytes.decode('utf-8').strip()
                # print(f"   SSE Received: {line}") # Uncomment for verbose SSE logging

                if line.startswith("event:"):
                    if line.split(":", 1)[1].strip() == "endpoint":
                        found_endpoint_event = True
                    else:
                        found_endpoint_event = False # Reset if it's another event type
                elif line.startswith("data:") and found_endpoint_event:
                    session_path = line.split("data:", 1)[1].strip()
                    print(f"   Extracted session path: {session_path}")
                    # We have the path, no need to listen further for this specific task
                    break 
            
            if not session_path:
                print("   Error: Could not extract session path from SSE stream.")
                return None

        # 2. Construct the full session URL for the manifest request
        # The session_path received is typically just the path, not the full URL
        if not session_path.startswith(("http://", "https://")):
            manifest_request_url = f"{base_url}{session_path}"
        else:
            manifest_request_url = session_path # If it was already a full URL
        
        print(f"\nStep 2: Requesting manifest from session URL: {manifest_request_url}")

        # 3. Prepare the GetManifest payload
        manifest_payload = {
            "mcp_protocol_version": "1.0",  # As per MCP spec
            "message_type": "GetManifest"
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json" 
        }

        # 4. Make the POST request to the session URL to get the manifest
        manifest_response = requests.post(
            manifest_request_url, 
            json=manifest_payload, 
            headers=headers, 
            timeout=REQUEST_TIMEOUT
        )
        manifest_response.raise_for_status()  # Check for HTTP errors

        # 5. Parse and return the manifest
        manifest = manifest_response.json()
        print("   Successfully retrieved and parsed manifest.")
        return manifest

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request Exception (e.g., connection error, timeout): {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response from manifest request: {e}")
        if 'manifest_response' in locals() and manifest_response is not None:
             print(f"   Response content that failed to parse: {manifest_response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    return None

if __name__ == "__main__":
    print(f"Attempting to retrieve MCP Manifest from: {SERVER_BASE_URL}\n")
    
    manifest_data = get_mcp_manifest_from_server(SERVER_BASE_URL)
    
    if manifest_data:
        print("\n--- MCP Manifest Retrieved ---")
        print(json.dumps(manifest_data, indent=2))
        
        # You can further process the manifest, e.g., list tool names
        tools = manifest_data.get("tools", [])
        if tools:
            print(f"\nFound {len(tools)} tools in the manifest:")
            for tool in tools:
                print(f"  - Name: {tool.get('name')}")
                print(f"    Description: {tool.get('description', 'N/A')}")
        else:
            print("\nNo tools found in the manifest.")
    else:
        print("\nFailed to retrieve MCP Manifest.")
