import requests
import json
import threading
import time

# --- Client Configuration ---
SERVER_BASE_URL = "http://localhost:8000"  # Change if your server is elsewhere
MCP_HANDSHAKE_PATH = "/mcp" # Or your specific MCP prefix
REQUEST_TIMEOUT = 10  # Seconds for HTTP requests
SSE_LISTEN_TIMEOUT = 20 # Seconds to wait for specific messages on SSE

# ** API Key Configuration for MCP Client **
MCP_CLIENT_API_KEY = "your_mcp_client_api_key_here"  # <--- REPLACE THIS!
MCP_API_KEY_HEADER_NAME = "X-MCP-API-Key" # Must match server configuration

# Shared data lists to communicate between threads
session_path_holder = [] 
manifest_holder = []

def sse_event_listener_simplified(sse_response_stream):
    """
    Continuously listens to the SSE stream.
    Extracts the session path upon handshake.
    Looks for a Manifest message sent by the server.
    """
    print("[SSE_Listener] Started.")
    is_expecting_session_path_data = False
    try:
        for line_bytes in sse_response_stream.iter_lines():
            if not line_bytes:
                is_expecting_session_path_data = False # Reset after an empty line
                continue
            
            line = line_bytes.decode('utf-8').strip()
            # print(f"[SSE_Listener] Line: {line}") # Uncomment for verbose SSE logging

            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
                if event_name == "endpoint":
                    is_expecting_session_path_data = True # Expect data line for session path
                # else:
                    # print(f"[SSE_Listener] Event: {event_name}") # e.g., could be 'mcp'
                continue

            if line.startswith("data:"):
                data_content = line.split("data:", 1)[1].strip()
                if is_expecting_session_path_data and not session_path_holder:
                    print(f"[SSE_Listener] Received session path: {data_content}")
                    session_path_holder.append(data_content)
                    is_expecting_session_path_data = False # Session path obtained
                    # Continue listening, server might send manifest next
                else:
                    # Try to parse any other data as JSON (could be the manifest)
                    try:
                        json_data = json.loads(data_content)
                        if isinstance(json_data, dict) and json_data.get("message_type") == "Manifest":
                            print("[SSE_Listener] Manifest received via SSE!")
                            manifest_holder.append(json_data)
                            break # Got the manifest, exit listener
                        # else:
                            # print(f"[SSE_Listener] Other JSON data: {json_data.get('message_type')}")
                    except json.JSONDecodeError:
                        # print(f"[SSE_Listener] Data is not JSON or not the expected message: {data_content}")
                        pass 
        
    except requests.exceptions.ChunkedEncodingError:
        print("[SSE_Listener] SSE stream connection closed by server (may be expected after manifest or timeout).")
    except Exception as e:
        print(f"[SSE_Listener] Error: {e}")
    finally:
        print("[SSE_Listener] Finished.")


def get_mcp_manifest_simplified_sse_only(api_key_to_send: str):
    mcp_handshake_url = f"{SERVER_BASE_URL}{MCP_HANDSHAKE_PATH}"
    
    session_path_holder.clear()
    manifest_holder.clear()

    listener_thread = None
    sse_response = None

    client_auth_headers = {
        MCP_API_KEY_HEADER_NAME: api_key_to_send,
        "Accept": "text/event-stream"
    }

    try:
        print(f"Step 1: Connecting to MCP SSE endpoint: {mcp_handshake_url}")
        print(f"   With Headers: {client_auth_headers}")
        
        sse_response = requests.get(
            mcp_handshake_url, 
            stream=True, 
            timeout=REQUEST_TIMEOUT,
            headers=client_auth_headers
        )
        sse_response.raise_for_status()
        print("   Successfully connected to SSE. Starting listener thread to get session path AND manifest.")

        listener_thread = threading.Thread(target=sse_event_listener_simplified, args=(sse_response,))
        listener_thread.daemon = True
        listener_thread.start()

        # Wait for the session_path (as confirmation of successful connection and handshake)
        # AND the manifest itself. The listener will populate both.
        print("   Waiting for session path and manifest from SSE listener...")
        start_time = time.time()
        while (not session_path_holder or not manifest_holder): # Wait for both
            if not listener_thread.is_alive():
                if not manifest_holder: # Check if manifest was found just before thread exit
                     print("   SSE listener thread died unexpectedly before retrieving manifest.")
                     return None # Or raise specific error
                break # Manifest was found, or listener exited
            if time.time() - start_time > SSE_LISTEN_TIMEOUT: # Overall timeout for both items
                if not session_path_holder:
                    raise TimeoutError("Timeout waiting for session path from SSE listener.")
                if not manifest_holder:
                    raise TimeoutError("Timeout waiting for manifest from SSE listener (session path was received).")
                break # Should not happen if one of the above raised
            time.sleep(0.1)
        
        if session_path_holder:
            print(f"   Session path confirmed: {session_path_holder[0]}")
        else:
            print("   Warning: Session path was not confirmed (listener might have exited early).")


        if manifest_holder:
            print("   Manifest successfully retrieved via SSE (no POST was needed).")
            return manifest_holder[0]
        else:
            print("   Failed to retrieve manifest via SSE. Server might not have sent it, or listener timed out/exited.")
            return None

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error during initial SSE connection: {e}")
        if e.response is not None:
            print(f"   Response status: {e.response.status_code}")
            print(f"   Response text: {e.response.text}")
        return None
    except TimeoutError as e:
        print(f"Operation Timed Out: {e}")
        return None
    except requests.exceptions.RequestException as e: # Other connection errors
        print(f"Request Exception: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        print("Cleaning up...")
        if sse_response: # Ensure this is the response from requests.get
            print("   Closing SSE response stream.")
            sse_response.close()
        if listener_thread and listener_thread.is_alive():
            print("   Waiting for SSE listener thread to join...")
            listener_thread.join(timeout=5)
            if listener_thread.is_alive():
                print("   Warning: SSE listener thread did not exit cleanly.")
        print("Cleanup finished.")


if __name__ == "__main__":
    print("Attempting to retrieve MCP Manifest (Simplified SSE-only client with Auth)...\n")
    
    if not MCP_CLIENT_API_KEY or MCP_CLIENT_API_KEY == "your_mcp_client_api_key_here":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please update MCP_CLIENT_API_KEY in the script with a valid key !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        manifest = get_mcp_manifest_simplified_sse_only(MCP_CLIENT_API_KEY)
        if manifest:
            print("\n--- MCP Manifest Retrieved ---")
            print(json.dumps(manifest, indent=2))
        else:
            print("\n--- Failed to retrieve MCP Manifest ---")
