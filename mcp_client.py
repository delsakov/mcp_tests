import requests
import json
import threading
import time

# Configuration
SERVER_BASE_URL = "http://localhost:8000"  # Change if your server is elsewhere
MCP_HANDSHAKE_PATH = "/mcp"
REQUEST_TIMEOUT = 10  # Seconds for HTTP requests
SSE_LISTEN_TIMEOUT = 20 # Seconds to wait for specific messages on SSE

# Shared data lists to communicate between threads
# Using lists because they are mutable and can be updated by the thread
session_path_holder = [] 
manifest_holder = []

def sse_event_listener(sse_response_stream):
    """
    Continuously listens to the SSE stream and processes messages.
    Populates session_path_holder and manifest_holder when relevant messages are found.
    """
    print("[SSE_Listener] Started.")
    is_expecting_session_path_data = False
    try:
        for line_bytes in sse_response_stream.iter_lines():
            if not line_bytes:  # Skip keep-alive newlines or message separators
                is_expecting_session_path_data = False # Reset after an empty line if needed
                continue
            
            line = line_bytes.decode('utf-8').strip()
            # print(f"[SSE_Listener] Line: {line}") # Uncomment for verbose SSE logging

            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
                if event_name == "endpoint":
                    is_expecting_session_path_data = True
                # else:
                    # print(f"[SSE_Listener] Event: {event_name}")
                continue # Next line should be data for this event or another field

            if line.startswith("data:"):
                data_content = line.split("data:", 1)[1].strip()
                if is_expecting_session_path_data and not session_path_holder:
                    print(f"[SSE_Listener] Received session path: {data_content}")
                    session_path_holder.append(data_content)
                    is_expecting_session_path_data = False # Consumed this specific data expectation
                else:
                    # Try to parse any other data as JSON (could be the manifest)
                    try:
                        json_data = json.loads(data_content)
                        if isinstance(json_data, dict) and json_data.get("message_type") == "Manifest":
                            print("[SSE_Listener] Manifest received via SSE!")
                            manifest_holder.append(json_data)
                            break # Got the manifest, can stop listening for this script's purpose
                        # else:
                            # print(f"[SSE_Listener] Other JSON data: {json_data.get('message_type')}")
                    except json.JSONDecodeError:
                        # print(f"[SSE_Listener] Data is not JSON or not the expected message: {data_content}")
                        pass # Ignore if it's not the manifest we're looking for right now
        
    except requests.exceptions.ChunkedEncodingError:
        print("[SSE_Listener] SSE stream connection closed by server (may be expected).")
    except Exception as e:
        print(f"[SSE_Listener] Error: {e}")
    finally:
        print("[SSE_Listener] Finished.")
        # If the loop finishes (e.g. `break` or stream ends) and holders are empty, it indicates timeout/failure for those items.
        # The main thread handles timeouts for waiting on these holders.


def get_mcp_manifest_v3():
    mcp_handshake_url = f"{SERVER_BASE_URL}{MCP_HANDSHAKE_PATH}"
    
    # Clear holders for a fresh run
    session_path_holder.clear()
    manifest_holder.clear()

    listener_thread = None
    sse_response = None

    try:
        print(f"Step 1: Connecting to MCP SSE endpoint: {mcp_handshake_url}")
        # Note: This 'sse_response' must be kept open and passed to the listener.
        # The 'with' statement here is tricky if the listener thread uses it long-term.
        # We will manage its closure manually.
        sse_response = requests.get(mcp_handshake_url, stream=True, timeout=REQUEST_TIMEOUT)
        sse_response.raise_for_status()
        print("   Successfully connected to SSE. Starting listener thread.")

        listener_thread = threading.Thread(target=sse_event_listener, args=(sse_response,))
        listener_thread.daemon = True # So main thread can exit if this hangs (though we join)
        listener_thread.start()

        # Wait for the session_path from the listener thread
        print("   Waiting for session path from SSE listener...")
        start_time = time.time()
        while not session_path_holder:
            if not listener_thread.is_alive():
                raise Exception("SSE listener thread died unexpectedly while waiting for session path.")
            if time.time() - start_time > SSE_LISTEN_TIMEOUT:
                raise TimeoutError("Timeout waiting for session path from SSE listener.")
            time.sleep(0.1)
        
        session_path = session_path_holder[0]

        # Construct manifest request URL
        if not session_path.startswith(("http://", "https://")):
            manifest_request_url = f"{SERVER_BASE_URL}{session_path}"
        else:
            manifest_request_url = session_path
        
        print(f"\nStep 2: POSTing GetManifest to session URL: {manifest_request_url}")
        manifest_payload = {
            "mcp_protocol_version": "1.0",
            "message_type": "GetManifest"
        }
        custom_headers = {"Accept": "application/json"} # Content-Type is auto for json=

        post_response = requests.post(
            manifest_request_url,
            json=manifest_payload,
            headers=custom_headers,
            timeout=REQUEST_TIMEOUT
        )
        print(f"   POST Response Status: {post_response.status_code}")
        print(f"   POST Response Body: {post_response.text}") # Should be {"message": "Accepted"}

        if not (200 <= post_response.status_code < 300): # Check for 2xx success codes
            raise Exception(f"POST request failed: {post_response.status_code} - {post_response.text}")

        # Wait for the manifest to be received by the listener thread
        print("\nStep 3: Waiting for Manifest data on the SSE stream...")
        start_time = time.time()
        while not manifest_holder:
            if not listener_thread.is_alive():
                # If thread finished and manifest_holder is still empty, it means manifest wasn't found by listener
                if not manifest_holder: # Double check
                    print("   SSE listener thread finished without finding the manifest.")
                    return None # Or raise specific error
                # else: it was found just as thread exited, proceed.
            if time.time() - start_time > SSE_LISTEN_TIMEOUT:
                raise TimeoutError("Timeout waiting for manifest from SSE listener.")
            time.sleep(0.1)

        if manifest_holder:
            print("   Manifest successfully retrieved via SSE.")
            return manifest_holder[0]
        else:
            # This case should ideally be caught by listener_thread.is_alive() check above if it exits early.
            print("   Failed to retrieve manifest (manifest_holder is empty).")
            return None

    except TimeoutError as e:
        print(f"Operation Timed Out: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        print("Cleaning up...")
        if sse_response:
            print("   Closing SSE response stream.")
            sse_response.close() # Ensure the initial SSE connection is closed
        if listener_thread and listener_thread.is_alive():
            print("   Waiting for SSE listener thread to join...")
            listener_thread.join(timeout=5) # Wait for the thread to finish
            if listener_thread.is_alive():
                print("   Warning: SSE listener thread did not exit cleanly.")
        print("Cleanup finished.")


if __name__ == "__main__":
    print("Attempting to retrieve MCP Manifest (v3 - persistent SSE)...\n")
    manifest = get_mcp_manifest_v3()
    if manifest:
        print("\n--- MCP Manifest Retrieved ---")
        print(json.dumps(manifest, indent=2))
    else:
        print("\n--- Failed to retrieve MCP Manifest ---")



def execute_sandboxed_query(db_engine, sql_query: str, allowed_schema: str):
    """
    Executes a SQL query within a secure, sandboxed transaction.

    Args:
        db_engine: The SQLAlchemy engine instance.
        sql_query: The raw SQL query string from the user.
        allowed_schema: The schema associated with the user's API key.

    Returns:
        A tuple: (success: bool, result: list | str).
        On success, result is a list of dictionaries (the query rows).
        On failure, result is an error message string.
    """
    # 1. Security Check: Validate the schema against a whitelist.
    if allowed_schema not in ALLOWED_SCHEMAS_FOR_EXECUTION:
        return (False, f"Error: Schema '{allowed_schema}' is not authorized for execution.")

    # 2. Construct the role name from the validated schema.
    role_to_set = f"{allowed_schema}_executor_role"

    # The 'with engine.connect() as conn' block handles connection pooling and closing.
    with db_engine.connect() as conn:
        # The 'with conn.begin() as trans' block handles the transaction.
        # It automatically COMMITS on success or ROLLS BACK on any exception.
        try:
            with conn.begin() as trans:
                # 3. Set the Role for this transaction. The f-string is safe here
                #    because `role_to_set` was built from a whitelisted schema name.
                conn.execute(text(f"SET ROLE '{role_to_set}'"))

                # 4. Execute the User's Query.
                #    PostgreSQL itself will now enforce the permissions of `tests_executor_role`.
                result_proxy = conn.execute(text(sql_query))

                # If the query was a SELECT, fetch the results.
                if result_proxy.returns_rows:
                    # .mappings() provides a dict-like interface for each row.
                    # .all() fetches all rows into a list.
                    results_list = result_proxy.mappings().all()
                else:
                    results_list = [] # For non-SELECT statements like INSERT/UPDATE

                # 5. Reset the Role (optional but good practice)
                #    The role is reset automatically at the end of the transaction,
                #    but explicit reset is clearer.
                conn.execute(text("RESET ROLE"))

                # The transaction will be committed automatically upon exiting this block.
                return (True, results_list)

        except ProgrammingError as e:
            # This block will catch permission errors from Postgres, syntax errors, etc.
            # The transaction has already been rolled back by the 'with' statement.
            # We return the original error from Postgres for debugging.
            return (False, f"Execution failed: {e.orig}")

        except Exception as e:
            # Catch any other unexpected errors
            return (False, f"An unexpected error occurred: {e}")
