# jira_agent.py

import re
import json
from typing import List, Dict, Any

# Import your tools and the custom LLM wrapper
import jira_services
from llm_wrapper import GaussLLMWrapper

# --- 1. Instantiate Tools and LLM ---
tools = [
    jira_services.get_jira_project_schema,
    jira_services.get_my_jira_issues,
    jira_services.create_jira_issue,
]
# Create a mapping from tool name to the actual tool function for easy lookup
tool_map = {tool.name: tool for tool in tools}

# This is where you would pass your application settings
mock_settings = {} 
llm = GaussLLMWrapper(settings=mock_settings)

# --- 2. Manually Create the ReAct Prompt ---
# We create a string representation of the tools for the prompt.
tools_string = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
tool_names = ", ".join([tool.name for tool in tools])

# This is the template that instructs the LLM how to behave.
prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

# --- 3. Manual Parser for the LLM's Output ---
def parse_llm_output(text: str) -> Dict[str, Any]:
    """Manually parses the LLM's text output to find an action or final answer."""
    # The model can sometimes surround the JSON with ```json ... ```
    text = re.sub(r"```json\n?([\s\S]*?)\n?```", r"\1", text).strip()

    if "Final Answer:" in text:
        return {"type": "final", "answer": text.split("Final Answer:")[-1].strip()}

    action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        action_input_str = action_match.group(2).strip()
        
        return {"type": "action", "action": action, "action_input": action_input_str}
    
    # If no action or final answer is found, assume it's a thought process
    return {"type": "thought", "details": text}


# --- 4. The Manual Agent Loop ---
async def run_agent_loop(user_input: str, chat_history: str, config: dict):
    """
    Manually runs the ReAct agent loop.
    """
    agent_scratchpad = "Thought:" # Start the thought process
    max_loops = 7 # Safety break to prevent infinite loops

    for i in range(max_loops):
        # Format the full prompt for the LLM
        full_prompt = prompt_template.format(
            tools=tools_string,
            tool_names=tool_names,
            chat_history=chat_history,
            input=user_input,
            agent_scratchpad=agent_scratchpad
        )

        # Call the LLM
        llm_response = await llm._acall(full_prompt, configurable=config)

        # Parse the LLM's response
        parsed_output = parse_llm_output(llm_response)

        if parsed_output["type"] == "final":
            # If we have a final answer, yield it and break the loop
            yield parsed_output["answer"]
            break
        
        if parsed_output["type"] == "action":
            action_name = parsed_output["action"]
            action_input_str = parsed_output["action_input"]

            if action_name in tool_map:
                tool_function = tool_map[action_name]
                
                # Try to parse the input as JSON for tools that expect dicts
                try:
                    action_input_dict = json.loads(action_input_str)
                    observation = tool_function.run(action_input_dict)
                except (json.JSONDecodeError, TypeError):
                    # If it fails, run with the raw string input
                    observation = tool_function.run(action_input_str)

                # FIX: Append the full thought process AND the observation to the scratchpad
                agent_scratchpad += f"{llm_response}\nObservation: {observation}\nThought:"
            else:
                agent_scratchpad += f"{llm_response}\nObservation: Unknown tool '{action_name}'\nThought:"
        else:
            # If the model just "thinks" without acting, append its thought and loop again
            agent_scratchpad += f"{llm_response}\nThought:"
    else:
        yield "Agent reached maximum loops without a final answer."
