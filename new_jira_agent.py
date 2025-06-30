# jira_agent.py

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

# Import your tools and the custom LLM wrapper
import jira_services
from llm_wrapper import InternalThreadedChatModel

# --- 1. Instantiate Tools and LLM ---
tools = [
    jira_services.get_jira_project_schema,
    jira_services.get_my_jira_issues,
    jira_services.create_jira_issue,
]

# This is where you would pass your application settings
mock_settings = {} 
llm = InternalThreadedChatModel(settings=mock_settings)

# --- 2. Create a ReAct Agent (The Fix) ---
# Pull the standard ReAct prompt from the LangChain Hub.
# This prompt is specifically designed to make any chat model use tools
# by instructing it to "think" and output a specific format.
prompt = hub.pull("hwchase17/react-chat")

# Create the agent runnable. This doesn't use `.bind_tools()`.
# Instead, the prompt itself contains the instructions and tool details.
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor. This is a pre-built graph that runs the ReAct loop.
# It takes the agent's decision, executes the tool, and feeds the result back.
# This replaces the manual LangGraph setup for this standard use case.
app_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

