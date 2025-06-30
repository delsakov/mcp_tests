# jira_agent.py

from langchain.agents import AgentExecutor, create_react_agent
# FIX: Import ChatPromptTemplate and MessagesPlaceholder to build the prompt locally
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

# FIX: Define the ReAct prompt locally instead of using hub.pull()
# This removes the external network dependency.
# This structure is the standard for ReAct agents.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the tools provided to answer user questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# Create the agent runnable.
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor. This is a pre-built graph that runs the ReAct loop.
# It takes the agent's decision, executes the tool, and feeds the result back.
app_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
