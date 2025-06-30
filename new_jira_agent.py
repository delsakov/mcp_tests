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

# FIX: Define the ReAct prompt locally, including the required placeholders.
# The `create_react_agent` function will automatically populate the {tools}
# and {tool_names} variables.
SYSTEM_PROMPT = """
You are a helpful JIRA assistant. Answer the user's questions as best as possible.
You have access to the following tools:

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
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
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
