# jira_agent.py

from langchain.agents import AgentExecutor, create_react_agent
# FIX: Import the more general PromptTemplate
from langchain_core.prompts import PromptTemplate

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

# --- 2. Create a ReAct Agent ---

# Define the ReAct prompt using a string template.
template = """
You are a helpful JIRA assistant. Answer the user's questions as best as possible.
You have access to the following tools:

{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

# FIX: Explicitly define all input variables for the prompt template.
# This is the crucial step that ensures the agent is constructed correctly.
prompt = PromptTemplate(
    template=template,
    input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"],
)


# Create the agent runnable.
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor. This is a pre-built graph that runs the ReAct loop.
# It takes the agent's decision, executes the tool, and feeds the result back.
app_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
