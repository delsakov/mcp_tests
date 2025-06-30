# jira_agent.py

from langchain.agents import AgentExecutor, create_react_agent
# We use the standard string-based PromptTemplate for ReAct agents
from langchain_core.prompts import PromptTemplate

# Import your tools and the custom LLM wrapper
import jira_services
# Make sure you are using the simpler GaussLLMWrapper that inherits from LLM
from llm_wrapper import GaussLLMWrapper

# --- 1. Instantiate Tools and LLM ---
tools = [
    jira_services.get_jira_project_schema,
    jira_services.get_my_jira_issues,
    jira_services.create_jira_issue,
]

# This is where you would pass your application settings
mock_settings = {} 
llm = GaussLLMWrapper(settings=mock_settings)

# --- 2. Create the ReAct Agent with the Correct Prompt ---

# This is the standard, battle-tested prompt for ReAct agents.
# It explicitly tells the LLM how to think, act, and format its response
# so the agent executor can parse it.
template = """
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

# Create the prompt template, ensuring all required variables are present.
prompt = PromptTemplate.from_template(template)


# Create the agent runnable. This function is designed to work with the
# LLM wrapper and prompt template defined above.
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor. This is the runtime for the agent.
# It will correctly parse the text output from your LLM.
app_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True # This provides stability
)




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You have access to the following tools:\n\n{tools}\n\nTo use a tool, respond with a JSON blob containing an 'action' and 'action_input' key."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# Create the agent runnable. This function is designed to work with the
# LLM wrapper and prompt template defined above.
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor. This is the runtime for the agent.
# It will correctly parse the text output from your LLM.
app_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True # This provides stability
)

