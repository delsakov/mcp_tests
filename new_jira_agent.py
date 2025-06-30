# jira_agent.py

from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# FIX: Import the specific formatters and parsers needed for a ReAct agent
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


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

# --- 2. Create a Robust Agent with Manual Construction ---

# FIX: We use a ChatPromptTemplate with MessagesPlaceholder. This is the most
# flexible and modern way to handle chat history and agent thoughts.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful JIRA assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# FIX: We bind the tools to the LLM in a way that works for ANY chat model.
# This adds the tool definitions to the prompt in a standardized way without
# relying on the model having a native `.bind_tools()` method.
llm_with_tools = llm.bind(tools=tools)


# FIX: This is the core of the new agent. It's a chain of runnables (LCEL).
# This explicitly defines how the agent thinks, acts, and parses output.
agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        # This is the key: it correctly formats the agent's intermediate steps
        # (the scratchpad) into a list of messages that the prompt expects.
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser() # This parses the LLM's output to decide if it's a tool call or a final answer.
)


# Create the AgentExecutor.
# We add `handle_parsing_errors=True` for extra stability.
app_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
