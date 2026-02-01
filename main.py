# main.py

import asyncio
from typing import Any, Iterator, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# LangChain imports
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Local module imports
from . import orchestration
from . import jira_services


# --- 1. YOUR CUSTOM LLM & SESSION MANAGEMENT ---

# This is a placeholder for your actual streaming LLM class.
# It simulates creating a thread and streaming back a response.
class MyLLM:
    def create_thread(self) -> str:
        import uuid
        new_thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        return new_thread_id

    def get_response(self, thread_id: str, message: str):
        # In a real app, this calls your model's API. Here we simulate it.
        print(f"\n--- LLM SIMULATOR (Thread: {thread_id}) ---")
        print(f"Received prompt: {message}")
        response = f"Simulated response for thread '{thread_id}' about '{message}'"
        for word in response.split():
            yield f"{word} "
            asyncio.sleep(0.05)
        print("--- END LLM SIMULATOR ---\n")

# This is the custom LangChain wrapper for your LLM.
class InternalThreadedChatModel(BaseChatModel):
    llm_instance: MyLLM

    def _get_thread_id(self, configurable: dict) -> str:
        thread_id = configurable.get("thread_id")
        if not thread_id:
            thread_id = self.llm_instance.create_thread()
        return thread_id

    def _stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        configurable = kwargs.get("configurable", {})
        thread_id = self._get_thread_id(configurable)
        user_prompt = messages[-1].content
        stream_iterator = self.llm_instance.get_response(thread_id, str(user_prompt))

        for chunk_str in stream_iterator:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=chunk_str, response_metadata={"thread_id": thread_id}
            ))

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> ChatResult:
        stream_iterator = self._stream(messages, stop, run_manager, **kwargs)
        full_response_content = "".join(chunk.message.content for chunk in stream_iterator)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=full_response_content))])

    @property
    def _llm_type(self) -> str:
        return "internal-threaded-streaming-chat-model"

# WARNING: In-memory store is for demonstration only. Use Redis or a DB in production.
USER_THREADS = {}
def get_user_thread_id(user_id: str) -> Optional[str]:
    return USER_THREADS.get(user_id)
def set_user_thread_id(user_id: str, thread_id: str):
    USER_THREADS[user_id] = thread_id


# --- 2. AGENT AND FASTAPI SETUP ---

app = FastAPI(title="Schema-Aware JIRA Agent")

# Instantiate your custom model and tools
my_llm_instance = MyLLM()
llm = InternalThreadedChatModel(llm_instance=my_llm_instance)
tools = [
    orchestration.get_my_jira_issues,
    orchestration.get_jira_project_schema,
    orchestration.create_jira_issue, # Add the new tool here
]

# The advanced system prompt that instructs the agent on the two-step process
SYSTEM_PROMPT = """You are a comprehensive JIRA assistant, capable of both reading and writing JIRA data.

**Finding Issues:**
When a user asks to find issues with specific criteria (like type, status), you MUST follow this sequence:
1.  First, identify the JIRA project key. If you are unsure, ask the user.
2.  Use the `get_jira_project_schema` tool to fetch the valid filter options for that project.
3.  Examine the schema. Map the user's request (e.g., "open defects") to the official terms from the schema.
4.  Finally, call the `get_my_jira_issues` tool with the correct, schema-validated parameters.

**Creating Issues:**
When a user asks to create a ticket, defect, story, etc., you MUST follow this sequence:
1.  Use the `create_jira_issue` tool. This tool requires a project key, an issue type, a summary, and a description.
2.  Gather ALL of the required information from the user's request.
3.  If any piece of information is missing (e.g., they provide a summary but no description), you MUST ask clarifying questions to get the missing details from the user.
4.  Do not call the `create_jira_issue` tool until you have all four required pieces of information. If you are unsure of the valid issue types for a project, you can use the `get_jira_project_schema` tool first.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


class ChatRequest(BaseModel):
    prompt: str
    user_id: str
    # In a real app, you'd get user_id from a decoded auth token
    
# --- 3. STREAMING API ENDPOINT ---

async def stream_agent_response(user_id: str, prompt_text: str):
    """Async generator to handle the agent streaming and session management."""
    # Set the user ID for the context of this request
    orchestration.get_current_user_id = lambda: user_id
    
    thread_id = get_user_thread_id(user_id)
    config = {"configurable": {"thread_id": thread_id}}

    is_first_chunk = True
    output_streamed = ""
    
    # Use the asynchronous stream method
    async for chunk in agent_executor.astream(
        {"input": prompt_text},
        config=config
    ):
        # The agent executor yields dictionaries for each step.
        # We are interested in the final answer chunks from the LLM.
        if "output" in chunk:
            chunk_content = chunk["output"]
            if chunk_content:
                # The output comes in full every time, so we send the new part
                new_content = chunk_content.replace(output_streamed, "", 1)
                yield new_content
                output_streamed = chunk_content

        # On the first LLM chunk, check if a new thread was created and save it
        if is_first_chunk and "messages" in chunk:
            last_message = chunk["messages"][-1]
            if isinstance(last_message, AIMessage) and not thread_id:
                new_thread_id = last_message.response_metadata.get("thread_id")
                if new_thread_id:
                    set_user_thread_id(user_id, new_thread_id)
                    print(f"✅ Saved new thread '{new_thread_id}' for user '{user_id}'")
                is_first_chunk = False


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """FastAPI endpoint that returns a streaming response from the JIRA agent."""
    return StreamingResponse(
        stream_agent_response(request.user_id, request.prompt),
        media_type="text/plain"
    )

@app.get("/")
def read_root():
    return {"message": "JIRA Agent API is running. POST to /chat/stream to interact."}



WITH ProjectIssueTypes AS (
    -- 1. Get Project and Issue Type Screen Scheme ID
    SELECT 
        p.id AS project_id,
        p.pkey,
        it.id AS issuetype_id,
        it.pname AS issuetype_name,
        na.sinkNodeId AS itss_id -- Issue Type Screen Scheme ID
    FROM project p
    JOIN nodeassociation na ON p.id = na.sourceNodeId 
    JOIN issuetype it ON 1=1 -- We filter valid ITs later via the scheme join
    WHERE na.sinkNodeEntity = 'IssueTypeScreenScheme'
),
EffectiveScreenScheme AS (
    -- 2. Resolve which Screen Scheme applies to which Issue Type
    -- (Checks specific mapping first, falls back to default mapping)
    SELECT 
        pit.project_id,
        pit.pkey,
        pit.issuetype_name,
        -- Logic: If specific mapping exists, use it; otherwise use default (NULL)
        COALESCE(spec.fieldscreenscheme, def.fieldscreenscheme) AS screenscheme_id
    FROM ProjectIssueTypes pit
    -- Try to find specific entry for this issue type
    LEFT JOIN issuetypescreenschemeentity spec 
        ON pit.itss_id = spec.scheme 
        AND spec.issuetype = CAST(pit.issuetype_id AS VARCHAR)
    -- Try to find default entry for this scheme
    LEFT JOIN issuetypescreenschemeentity def 
        ON pit.itss_id = def.scheme 
        AND def.issuetype IS NULL
    WHERE COALESCE(spec.fieldscreenscheme, def.fieldscreenscheme) IS NOT NULL
),
EffectiveScreen AS (
    -- 3. Resolve the Screen ID (Targeting the 'Default' Operation)
    -- This picks the screen used for View/Edit. 
    -- If you strictly want the 'Create' screen, change logic to check operation=2
    SELECT 
        ess.project_id,
        ess.pkey,
        ess.issuetype_name,
        fssi.fieldscreen AS screen_id
    FROM EffectiveScreenScheme ess
    JOIN fieldscreenschemeitem fssi ON ess.screenscheme_id = fssi.fieldscreenscheme
    -- We filter for operation IS NULL to get the "Default" screen (used for View/Edit)
    WHERE fssi.operation IS NULL
)
SELECT 
    es.pkey AS "Project Key",
    es.issuetype_name AS "Issue Type",
    fst.name AS "Tab Name",
    fst.sequence AS "Tab Order",
    fsli.sequence AS "Field Order",
    -- Resolve Field Name
    CASE 
        WHEN cf.cfname IS NOT NULL THEN cf.cfname 
        ELSE fsli.fieldidentifier 
    END AS "Field Name"
FROM EffectiveScreen es
JOIN fieldscreen fs ON es.screen_id = fs.id
JOIN fieldscreentab fst ON fs.id = fst.fieldscreen
JOIN fieldscreenlayoutitem fsli ON fst.id = fsli.fieldscreentab
-- Join Custom Fields to get readable names
LEFT JOIN customfield cf ON 'customfield_' || CAST(cf.id AS VARCHAR) = fsli.fieldidentifier
ORDER BY 
    es.pkey, 
    es.issuetype_name, 
    fst.sequence, 
    fsli.sequence;


WITH ProjectIssueTypes AS (
    -- 1. Get Valid Issue Types per Project
    SELECT 
        p.id AS project_id,
        p.pkey,
        it.id AS issuetype_id,
        it.pname AS issuetype_name
    FROM JIRADC.project p
    JOIN JIRADC.configurationcontext cc ON p.id = cc.project
    JOIN JIRADC.fieldconfigscheme fcs ON cc.fieldconfigscheme = fcs.id
    JOIN JIRADC.fieldconfigschemeissuetype fcsit ON fcs.id = fcsit.fieldconfigscheme
    JOIN JIRADC.optionconfiguration oc ON fcsit.fieldconfiguration = oc.fieldconfig
    JOIN JIRADC.issuetype it ON oc.optionid = it.id
    WHERE fcs.fieldid = 'issuetype'
),
WorkflowMap AS (
    -- 2. Determine which Workflow Name is used
    SELECT 
        pit.project_id,
        pit.pkey,
        pit.issuetype_name,
        -- LOGIC FIX: Coalesce the specific mapping vs the default (NULL) mapping
        COALESCE(wse_specific.workflow, wse_default.workflow) AS workflow_name
    FROM ProjectIssueTypes pit
    -- Link Project to Workflow Scheme
    JOIN JIRADC.nodeassociation na ON pit.project_id = na.sourceNodeId 
        AND na.sinkNodeEntity = 'WorkflowScheme'
    JOIN JIRADC.workflowscheme ws ON na.sinkNodeId = ws.id
    -- Link 1: Try to find a specific workflow for this issue type
    LEFT JOIN JIRADC.workflowschemeentity wse_specific 
        ON ws.id = wse_specific.scheme 
        AND wse_specific.issuetype = CAST(pit.issuetype_id AS VARCHAR2(255))
    -- Link 2: Find the default workflow (where issuetype is NULL)
    LEFT JOIN JIRADC.workflowschemeentity wse_default 
        ON ws.id = wse_default.scheme 
        AND wse_default.issuetype IS NULL
),
ParsedWorkflowStatuses AS (
    -- 3. Parse XML to get Status IDs linked to Steps
    SELECT 
        jw.workflowname,
        xml_step.linkedStatusId
    FROM JIRADC.jiraworkflows jw,
    XMLTABLE(
        '/workflow/steps/step'
        PASSING XMLTYPE(jw.descriptor)
        COLUMNS 
            linkedStatusId VARCHAR2(50) PATH '@linkedStatus'
    ) xml_step
)
SELECT DISTINCT
    wm.pkey AS "Project Key",
    wm.issuetype_name AS "Issue Type",
    wm.workflow_name AS "Workflow Name",
    ist.pname AS "Status Name",
    CASE 
        WHEN ist.statuscategory = 2 THEN 'To Do'
        WHEN ist.statuscategory = 4 THEN 'In Progress'
        WHEN ist.statuscategory = 3 THEN 'Done'
        ELSE 'No Category' 
    END AS "Status Category"
FROM WorkflowMap wm
JOIN ParsedWorkflowStatuses pws ON wm.workflow_name = pws.workflowname
JOIN JIRADC.issuestatus ist ON ist.id = pws.linkedStatusId
ORDER BY 
    wm.pkey, 
    wm.issuetype_name, 
    "Status Category";


    WITH WorkflowMap AS (
    -- 1. Map Project & Issue Type -> Workflow Name (Fixed Logic)
    SELECT 
        p.pkey,
        it.pname AS issuetype_name,
        -- LOGIC FIX: Coalesce specific vs default entity row
        COALESCE(wse_specific.workflow, wse_default.workflow) AS workflow_name
    FROM JIRADC.project p
    JOIN JIRADC.configurationcontext cc ON p.id = cc.project
    JOIN JIRADC.fieldconfigscheme fcs ON cc.fieldconfigscheme = fcs.id
    JOIN JIRADC.fieldconfigschemeissuetype fcsit ON fcs.id = fcsit.fieldconfigscheme
    JOIN JIRADC.optionconfiguration oc ON fcsit.fieldconfiguration = oc.fieldconfig
    JOIN JIRADC.issuetype it ON oc.optionid = it.id
    JOIN JIRADC.nodeassociation na ON p.id = na.sourceNodeId 
        AND na.sinkNodeEntity = 'WorkflowScheme'
    JOIN JIRADC.workflowscheme ws ON na.sinkNodeId = ws.id
    -- Join specific workflow mapping
    LEFT JOIN JIRADC.workflowschemeentity wse_specific 
        ON ws.id = wse_specific.scheme 
        AND wse_specific.issuetype = CAST(it.id AS VARCHAR2(255))
    -- Join default workflow mapping
    LEFT JOIN JIRADC.workflowschemeentity wse_default 
        ON ws.id = wse_default.scheme 
        AND wse_default.issuetype IS NULL
    WHERE fcs.fieldid = 'issuetype'
),
RawWorkflowSteps AS (
    -- 2. Create Map: Step ID -> Status ID (To resolve destinations)
    SELECT 
        jw.workflowname,
        xml_step.stepId,
        xml_step.linkedStatusId
    FROM JIRADC.jiraworkflows jw,
    XMLTABLE(
        '/workflow/steps/step'
        PASSING XMLTYPE(jw.descriptor)
        COLUMNS 
            stepId VARCHAR2(50) PATH '@id',
            linkedStatusId VARCHAR2(50) PATH '@linkedStatus'
    ) xml_step
    WHERE jw.workflowname IN (SELECT DISTINCT workflow_name FROM WorkflowMap)
),
RegularTransitions AS (
    -- 3a. Extract Regular Transitions (Step -> Step)
    SELECT 
        jw.workflowname,
        xml_step.linkedStatusId AS source_status_id,
        xml_action.transition_name,
        xml_action.target_step_id,
        'Regular' AS transition_type
    FROM JIRADC.jiraworkflows jw,
    XMLTABLE(
        '/workflow/steps/step'
        PASSING XMLTYPE(jw.descriptor)
        COLUMNS 
            linkedStatusId VARCHAR2(50) PATH '@linkedStatus',
            actionsXml XMLTYPE PATH 'actions'
    ) xml_step,
    XMLTABLE(
        '/actions/action'
        PASSING xml_step.actionsXml
        COLUMNS 
            transition_name VARCHAR2(100) PATH '@name',
            target_step_id VARCHAR2(50) PATH 'results/unconditional-result/@step'
    ) xml_action
    WHERE jw.workflowname IN (SELECT DISTINCT workflow_name FROM WorkflowMap)
),
GlobalTransitions AS (
    -- 3b. Extract Global Transitions (Any -> Step)
    SELECT 
        jw.workflowname,
        NULL AS source_status_id,
        xml_action.transition_name,
        xml_action.target_step_id,
        'Global' AS transition_type
    FROM JIRADC.jiraworkflows jw,
    XMLTABLE(
        '/workflow/global-actions/action'
        PASSING XMLTYPE(jw.descriptor)
        COLUMNS 
            transition_name VARCHAR2(100) PATH '@name',
            target_step_id VARCHAR2(50) PATH 'results/unconditional-result/@step'
    ) xml_action
    WHERE jw.workflowname IN (SELECT DISTINCT workflow_name FROM WorkflowMap)
),
AllTransitions AS (
    SELECT * FROM RegularTransitions
    UNION ALL
    SELECT * FROM GlobalTransitions
)
-- 4. Final Join
SELECT 
    wm.pkey AS "Project Key",
    wm.issuetype_name AS "Issue Type",
    wm.workflow_name AS "Workflow Name",
    
    -- Source Status
    COALESCE(stat_source.pname, '(Any Status)') AS "From Status",
    
    -- Transition
    at.transition_name AS "Transition Name",
    at.transition_type AS "Type",
    
    -- Destination Status
    stat_dest.pname AS "To Status"

FROM WorkflowMap wm
JOIN AllTransitions at ON wm.workflow_name = at.workflowname
-- Join Source Status
LEFT JOIN JIRADC.issuestatus stat_source ON at.source_status_id = stat_source.id
-- Join Destination Status (Lookup Target Step -> Target Status)
LEFT JOIN RawWorkflowSteps rws_dest 
    ON at.workflowname = rws_dest.workflowname 
    AND at.target_step_id = rws_dest.stepId
LEFT JOIN JIRADC.issuestatus stat_dest 
    ON rws_dest.linkedStatusId = stat_dest.id

ORDER BY 
    wm.pkey, 
    wm.issuetype_name, 
    "From Status" NULLS FIRST, 
    "Transition Name";



SELECT DISTINCT
    p.pkey AS "Project Key",
    p.pname AS "Project Name",
    pr.pname AS "Priority Name",
    pr.description AS "Description",
    pr.sequence AS "Order Sequence",
    pr.id AS "Priority ID"
FROM JIRADC.project p
-- 1. Link Project to Configuration Context
JOIN JIRADC.configurationcontext cc ON p.id = cc.project
-- 2. Link Context to the Priority Scheme
JOIN JIRADC.fieldconfigscheme fcs ON cc.fieldconfigscheme = fcs.id
-- 3. Link Scheme to the Configuration Mapping
JOIN JIRADC.fieldconfigschemeissuetype fcsit ON fcs.id = fcsit.fieldconfigscheme
-- 4. Link Configuration to Selected Options (Priorities)
JOIN JIRADC.optionconfiguration oc ON fcsit.fieldconfiguration = oc.fieldconfig
-- 5. Link Options to the actual Priority table
JOIN JIRADC.priority pr ON oc.optionid = pr.id
WHERE 
    fcs.fieldid = 'priority' -- Vital filter to exclude Issue Type schemes
ORDER BY 
    p.pkey, 
    pr.sequence;

WITH ProjectPrioritySchemes AS (
    -- Get Projects that have a specific scheme assigned
    SELECT 
        p.id AS project_id,
        oc.optionid AS priority_id
    FROM JIRADC.project p
    JOIN JIRADC.configurationcontext cc ON p.id = cc.project
    JOIN JIRADC.fieldconfigscheme fcs ON cc.fieldconfigscheme = fcs.id
    JOIN JIRADC.fieldconfigschemeissuetype fcsit ON fcs.id = fcsit.fieldconfigscheme
    JOIN JIRADC.optionconfiguration oc ON fcsit.fieldconfiguration = oc.fieldconfig
    WHERE fcs.fieldid = 'priority'
),
GlobalPriorities AS (
    -- Get all priorities for the fallback
    SELECT id FROM JIRADC.priority
)
SELECT 
    p.pkey AS "Project Key",
    pr.pname AS "Priority Name",
    pr.sequence AS "Sequence"
FROM JIRADC.project p
-- Join to the specific scheme map
LEFT JOIN ProjectPrioritySchemes pps ON p.id = pps.project_id
-- Join to the priority table
JOIN JIRADC.priority pr 
    ON pr.id = COALESCE(pps.priority_id, pr.id) -- Logic: Use specific ID if exists, otherwise join on itself (all)
WHERE 
    -- If specific scheme exists, only show those matches
    (pps.project_id IS NOT NULL AND pr.id = pps.priority_id)
    OR 
    -- If NO specific scheme exists, show ALL priorities (Global Default)
    (pps.project_id IS NULL)
ORDER BY 
    p.pkey, 
    pr.sequence;
