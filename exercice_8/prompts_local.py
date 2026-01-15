"""
Prompts for Local RAG + DuckDB Multi-Agent System
Modified from L6 to support local agents
"""
from typing import Dict, Any, List, Optional
from langchain.schema import HumanMessage
import json
from langgraph.graph import MessagesState

MAX_REPLANS = 2

# ============================================================================
# STATE DEFINITION (for reference)
# ============================================================================

class State(MessagesState):
    enabled_agents: Optional[List[str]]
    plan: Optional[Dict[str, Dict[str, Any]]]
    user_query: Optional[str]
    current_step: int
    replan_flag: Optional[bool]
    last_reason: Optional[str]
    replan_attempts: Optional[Dict[int, int]]
    agent_query: Optional[str]

# ============================================================================
# AGENT DESCRIPTIONS
# ============================================================================

def get_agent_descriptions() -> Dict[str, Dict[str, Any]]:
    """
    Return structured agent descriptions for local setup
    """
    return {
        "local_rag_researcher": {
            "name": "Local RAG Researcher",
            "capability": "Search local document database (PDF files) using semantic search",
            "use_when": "Need information from internal documents, research papers, or knowledge base",
            "limitations": "Cannot access external web data or structured databases",
            "output_format": "Document excerpts with sources and page numbers",
        },
        "duckdb_researcher": {
            "name": "DuckDB Researcher",
            "capability": "Query structured sales data in local DuckDB database (deals, revenue, companies)",
            "use_when": "Need structured data queries about sales metrics, deal status, revenue analysis",
            "limitations": "Only has access to sales data schema (deals table)",
            "output_format": "SQL query results with formatted tables",
        },
        "chart_generator": {
            "name": "Chart Generator",
            "capability": "Build visualizations from structured data using Python",
            "use_when": "User explicitly requests charts, graphs, plots, or visualizations",
            "limitations": "Requires structured data from previous steps",
            "output_format": "Visual charts saved to file",
            "position_requirement": "Must be used as final step after data gathering",
        },
        "chart_summarizer": {
            "name": "Chart Summarizer",
            "capability": "Summarize and explain chart visualizations",
            "use_when": "After chart_generator has created a visualization",
            "limitations": "Requires a chart as input",
            "output_format": "Written summary (max 3 sentences)",
        },
        "synthesizer": {
            "name": "Synthesizer",
            "capability": "Write comprehensive prose summaries combining all findings",
            "use_when": "Final step when no visualization is requested",
            "limitations": "Requires research data from previous steps",
            "output_format": "Coherent written summary",
            "position_requirement": "Should be used as final step when no chart is needed",
        },
    }

def _get_enabled_agents(state: State | None = None) -> List[str]:
    """Return enabled agents, defaulting to baseline if not specified"""
    baseline = ["local_rag_researcher", "duckdb_researcher", "chart_generator", 
                "chart_summarizer", "synthesizer"]
    
    if not state:
        return baseline
    
    val = state.get("enabled_agents") if hasattr(state, "get") else getattr(state, "enabled_agents", None)
    
    if isinstance(val, list) and val:
        allowed = {"local_rag_researcher", "duckdb_researcher", "chart_generator",
                   "chart_summarizer", "synthesizer"}
        filtered = [a for a in val if a in allowed]
        return filtered
    
    return baseline

def format_agent_list_for_planning(state: State | None = None) -> str:
    """Format agent descriptions for planning prompt"""
    descriptions = get_agent_descriptions()
    enabled_list = _get_enabled_agents(state)
    agent_list = []
    
    for agent_key, details in descriptions.items():
        if agent_key not in enabled_list:
            continue
        agent_list.append(f"  • `{agent_key}` – {details['capability']}")
    
    return "\n".join(agent_list)

def format_agent_guidelines_for_planning(state: State | None = None) -> str:
    """Format agent usage guidelines for planning prompt"""
    descriptions = get_agent_descriptions()
    enabled = set(_get_enabled_agents(state))
    guidelines = []
    
    # RAG vs DuckDB guidance
    if "local_rag_researcher" in enabled:
        guidelines.append(
            f"- Use `local_rag_researcher` when {descriptions['local_rag_researcher']['use_when'].lower()}."
        )
    if "duckdb_researcher" in enabled:
        guidelines.append(
            f"- Use `duckdb_researcher` for {descriptions['duckdb_researcher']['use_when'].lower()}."
        )
    
    # Chart generator rules
    if "chart_generator" in enabled:
        chart_desc = descriptions['chart_generator']
        cs_hint = " A `chart_summarizer` should follow." if "chart_summarizer" in enabled else ""
        guidelines.append(
            f"- **Include `chart_generator` _only_ if {chart_desc['use_when'].lower()}**. "
            f"Must be {chart_desc['position_requirement'].lower()}.{cs_hint}"
        )
    
    # Synthesizer default
    if "synthesizer" in enabled:
        synth_desc = descriptions['synthesizer']
        guidelines.append(
            f"- Otherwise use `synthesizer` as {synth_desc['position_requirement'].lower()}, "
            "including all data from previous steps."
        )
    
    return "\n".join(guidelines)

def format_agent_guidelines_for_executor(state: State | None = None) -> str:
    """Format agent usage guidelines for executor prompt"""
    descriptions = get_agent_descriptions()
    enabled = _get_enabled_agents(state)
    guidelines = []
    
    if "local_rag_researcher" in enabled:
        guidelines.append(
            f"- Use `\"local_rag_researcher\"` when {descriptions['local_rag_researcher']['use_when'].lower()}."
        )
    if "duckdb_researcher" in enabled:
        guidelines.append(
            f"- Use `\"duckdb_researcher\"` for {descriptions['duckdb_researcher']['use_when'].lower()}."
        )
    
    return "\n".join(guidelines)

# ============================================================================
# PLANNER PROMPT
# ============================================================================

def plan_prompt(state: State) -> HumanMessage:
    """Build the prompt for the planner LLM"""
    
    replan_flag = state.get("replan_flag", False)
    user_query = state.get("user_query", state["messages"][0].content)
    prior_plan = state.get("plan") or {}
    replan_reason = state.get("last_reason", "")
    
    agent_list = format_agent_list_for_planning(state)
    agent_guidelines = format_agent_guidelines_for_planning(state)
    
    enabled_list = _get_enabled_agents(state)
    enabled_for_planner = [
        a for a in enabled_list
        if a in ("local_rag_researcher", "duckdb_researcher", "chart_generator", "synthesizer")
    ]
    planner_agent_enum = " | ".join(enabled_for_planner) or "local_rag_researcher | duckdb_researcher | synthesizer"
    
    prompt = f"""
You are the **Planner** in a multi-agent system. Break the user's request
into a sequence of numbered steps (1, 2, 3, ...). **There is no hard limit on
step count** as long as the plan is concise and each step has a clear goal.

Break complex queries into the smallest possible sub-queries, where each
sub-query can be answered by a single data source.

Available agents:

{agent_list}

Return **ONLY** valid JSON (no markdown, no explanations) in this format:

{{
  "1": {{
    "agent": "{planner_agent_enum}",
    "action": "string",
    "pre_conditions": ["string", ...],
    "post_conditions": ["string", ...],
    "goal": "string"
  }},
  "2": {{ ... }},
  "3": {{ ... }}
}}

Guidelines:
{agent_guidelines}

**Important for structured queries:**
- For questions about sales data, revenue, deals → use `duckdb_researcher`
- For questions from documents, papers, knowledge base → use `local_rag_researcher`
- For mixed queries, separate into distinct steps for each data source
"""
    
    if replan_flag:
        prompt += f"""

The current plan needs revision because: {replan_reason}

Current plan:
{json.dumps(prior_plan, indent=2)}

When replanning:
- Focus on UNBLOCKING the workflow
- Only modify steps preventing progress
- Prefer simpler alternatives
"""
    else:
        prompt += "\n\nGenerate a new plan from scratch."
    
    prompt += f'\n\nUser query: "{user_query}"'
    
    return HumanMessage(content=prompt)

# ============================================================================
# EXECUTOR PROMPT
# ============================================================================

def executor_prompt(state: State) -> HumanMessage:
    """Build the prompt for the executor LLM"""
    
    step = int(state.get("current_step", 0))
    latest_plan: Dict[str, Any] = state.get("plan") or {}
    plan_block: Dict[str, Any] = latest_plan.get(str(step), {})
    max_replans = MAX_REPLANS
    attempts = (state.get("replan_attempts", {}) or {}).get(step, 0)
    
    executor_guidelines = format_agent_guidelines_for_executor(state)
    plan_agent = plan_block.get("agent", "local_rag_researcher")
    
    messages_tail = (state.get("messages") or [])[-4:]
    
    enabled_agents = [
        a for a in _get_enabled_agents(state)
        if a in ['local_rag_researcher', 'duckdb_researcher', 'chart_generator',
                 'chart_summarizer', 'synthesizer']
    ] + ['planner']
    
    agent_enum = '|'.join(sorted(set(enabled_agents)))
    
    prompt = f"""
You are the **executor** in a multi-agent system with these agents:
`{', '.join(sorted(set(enabled_agents)))}`

**Tasks**
1. Decide if the current plan needs revision  → `"replan": true|false`
2. Decide which agent to run next            → `"goto": "<agent_name>"`
3. Give one-sentence justification          → `"reason": "<text>"`
4. Write the exact question for the agent   → `"query": "<text>"`

**Guidelines**
{executor_guidelines}
- After **{MAX_REPLANS}** failed replans, move forward
- If just replanned, let the assigned agent try once before reconsidering

Respond **only** with valid JSON (no additional text):

{{
  "replan": <true|false>,
  "goto": "<{agent_enum}>",
  "reason": "<1 sentence>",
  "query": "<text>"
}}

**PRIORITIZE FORWARD PROGRESS:**
Only replan if completely blocked. Set `"replan": true` **only if**:
• The step produced zero useful information
• Missing information cannot be obtained by remaining steps
• `attempts < {max_replans}`

When `attempts == {max_replans}`, always move forward (`"replan": false`).

### Decide `"goto"`
- If `"replan": true` → `"goto": "planner"`
- If current step made progress → move to next step's agent
- Otherwise execute current step's assigned agent (`{plan_agent}`)

### Build `"query"`
Write a clear, standalone instruction for the chosen agent.
For researchers, write a question in plain language.
Use consistent language with the user's query.

### Sub-goal awareness
Consider the current step's goal and post-conditions from the plan:
- Goal: {plan_block.get('goal', 'N/A')}
- Post-conditions: {plan_block.get('post_conditions', [])}

Context:
- User query: {state.get("user_query")}
- Current step: {step}
- Current plan step: {plan_block}
- Just replanned: {state.get("replan_flag")}
- Previous messages: {messages_tail}

Respond **only** with JSON, no extra text.
"""
    
    return HumanMessage(content=prompt)

# ============================================================================
# AGENT SYSTEM PROMPT
# ============================================================================

def agent_system_prompt(specific_instructions: str) -> str:
    """
    Build a system prompt for an agent with specific instructions
    
    Args:
        specific_instructions: Specific instructions for this agent
        
    Returns:
        Formatted system prompt
    """
    base = """
You are a specialized agent in a multi-agent system.

Your responsibilities:
- Follow your specific role and capabilities
- Provide clear, accurate information
- Cite sources when available
- Be concise but thorough
- Do not attempt actions outside your capabilities

"""
    return base + "\n" + specific_instructions

print("✓ Prompts module loaded successfully!")
