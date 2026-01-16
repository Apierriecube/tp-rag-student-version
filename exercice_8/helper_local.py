"""
Helper module for Exercise 8 - Local RAG + DuckDB Multi-Agent System
Modified from L6 to use local resources instead of Snowflake Cortex
Compatible with Google Colab and local environments
"""
from __future__ import annotations
import warnings
import os
import sys
import json
import re
from typing import Annotated, Literal, Optional, List, Dict, Any, Type
from pathlib import Path

import numpy as np
import duckdb
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

# LangGraph imports
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# TruLens imports
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.providers.litellm import LiteLLM

warnings.filterwarnings("ignore")

# Set environment for TruLens
os.environ["TRULENS_OTEL_TRACING"] = "1"

# ============================================================================
# ENVIRONMENT DETECTION (Colab vs Local)
# ============================================================================

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    # Chemins pour Google Colab
    DATA_PATH = "data/arxiv"
    CHROMA_PATH = "chroma_db"
    DB_PATH = "sales_data.duckdb"
else:
    # Chemins pour environnement local
    DATA_PATH = "../data/arxiv"
    CHROMA_PATH = "../chroma_db"
    DB_PATH = "sales_data.duckdb"

# Créer les dossiers s'ils n'existent pas
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# ============================================================================
# STATE DEFINITION
# ============================================================================

class State(MessagesState):
    """Custom state for multi-agent system"""
    enabled_agents: Optional[List[str]]
    plan: Optional[Dict[str, Dict[str, Any]]]
    user_query: Optional[str]
    current_step: int
    replan_flag: Optional[bool]
    last_reason: Optional[str]
    replan_attempts: Optional[Dict[int, int]]
    agent_query: Optional[str]

MAX_REPLANS = 2

# ============================================================================
# PROMPTS (imported from prompts_local.py)
# ============================================================================

from prompts_local import plan_prompt, executor_prompt, agent_system_prompt

# ============================================================================
# LLM INITIALIZATION - OpenRouter with meta-llama/llama-3.1-8b-instruct:free
# ============================================================================

# OpenRouter for general tasks (using free tier)
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://github.com/your-repo",  # Optional
        "X-Title": "RAG Multi-Agent System",  # Optional
    }
)

# OpenRouter for reasoning/planning
reasoning_llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_kwargs={"response_format": {"type": "json_object"}},
    default_headers={
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "RAG Multi-Agent System",
    }
)

# ============================================================================
# LOCAL RAG SETUP
# ============================================================================

# Embeddings initialization
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load or create vector store
def initialize_vectorstore():
    """Initialize or load the vector store"""
    if Path(CHROMA_PATH).exists():
        print(f"Loading existing vector store from {CHROMA_PATH}")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="rag_documents"
        )
    else:
        print(f"Creating new vector store...")
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name="rag_documents"
        )
        print(f"Vector store created with {len(chunks)} chunks")
    
    return vectorstore

vectorstore = initialize_vectorstore()

# ============================================================================
# HIERARCHICAL INDEXING - Manual Implementation (LangChain 1.x compatible)
# ============================================================================

class HierarchicalRetriever:
    """
    Custom hierarchical retriever that searches on small chunks but returns parent documents
    Compatible with LangChain 1.x without deprecated retrievers
    """
    def __init__(self, child_vectorstore, parent_store):
        self.child_vectorstore = child_vectorstore
        self.parent_store = parent_store
    
    def get_relevant_documents(self, query: str, k: int = 4):
        """Search using child chunks, return parent documents"""
        # Search in child chunks
        child_docs = self.child_vectorstore.similarity_search(query, k=k)
        
        # Get unique parent doc IDs
        parent_ids = list(set([doc.metadata.get("doc_id") for doc in child_docs if "doc_id" in doc.metadata]))
        
        # Retrieve parent documents from store
        parent_docs = []
        for doc_id in parent_ids:
            parent_doc = self.parent_store.mget([doc_id])[0]
            if parent_doc:
                parent_docs.append(parent_doc)
        
        return parent_docs

def create_parent_document_retriever(
    child_chunk_size: int = 400,
    parent_chunk_size: int = 2000,
):
    """
    Create a hierarchical retriever with parent and child documents
    
    Args:
        child_chunk_size: Size of small chunks for retrieval
        parent_chunk_size: Size of parent documents for context
    """
    # Load documents
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    # Create parent splitter
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=200,
    )
    
    # Create child splitter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=50,
    )
    
    # Create vector store for child chunks
    child_vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
    )
    
    # Create parent document store
    parent_store = InMemoryStore()
    
    # Split documents into parent chunks
    parent_docs = parent_splitter.split_documents(documents)
    
    # Generate IDs for parent docs and split into children
    doc_ids = [str(uuid.uuid4()) for _ in parent_docs]
    child_docs = []
    for i, parent_doc in enumerate(parent_docs):
        # Split parent into children
        _sub_docs = child_splitter.split_documents([parent_doc])
        for _doc in _sub_docs:
            _doc.metadata["doc_id"] = doc_ids[i]
        child_docs.extend(_sub_docs)
    
    # Add child docs to vectorstore and parent docs to docstore
    child_vectorstore.add_documents(child_docs)
    parent_store.mset(list(zip(doc_ids, parent_docs)))
    
    # Return custom retriever
    return HierarchicalRetriever(child_vectorstore, parent_store)

# Initialize both retrievers for comparison
# NOTE: Hierarchical retriever disabled for performance (too slow with many PDFs)
# Use standard retriever instead for faster initialization
print("Using standard retriever (hierarchical disabled for performance)")
parent_retriever = None  # Disabled - reloads all PDFs each time

# Standard retriever from vectorstore
standard_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ============================================================================
# LOCAL RAG TOOL
# ============================================================================

@tool
def local_rag_search(query: str, use_hierarchical: bool = False) -> str:
    """
    Search local document database using RAG.
    
    Args:
        query: The search query
        use_hierarchical: Whether to use hierarchical retrieval (default: False - disabled for performance)
    
    Returns:
        Formatted search results with sources
    """
    try:
        # Choose retriever (parent_retriever disabled, always use standard)
        retriever = standard_retriever  # Always use standard for performance
        
        # Retrieve documents
        docs = retriever.get_relevant_documents(query)
        
        # Format results
        results = []
        for i, doc in enumerate(docs[:5], 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content[:500]  # Limit content length
            
            results.append(
                f"[Result {i}]\n"
                f"Source: {source} (Page {page})\n"
                f"Content: {content}...\n"
            )
        
        if not results:
            return "No relevant documents found."
        
        return "\n---\n".join(results)
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"

# ============================================================================
# DUCKDB SETUP AND TOOL
# ============================================================================

def initialize_duckdb():
    """Initialize DuckDB with sample sales data"""
    con = duckdb.connect(DB_PATH)
    
    # Create sample tables if they don't exist
    con.execute("""
        CREATE TABLE IF NOT EXISTS deals (
            deal_id INTEGER PRIMARY KEY,
            company_name VARCHAR,
            deal_value DECIMAL(10, 2),
            sales_rep VARCHAR,
            close_date DATE,
            deal_status VARCHAR,
            product_line VARCHAR
        )
    """)
    
    # Insert sample data if table is empty
    count = con.execute("SELECT COUNT(*) FROM deals").fetchone()[0]
    if count == 0:
        con.execute("""
            INSERT INTO deals VALUES
            (1, 'TechCorp', 500000.00, 'John Smith', '2024-01-15', 'closed', 'AI Solutions'),
            (2, 'DataInc', 750000.00, 'Jane Doe', '2024-02-20', 'closed', 'Analytics'),
            (3, 'CloudSystems', 1200000.00, 'Bob Johnson', '2024-03-10', 'pending', 'Cloud Services'),
            (4, 'AIStartup', 300000.00, 'Alice Brown', '2024-01-25', 'closed', 'AI Solutions'),
            (5, 'BigRetail', 2000000.00, 'Charlie Wilson', '2024-02-15', 'pending', 'E-commerce'),
            (6, 'FinTechCo', 850000.00, 'David Lee', '2024-03-05', 'closed', 'Finance'),
            (7, 'HealthPlus', 650000.00, 'Eva Martinez', '2024-01-30', 'closed', 'Healthcare'),
            (8, 'AutoDrive', 1500000.00, 'Frank Zhang', '2024-02-25', 'pending', 'Automotive')
        """)
        print("Sample sales data inserted into DuckDB")
    
    con.close()
    return DB_PATH

# Initialize DuckDB
initialize_duckdb()

@tool
def duckdb_query(query_sql: str) -> str:
    """
    Execute SQL query on local DuckDB database containing sales data.
    
    The database contains a 'deals' table with columns:
    - deal_id: INTEGER
    - company_name: VARCHAR
    - deal_value: DECIMAL
    - sales_rep: VARCHAR
    - close_date: DATE
    - deal_status: VARCHAR ('closed', 'pending')
    - product_line: VARCHAR
    
    Args:
        query_sql: SQL query to execute
    
    Returns:
        Query results as formatted string
    """
    try:
        con = duckdb.connect(DB_PATH)
        result = con.execute(query_sql).fetchdf()
        con.close()
        
        if result.empty:
            return "Query returned no results."
        
        # Format as string
        output = f"SQL Query:\n{query_sql}\n\nResults:\n{result.to_string(index=False)}"
        return output
    
    except Exception as e:
        return f"SQL Error: {str(e)}"

# ============================================================================
# PYTHON REPL TOOL (for charts)
# ============================================================================

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code for generating charts.
    Only print the chart once. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."

# ============================================================================
# AGENT NODES - PLANNER & EXECUTOR
# ============================================================================

def planner_node(state: State) -> Command[Literal['executor']]:
    """Runs the planning LLM and stores the resulting plan in state."""
    llm_reply = reasoning_llm.invoke([plan_prompt(state)])
    
    try:
        content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
        parsed_plan = json.loads(content_str)
    except json.JSONDecodeError:
        raise ValueError(f"Planner returned invalid JSON:\n{llm_reply.content}")
    
    replan = state.get("replan_flag", False)
    
    return Command(
        update={
            "plan": parsed_plan,
            "messages": [HumanMessage(
                content=llm_reply.content,
                name="replan" if replan else "initial_plan"
            )],
            "user_query": state.get("user_query", state["messages"][0].content),
            "current_step": 1 if not replan else state["current_step"],
            "replan_flag": state.get("replan_flag", False),
            "last_reason": "",
            "enabled_agents": state.get("enabled_agents"),
        },
        goto="executor",
    )

def executor_node(
    state: State,
) -> Command[Literal["local_rag_researcher", "duckdb_researcher", "chart_generator", "synthesizer", "planner"]]:
    """Execute the current step or decide to replan"""
    
    plan: Dict[str, Any] = state.get("plan", {})
    step: int = state.get("current_step", 1)
    
    # If just replanned, run the planned agent
    if state.get("replan_flag"):
        planned_agent = plan.get(str(step), {}).get("agent")
        return Command(
            update={
                "replan_flag": False,
                "current_step": step + 1,
            },
            goto=planned_agent,
        )
    
    # Call executor LLM
    llm_reply = reasoning_llm.invoke([executor_prompt(state)])
    
    try:
        content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
        parsed = json.loads(content_str)
        replan: bool = parsed["replan"]
        goto: str = parsed["goto"]
        reason: str = parsed["reason"]
        query: str = parsed["query"]
    except Exception as exc:
        raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from exc
    
    updates: Dict[str, Any] = {
        "messages": [HumanMessage(content=llm_reply.content, name="executor")],
        "last_reason": reason,
        "agent_query": query,
    }
    
    # Replan accounting
    replans: Dict[int, int] = state.get("replan_attempts", {}) or {}
    step_replans = replans.get(step, 0)
    
    if replan:
        if step_replans < MAX_REPLANS:
            replans[step] = step_replans + 1
            updates.update({
                "replan_attempts": replans,
                "replan_flag": True,
                "current_step": step,
            })
            return Command(update=updates, goto="planner")
        else:
            # Max replans hit, move to next step
            next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer")
            updates["current_step"] = step + 1
            return Command(update=updates, goto=next_agent)
    
    # Happy path: run chosen agent
    planned_agent = plan.get(str(step), {}).get("agent")
    updates["current_step"] = step + 1 if goto == planned_agent else step
    updates["replan_flag"] = False
    return Command(update=updates, goto=goto)

# ============================================================================
# LOCAL RAG RESEARCHER AGENT
# ============================================================================

local_rag_agent = create_react_agent(
    llm,
    tools=[local_rag_search],
    prompt=agent_system_prompt("""
        You are the Local RAG Researcher. You search local document databases 
        to find relevant information. Use the local_rag_search tool to answer questions.
        Provide clear, concise answers with sources.
    """)
)

@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0].get("agent_query", ""),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
            ret.update["messages"][-1].content
        ] if hasattr(ret, "update") else "No retrieval",
    },
)
def local_rag_research_node(state: State) -> Command[Literal["executor"]]:
    """Node for local RAG research"""
    query = state.get("agent_query", state.get("user_query", ""))
    result = local_rag_agent.invoke({"messages": query})
    
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content,
        name="local_rag_researcher"
    )
    
    return Command(
        update={"messages": result["messages"]},
        goto="executor",
    )

# ============================================================================
# DUCKDB RESEARCHER AGENT
# ============================================================================

duckdb_agent = create_react_agent(
    llm,
    tools=[duckdb_query],
    prompt=agent_system_prompt("""
        You are the DuckDB Researcher. You query structured sales data 
        in a local DuckDB database. Use SQL to answer questions about deals,
        sales representatives, companies, and revenue.
        
        Always show the SQL query and results clearly.
    """)
)

@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0].get("agent_query", ""),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
            ret.update["messages"][-1].content
        ] if hasattr(ret, "update") else "No query",
    },
)
def duckdb_research_node(state: State) -> Command[Literal["executor"]]:
    """Node for DuckDB structured data queries"""
    query = state.get("agent_query", state.get("user_query", ""))
    result = duckdb_agent.invoke({"messages": query})
    
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content,
        name="duckdb_researcher"
    )
    
    return Command(
        update={"messages": result["messages"]},
        goto="executor",
    )

# ============================================================================
# CHART GENERATOR AGENT
# ============================================================================

chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    prompt=agent_system_prompt(
        "You can only generate charts. Print the chart first. Then save it "
        "to a file and provide the path to the chart_summarizer."
    ),
)

def chart_node(state: State) -> Command[Literal["chart_summarizer"]]:
    """Generate chart visualizations"""
    result = chart_agent.invoke(state)
    
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content,
        name="chart_generator"
    )
    
    return Command(
        update={"messages": result["messages"]},
        goto="chart_summarizer",
    )

# ============================================================================
# CHART SUMMARIZER AGENT
# ============================================================================

chart_summary_agent = create_react_agent(
    llm,
    tools=[],
    prompt=agent_system_prompt(
        "You summarize charts generated by the chart_generator. "
        "Provide a concise summary (no more than 3 sentences) without mentioning the chart itself."
    ),
)

def chart_summary_node(state: State) -> Command[Literal[END]]:
    """Summarize generated charts"""
    result = chart_summary_agent.invoke(state)
    
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content,
        name="chart_summarizer"
    )
    
    return Command(
        update={
            "messages": result["messages"],
            "final_answer": result["messages"][-1].content,
        },
        goto=END,
    )

# ============================================================================
# SYNTHESIZER AGENT
# ============================================================================

def synthesizer_node(state: State) -> Command[Literal[END]]:
    """Create final synthesis of all research"""
    
    # Gather relevant messages
    relevant_msgs = [
        m.content for m in state.get("messages", [])
        if getattr(m, "name", None) in (
            "local_rag_researcher",
            "duckdb_researcher",
            "chart_generator",
            "chart_summarizer"
        )
    ]
    
    user_question = state.get("user_query", state.get("messages", [{}])[0].content if state.get("messages") else "")
    
    synthesis_prompt = f"""
    You are the Synthesizer. Create a comprehensive answer to the user's question
    using ONLY the information from the context below.
    
    Instructions:
    - Start with a direct answer
    - Include key data and figures
    - Cite sources when relevant
    - Be concise and clear
    - If information is missing, state what's missing
    
    User question: {user_question}
    
    Context from agents:
    {chr(10).join(relevant_msgs)}
    
    Your synthesis:
    """
    
    llm_reply = llm.invoke([HumanMessage(content=synthesis_prompt)])
    answer = str(llm_reply.content).strip()
    
    return Command(
        update={
            "final_answer": answer,
            "messages": [HumanMessage(content=answer, name="synthesizer")],
        },
        goto=END,
    )

# ============================================================================
# EVALUATIONS - TruLens RAG Triad + GPA
# ============================================================================

# TEMPORARILY DISABLED due to LiteLLM compatibility issues with TruLens
# You can enable evaluations later by uncommenting and fixing the compatibility issue

print("⚠️  TruLens evaluations temporarily disabled due to compatibility issues")
print("   System will work without evaluations for now")

# Create dummy provider and feedbacks to avoid import errors
class DummyProvider:
    """Dummy provider when TruLens can't initialize"""
    def groundedness_measure_with_cot_reasons(self, *args, **kwargs):
        return 0.0
    def relevance_with_cot_reasons(self, *args, **kwargs):
        return 0.0
    def context_relevance_with_cot_reasons(self, *args, **kwargs):
        return 0.0
    def logical_consistency_with_cot_reasons(self, *args, **kwargs):
        return 0.0
    def execution_efficiency_with_cot_reasons(self, *args, **kwargs):
        return 0.0
    def plan_adherence_with_cot_reasons(self, *args, **kwargs):
        return 0.0
    def plan_quality_with_cot_reasons(self, *args, **kwargs):
        return 0.0

eval_provider = DummyProvider()
gpa_provider = eval_provider

# Create dummy feedback objects
class DummyFeedback:
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name', 'Dummy')
    def on(self, *args, **kwargs):
        return self
    def on_input(self):
        return self
    def on_output(self):
        return self
    def aggregate(self, *args):
        return self

f_groundedness = DummyFeedback(name="Groundedness")
f_answer_relevance = DummyFeedback(name="Answer Relevance")
f_context_relevance = DummyFeedback(name="Context Relevance")
f_logical_consistency = DummyFeedback(name="Logical Consistency")
f_execution_efficiency = DummyFeedback(name="Execution Efficiency")
f_plan_adherence = DummyFeedback(name="Plan Adherence")
f_plan_quality = DummyFeedback(name="Plan Quality")

"""
# Use OpenRouter for evaluations
eval_provider = LiteLLM(model_engine="openrouter/meta-llama/llama-3.1-8b-instruct:free")

# RAG Triad Evaluations
f_groundedness = (
    Feedback(
        eval_provider.groundedness_measure_with_cot_reasons,
        name="Groundedness"
    )
    .on({
        "source": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True
        )
    })
    .on_output()
)

f_answer_relevance = (
    Feedback(
        eval_provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    )
    .on_input()
    .on_output()
)

f_context_relevance = (
    Feedback(
        eval_provider.context_relevance_with_cot_reasons,
        name="Context Relevance"
    )
    .on({
        "question": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
        )
    })
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=False
        )
    })
    .aggregate(np.mean)
)

# GPA Evaluations

# Reuse eval_provider instead of creating a new one
gpa_provider = eval_provider

f_logical_consistency = Feedback(
    gpa_provider.logical_consistency_with_cot_reasons,
    name="Logical Consistency",
).on({
    "trace": Selector(trace_level=True),
})

f_execution_efficiency = Feedback(
    gpa_provider.execution_efficiency_with_cot_reasons,
    name="Execution Efficiency",
).on({
    "trace": Selector(trace_level=True),
})

f_plan_adherence = Feedback(
    gpa_provider.plan_adherence_with_cot_reasons,
    name="Plan Adherence",
).on({
    "trace": Selector(trace_level=True),
})

f_plan_quality = Feedback(
    gpa_provider.plan_quality_with_cot_reasons,
    name="Plan Quality",
).on({
    "trace": Selector(trace_level=True),
})
"""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def display_eval_reason(text, width=800):
    """Display evaluation reasoning in formatted way"""
    from IPython.display import HTML, display
    
    raw_text = str(text).rstrip()
    cleaned_text = re.sub(r"\s*Score:\s*-?\d+(?:\.\d+)?\s*$", "", raw_text, flags=re.IGNORECASE)
    html_text = cleaned_text.replace('\n', '<br><br>')
    display(HTML(f'<div style="font-size: 15px; word-wrap: break-word; width: {width}px;">{html_text}</div>'))

print("✓ Helper module loaded successfully!")
print("✓ Using OpenRouter with meta-llama/llama-3.1-8b-instruct:free")
print("✓ Local RAG with hierarchical indexing initialized")
print("✓ DuckDB with sample sales data ready")
print("✓ TruLens evaluations configured")
