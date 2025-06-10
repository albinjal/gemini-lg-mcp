
from src.models import SearchQueryList, Reflection
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from src.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from src.config import ResearchConfig
from src.prompts import (
    get_current_date,
    QUERY_WRITER_INSTRUCTIONS,
    WEB_SEARCHER_INSTRUCTIONS,
    REFLECTION_INSTRUCTIONS,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)


# Global config instance
global_config = ResearchConfig.from_env()

# Used for Google Search API
genai_client = Client(api_key=global_config.gemini_api_key)


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = global_config.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=global_config.query_generator_model,
        temperature=global_config.query_temperature,
        max_retries=2,
        api_key=global_config.gemini_api_key,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = QUERY_WRITER_INSTRUCTIONS.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    formatted_prompt = WEB_SEARCHER_INSTRUCTIONS.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=global_config.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or global_config.reflection_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = REFLECTION_INSTRUCTIONS.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=global_config.reflection_temperature,
        max_retries=2,
        api_key=global_config.gemini_api_key,
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to end the research based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "format_results")
    """
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else global_config.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "format_results"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def format_results(state: OverallState, config: RunnableConfig):
    """LangGraph node that formats the research results for the calling agent.

    Instead of generating a final answer, this returns structured research data
    that the calling agent can use to synthesize as needed.

    Args:
        state: Current graph state containing all research data

    Returns:
        Dictionary with formatted research results
    """
    from langchain_core.messages import AIMessage

    # Format the research results into a structured format
    research_summary = {
        "research_topic": get_research_topic(state["messages"]),
        "total_search_queries": len(state["search_query"]),
        "search_queries_used": state["search_query"],
        "research_loops_completed": state["research_loop_count"],
        "research_results": state["web_research_result"],
        "sources_and_citations": state["sources_gathered"],
        "reflection_analysis": {
            "knowledge_gaps_identified": state.get("knowledge_gap", "None"),
            "research_deemed_sufficient": state.get("is_sufficient", False),
            "follow_up_queries_suggested": state.get("follow_up_queries", []),
        },
    }

    # Create a formatted message with the research data
    formatted_content = f"""Research completed for: {research_summary['research_topic']}

RESEARCH SUMMARY:
- Total search queries executed: {research_summary['total_search_queries']}
- Research loops completed: {research_summary['research_loops_completed']}
- Research deemed sufficient: {research_summary['reflection_analysis']['research_deemed_sufficient']}

SEARCH QUERIES USED:
{chr(10).join(f"• {query}" for query in research_summary['search_queries_used'])}

RESEARCH FINDINGS:
{chr(10).join(f"{i+1}. {result}" for i, result in enumerate(research_summary['research_results']))}

SOURCES & CITATIONS:
{chr(10).join(f"• {source.get('value', str(source))}" for source in research_summary['sources_and_citations'])}

REFLECTION ANALYSIS:
- Knowledge gaps identified: {research_summary['reflection_analysis']['knowledge_gaps_identified']}
- Suggested follow-up queries: {', '.join(research_summary['reflection_analysis']['follow_up_queries_suggested']) if research_summary['reflection_analysis']['follow_up_queries_suggested'] else 'None'}

This research data can now be synthesized and analyzed as needed."""

    return {
        "messages": [AIMessage(content=formatted_content)],
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=ResearchConfig)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("format_results", format_results)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "format_results"]
)
# Format and return results instead of generating final answer
builder.add_edge("format_results", END)

graph = builder.compile(name="pro-search-agent")
