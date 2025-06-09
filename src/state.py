from typing import List, TypedDict, Optional, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import operator


class OverallState(TypedDict):
    """The overall state of the research agent graph."""

    # Input
    messages: Annotated[List[BaseMessage], add_messages]

    # Configuration options
    initial_search_query_count: Optional[int]
    max_research_loops: Optional[int]
    reasoning_model: Optional[str]

    # Search queries and results
    query_list: Optional[List[str]]
    search_query: Annotated[List[str], operator.add]
    web_research_result: Annotated[List[str], operator.add]

    # Research loop tracking
    research_loop_count: int
    number_of_ran_queries: int

    # Sources and citations
    sources_gathered: Annotated[List[Any], operator.add]


class QueryGenerationState(TypedDict):
    """State for the query generation node."""

    query_list: List[str]
    initial_search_query_count: Optional[int]


class WebSearchState(TypedDict):
    """State for the web search node."""

    search_query: str
    id: int


class ReflectionState(TypedDict):
    """State for the reflection node."""

    is_sufficient: bool
    knowledge_gap: Optional[str]
    follow_up_queries: List[str]
    research_loop_count: int
    number_of_ran_queries: int
    max_research_loops: Optional[int]
