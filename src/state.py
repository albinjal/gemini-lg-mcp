from typing import List, TypedDict, Optional, Any
from langchain_core.messages import BaseMessage


class OverallState(TypedDict):
    """The overall state of the research agent graph."""

    # Input
    messages: List[BaseMessage]

    # Configuration options
    initial_search_query_count: Optional[int]
    max_research_loops: Optional[int]
    reasoning_model: Optional[str]

    # Search queries and results
    query_list: Optional[List[str]]
    search_query: List[str]
    web_research_result: List[str]

    # Research loop tracking
    research_loop_count: int
    number_of_ran_queries: int

    # Sources and citations
    sources_gathered: List[Any]


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
