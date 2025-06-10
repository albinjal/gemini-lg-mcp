from typing import List, Optional
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """A search query with rationale."""

    query: str = Field(description="The search query")
    rationale: str = Field(description="Explanation of why this query is relevant")


class SearchQueryList(BaseModel):
    """List of search queries with overall rationale."""

    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    """Analysis of research completeness."""

    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class ResearchResult(BaseModel):
    """Result from a web research operation."""

    query: str = Field(description="The search query that was executed")
    content: str = Field(description="The research content with citations")
    sources: List[dict] = Field(description="List of source information")


class ComprehensiveResearchResult(BaseModel):
    """Final result from the complete research process."""

    question: str = Field(description="The original research question")
    answer: str = Field(description="Comprehensive answer with citations")
    sources: List[dict] = Field(description="All sources used in the research")
    queries_used: List[str] = Field(description="All search queries that were executed")
    research_loops: int = Field(description="Number of research loops performed")


class ResearchSession(BaseModel):
    """State of an ongoing research session."""

    question: str = Field(description="The research question")
    queries_executed: List[str] = Field(default_factory=list)
    research_results: List[str] = Field(default_factory=list)
    sources_gathered: List[dict] = Field(default_factory=list)
    loop_count: int = Field(default=0)
    is_complete: bool = Field(default=False)
    final_answer: Optional[str] = Field(default=None)
