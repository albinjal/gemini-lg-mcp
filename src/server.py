#!/usr/bin/env python3
"""
Gemini Research Agent MCP Server

A comprehensive research agent powered by Gemini models that performs web research
using Google Search API and provides detailed, cited responses.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage
from mcp.server.fastmcp import FastMCP

from src.config import ResearchConfig
from src.graph import graph


# Initialize configuration
config = ResearchConfig.from_env()

# Create the MCP server
mcp = FastMCP("Gemini Research Agent")


@mcp.tool()
def research_question(question: str) -> str:
    """
    Perform comprehensive research on a question using web search and AI analysis.

    This tool runs the complete research workflow:
    1. Generates optimized search queries
    2. Performs web research with Google Search
    3. Analyzes completeness and identifies knowledge gaps
    4. Performs additional research if needed
    5. Synthesizes findings into a comprehensive answer with citations

    Args:
        question: The research question to investigate

    Returns:
        A comprehensive research report with citations and sources
    """
    try:
        # Create initial state with the question
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "search_query": [],
            "web_research_result": [],
            "research_loop_count": 0,
            "number_of_ran_queries": 0,
            "sources_gathered": [],
        }

        # Run the research graph
        result = graph.invoke(initial_state)

        # Extract the final answer
        if result.get("messages") and len(result["messages"]) > 0:
            final_answer = result["messages"][-1].content

            # Add source information if available
            sources = result.get("sources_gathered", [])
            if sources:
                final_answer += "\n\n## Sources\n"
                for i, source in enumerate(sources, 1):
                    final_answer += f"{i}. {source.get('value', 'Unknown source')}\n"

            return final_answer
        else:
            return "No research results were generated."

    except Exception as e:
        return f"Error during research: {str(e)}"


@mcp.resource("research://config")
def get_config() -> str:
    """Get current research configuration."""
    config_dict = {
        "query_generator_model": config.query_generator_model,
        "reflection_model": config.reflection_model,
        "answer_model": config.answer_model,
        "number_of_initial_queries": config.number_of_initial_queries,
        "max_research_loops": config.max_research_loops,
        "query_temperature": config.query_temperature,
        "reflection_temperature": config.reflection_temperature,
        "answer_temperature": config.answer_temperature,
    }
    return json.dumps(config_dict, indent=2)


if __name__ == "__main__":
    # Run the server
    mcp.run()
