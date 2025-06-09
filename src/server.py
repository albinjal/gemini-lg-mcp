#!/usr/bin/env python3
"""
Gemini Research Agent MCP Server

A comprehensive research agent powered by Gemini models that performs web research
using Google Search API and provides detailed, cited responses.
"""

import asyncio
import json
import os
from typing import Any, Sequence

from mcp.server.fastmcp import FastMCP
from mcp.types import (
    TextContent,
    Tool,
    INVALID_PARAMS,
    INTERNAL_ERROR
)

from src.config import ResearchConfig
from src.graph import graph

# Initialize MCP server
mcp = FastMCP("Gemini Research Agent")

# Initialize configuration
config = ResearchConfig.from_env()

@mcp.tool()
async def research_question(question: str) -> str:
    """
    Conduct comprehensive research on any topic using Google's Gemini AI models.

    This tool provides sophisticated AI-powered research capabilities including:
    - Multi-query search strategy generation
    - Web search with Gemini grounding
    - Iterative refinement and reflection
    - Source citation and verification
    - Comprehensive synthesis of findings

    The research process uses a multi-agent workflow that generates search queries,
    conducts web searches, reflects on findings, and produces detailed analyses with
    proper source citations.

    Args:
        question: The research question or topic to investigate

    Returns:
        A comprehensive research report with findings, analysis, and source citations

    Raises:
        TimeoutError: If research takes longer than configured timeout
        ValueError: If question is empty or invalid
        RuntimeError: If research workflow fails
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    try:
        # Set up timeout handling
        timeout_seconds = int(os.environ.get('MCP_SERVER_REQUEST_TIMEOUT', '60'))

        async def run_research():
            """Run the research workflow with timeout protection"""
            try:
                # Prepare initial state
                from langchain_core.messages import HumanMessage

                initial_state = {
                    "messages": [HumanMessage(content=question.strip())],
                    "query_list": [],
                    "search_query": [],
                    "web_research_result": [],
                    "research_loop_count": 0,
                    "number_of_ran_queries": 0,
                    "sources_gathered": [],
                    "initial_search_query_count": None,
                    "max_research_loops": None,
                    "reasoning_model": None,
                    "is_sufficient": False,
                    "knowledge_gap": None,
                    "follow_up_queries": []
                }

                # Run the research graph
                result = await graph.ainvoke(initial_state)

                # Return the final answer or a summary of results
                if result.get("messages") and len(result["messages"]) > 0:
                    # Get the final AI message content
                    final_message = result["messages"][-1]
                    if hasattr(final_message, 'content'):
                        return final_message.content
                    else:
                        return str(final_message)
                elif result.get("web_research_result"):
                    # Fallback: provide what we have from research
                    summary = f"Research on: {question}\n\n"
                    summary += "Key findings:\n\n"
                    for i, research_result in enumerate(result["web_research_result"], 1):
                        summary += f"{i}. {research_result}\n\n"

                    # Add sources if available
                    if result.get("sources_gathered"):
                        summary += "\n--- Sources ---\n"
                        for source in result["sources_gathered"]:
                            if isinstance(source, dict) and "value" in source:
                                summary += f"• {source['value']}\n"

                    return summary
                else:
                    return f"Unable to complete research on '{question}'. Please try a more specific question."

            except Exception as e:
                error_msg = f"Research workflow error: {str(e)}"

                # Check if it's a timeout-related error
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    return f"""Research timed out for: {question}

This typically happens with complex research topics that require extensive web searches.

Suggestions:
- Try a more specific question
- Break down complex topics into smaller questions
- Set a higher timeout with MCP_SERVER_REQUEST_TIMEOUT environment variable

Error details: {str(e)}"""
                else:
                    return f"Research failed: {error_msg}"

        # Run with timeout - use asyncio.wait_for instead of asyncio.run
        try:
            result = await asyncio.wait_for(run_research(), timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError:
            return f"""Research timed out after {timeout_seconds} seconds for: {question}

The research workflow exceeded the configured timeout limit.

To resolve this:
1. Set a higher timeout: export MCP_SERVER_REQUEST_TIMEOUT=120
2. Try a more specific research question
3. Break complex topics into smaller questions

Current timeout: {timeout_seconds}s
Suggested timeout for complex research: 120-300s"""

    except Exception as e:
        error_msg = f"Error in research_question: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"Research failed: {error_msg}"

@mcp.resource("research://config")
def get_config() -> str:
    """Get the current research configuration settings"""
    config_dict = {
        "current_models": {
            "query_generation": config.query_generator_model,
            "reflection": config.reflection_model,
            "final_answer": config.answer_model
        },
        "research_parameters": {
            "number_of_initial_queries": config.number_of_initial_queries,
            "max_research_loops": config.max_research_loops,
            "temperature": config.query_temperature
        },
        "timeout_settings": {
            "mcp_server_request_timeout": os.environ.get('MCP_SERVER_REQUEST_TIMEOUT', '60'),
            "mcp_request_max_total_timeout": os.environ.get('MCP_REQUEST_MAX_TOTAL_TIMEOUT', '300'),
            "suggestion": "Set MCP_SERVER_REQUEST_TIMEOUT=120 for complex research"
        },
        "environment": {
            "current_date": config.get_current_date(),
            "gemini_api_configured": bool(config.gemini_api_key),
            "langchain_tracing": bool(config.langchain_api_key)
        }
    }
    return json.dumps(config_dict, indent=2)

if __name__ == "__main__":
    # Print configuration info on startup
    print("=" * 60)
    print("🔬 Gemini Research Agent MCP Server")
    print("=" * 60)
    print(f"📊 Query Model: {config.query_generator_model}")
    print(f"🤔 Reflection Model: {config.reflection_model}")
    print(f"📝 Answer Model: {config.answer_model}")
    print(f"🔍 Initial Queries: {config.number_of_initial_queries}")
    print(f"🔄 Max Research Loops: {config.max_research_loops}")

    # Print timeout configuration
    timeout_env = os.environ.get('MCP_SERVER_REQUEST_TIMEOUT', '60')
    max_timeout_env = os.environ.get('MCP_REQUEST_MAX_TOTAL_TIMEOUT', '300')
    print(f"⏱️  Request Timeout: {timeout_env}s")
    print(f"⏱️  Max Total Timeout: {max_timeout_env}s")

    if int(timeout_env) < 120:
        print("⚠️  Consider setting MCP_SERVER_REQUEST_TIMEOUT=120 for complex research")

    print("=" * 60)
    print("Tools: research_question")
    print("Resources: research://config")
    print("=" * 60)

    # Start the server
    mcp.run()
