# Gemini Research MCP Server

🔬 **AI-powered research assistant** that performs comprehensive web research using Google's Gemini AI models.

![Demo](demo.png)

## What it does

- **Smart search strategies** - Generates multiple optimized search queries
- **Iterative research** - Follows up on knowledge gaps automatically
- **Citation tracking** - Preserves source URLs and references
- **Structured output** - Returns organized research data for easy use

Perfect for research, fact-checking, market analysis, competitive intelligence, and staying current on any topic.

## Quick Start

### 1. Get a Gemini API Key
Get your free API key at: https://aistudio.google.com/app/prompts/new_chat

### 2. Add to Cursor / Claude Desktop

Add the following to your `.cursor/mcp.json` file or Claude Desktop `mcp.json` file:

```json
{
  "mcpServers": {
    "gemini-research": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/albinjal/gemini-lg-mcp.git", "python", "-m", "src.server"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Your client might not be able to find `uvx`. In that case run `which uvx` to find the path to `uvx` and add replace the `command` with the path to `uvx`:

```json
{
  "mcpServers": {
    "gemini-research": {
      "command": "<output from which uvx>",
      "args": ["--from", "git+https://github.com/albinjal/gemini-lg-mcp.git", "python", "-m", "src.server"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```




## What you get back

- **Search queries used** - See the research strategy
- **Research findings** - Comprehensive results with citations
- **Source URLs** - All references preserved
- **Knowledge gaps** - What areas need more research
- **Follow-up suggestions** - Ideas for deeper investigation

## Models Used

- **Query generation**: Gemini 2.0 Flash (fast, smart search strategies)
- **Reflection**: Gemini 2.5 Flash (identifies knowledge gaps)
- **No final synthesis** - Raw research data returned for your use

## Installation

```bash
git clone https://github.com/albinjal/gemini-lg-mcp.git
cd gemini-lg-mcp
uv sync
export GEMINI_API_KEY="your-key"
uv run src/server.py
```

## Why use this?

✅ **Saves time** - Automated multi-angle research
✅ **Finds quality sources** - Smart search with Google's index
✅ **Preserves citations** - Never lose track of sources
✅ **Customizable** - Adjust depth and scope as needed
✅ **Cost effective** - No expensive final processing

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) and the core logic is inspired by [Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)



*Built with LangGraph • Powered by Gemini • Designed for MCP*
