import os
from typing import Any, Optional
from pydantic import BaseModel, Field


class ResearchConfig(BaseModel):
    """Configuration for the research agent MCP server."""

    # API Keys (MCP-specific)
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        metadata={"description": "Gemini API key (required)"}
    )

    langchain_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY"),
        metadata={"description": "LangChain API key for tracing (optional)"}
    )

    # Model Configuration (matching original agent)
    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash-preview-04-17",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-pro-preview-05-06",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    # Research Parameters (matching original agent)
    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    # Temperature settings (MCP-specific enhancements)
    query_temperature: float = Field(
        default=1.0,
        metadata={"description": "Temperature for query generation"}
    )

    reflection_temperature: float = Field(
        default=1.0,
        metadata={"description": "Temperature for reflection"}
    )

    answer_temperature: float = Field(
        default=0.0,
        metadata={"description": "Temperature for final answer generation"}
    )

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """Create configuration from environment variables."""
        # Get raw values from environment variables
        raw_values: dict[str, Any] = {}

        # Handle model configuration with environment override
        for field_name in ["query_generator_model", "reflection_model", "answer_model"]:
            env_value = os.environ.get(field_name.upper())
            if env_value:
                raw_values[field_name] = env_value

        # Handle numeric fields with environment override
        for field_name in ["number_of_initial_queries", "max_research_loops"]:
            env_value = os.environ.get(field_name.upper())
            if env_value:
                raw_values[field_name] = int(env_value)

        # Handle temperature fields with environment override
        for field_name in ["query_temperature", "reflection_temperature", "answer_temperature"]:
            env_value = os.environ.get(field_name.upper())
            if env_value:
                raw_values[field_name] = float(env_value)

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        config = cls(**values)
        config.validate()
        return config
