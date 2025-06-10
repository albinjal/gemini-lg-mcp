import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ResearchConfig(BaseModel):
    """Configuration for the research agent MCP server."""

    # API Keys (MCP-specific)
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        metadata={"description": "Gemini API key (required)"},
    )

    langchain_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_API_KEY"),
        metadata={"description": "LangChain API key for tracing (optional)"},
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
        default="gemini-2.5-flash-preview-04-17",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    # Research Parameters (optimized for MCP timeout constraints)
    number_of_initial_queries: int = Field(
        default=2,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=1,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    # Temperature settings (MCP-specific enhancements)
    query_temperature: float = Field(
        default=1.0, metadata={"description": "Temperature for query generation"}
    )

    reflection_temperature: float = Field(
        default=1.0, metadata={"description": "Temperature for reflection"}
    )

    answer_temperature: float = Field(
        default=0.0, metadata={"description": "Temperature for final answer generation"}
    )

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

    @field_validator("gemini_api_key")
    @classmethod
    def validate_gemini_api_key(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            print(
                "WARNING: GEMINI_API_KEY not found. Please set it as an environment variable."
            )
        return v

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """Create configuration from environment variables with better timeout defaults"""

        # Set default MCP timeout values if not already set
        if not os.environ.get("MCP_SERVER_REQUEST_TIMEOUT"):
            os.environ["MCP_SERVER_REQUEST_TIMEOUT"] = (
                "120"  # 2 minutes for research operations
            )
        if not os.environ.get("MCP_REQUEST_MAX_TOTAL_TIMEOUT"):
            os.environ["MCP_REQUEST_MAX_TOTAL_TIMEOUT"] = "300"  # 5 minutes total

        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
            query_generator_model=os.getenv(
                "QUERY_GENERATION_MODEL",
                cls.model_fields["query_generator_model"].default,
            ),
            reflection_model=os.getenv(
                "REFLECTION_MODEL", cls.model_fields["reflection_model"].default
            ),
            answer_model=os.getenv(
                "FINAL_ANSWER_MODEL", cls.model_fields["answer_model"].default
            ),
            number_of_initial_queries=int(
                os.getenv(
                    "NUMBER_OF_INITIAL_QUERIES",
                    str(cls.model_fields["number_of_initial_queries"].default),
                )
            ),
            max_research_loops=int(
                os.getenv(
                    "MAX_RESEARCH_LOOPS",
                    str(cls.model_fields["max_research_loops"].default),
                )
            ),
            query_temperature=float(
                os.getenv(
                    "TEMPERATURE", str(cls.model_fields["query_temperature"].default)
                )
            ),
            reflection_temperature=float(
                os.getenv(
                    "REFLECTION_TEMPERATURE",
                    str(cls.model_fields["reflection_temperature"].default),
                )
            ),
            answer_temperature=float(
                os.getenv(
                    "ANSWER_TEMPERATURE",
                    str(cls.model_fields["answer_temperature"].default),
                )
            ),
        )

    def get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d")
