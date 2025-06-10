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
        """Create configuration from environment variables"""

        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            query_generator_model=os.getenv(
                "QUERY_GENERATION_MODEL",
                cls.model_fields["query_generator_model"].default,
            ),
            reflection_model=os.getenv(
                "REFLECTION_MODEL", cls.model_fields["reflection_model"].default
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
        )

    def get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d")
