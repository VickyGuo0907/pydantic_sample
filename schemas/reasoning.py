"""Pydantic models for multi-step reasoning chains."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    """Record of a single tool invocation during a reasoning step."""

    tool_name: str = Field(description="Name of the tool that was called")
    tool_input: dict[str, Any] = Field(description="Arguments passed to the tool")
    tool_output: str = Field(description="Result returned by the tool")


class ReasoningStep(BaseModel):
    """A single step in the reasoning chain."""

    step_number: int = Field(ge=1, description="Sequential step number")
    description: str = Field(description="What this step aims to accomplish")
    reasoning: str = Field(description="Chain-of-thought reasoning for this step")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tools invoked during this step (can be empty)",
    )
    conclusion: str = Field(description="What was established by this step")


class ReasoningChain(BaseModel):
    """Complete reasoning chain from query to final answer."""

    query: str = Field(description="The original user query")
    steps: list[ReasoningStep] = Field(
        min_length=1,
        description="Ordered reasoning steps",
    )
    final_answer: str = Field(description="The synthesized final answer")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the final answer (0.0 to 1.0)",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_string_fields(cls, data: Any) -> Any:
        """Handle local models that return steps as a JSON-encoded string.

        Some LLMs (e.g. LM Studio) serialize nested objects as strings instead
        of proper JSON arrays. This validator parses them back to Python objects
        before normal field validation runs.
        """
        if not isinstance(data, dict):
            return data
        if isinstance(data.get("steps"), str):
            try:
                data["steps"] = json.loads(data["steps"])
            except (json.JSONDecodeError, ValueError):
                pass
        return data
