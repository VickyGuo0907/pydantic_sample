"""Pydantic models for reasoning chain verification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StepVerification(BaseModel):
    """Verification result for a single reasoning step."""

    step_number: int = Field(ge=1, description="Which step this verifies")
    is_valid: bool = Field(description="Whether the step's logic is sound")
    issues: list[str] = Field(
        default_factory=list,
        description="Issues found (empty if valid)",
    )
    severity: Literal["none", "low", "medium", "high", "critical"] = Field(
        default="none",
        description="Highest severity of issues found",
    )


class VerificationReport(BaseModel):
    """Full verification report for a reasoning chain."""

    chain_is_valid: bool = Field(description="Whether the overall chain is sound")
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score (0.0 to 1.0)",
    )
    step_verifications: list[StepVerification] = Field(
        description="Per-step verification results",
    )
    logical_errors: list[str] = Field(
        default_factory=list,
        description="Logical errors found across the chain",
    )
    potential_hallucinations: list[str] = Field(
        default_factory=list,
        description="Claims that may be fabricated",
    )
    completeness_issues: list[str] = Field(
        default_factory=list,
        description="Gaps in the reasoning",
    )
    summary: str = Field(description="Human-readable verification summary")
