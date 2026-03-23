"""Shared test fixtures for the reasoning agent test suite."""

from __future__ import annotations

import pytest

from schemas.reasoning import ReasoningChain, ReasoningStep, ToolCall
from schemas.verification import StepVerification, VerificationReport


@pytest.fixture
def sample_tool_call() -> ToolCall:
    """Return a sample ToolCall for testing."""
    return ToolCall(
        tool_name="calculator",
        tool_input={"expression": "100 * 0.15"},
        tool_output="15",
    )


@pytest.fixture
def sample_reasoning_step(sample_tool_call: ToolCall) -> ReasoningStep:
    """Return a sample ReasoningStep for testing."""
    return ReasoningStep(
        step_number=1,
        description="Calculate 15% of 100",
        reasoning="To find 15% of 100, multiply 100 by 0.15",
        tool_calls=[sample_tool_call],
        conclusion="15% of 100 is 15",
    )


@pytest.fixture
def sample_chain(sample_reasoning_step: ReasoningStep) -> ReasoningChain:
    """Return a sample ReasoningChain for testing."""
    return ReasoningChain(
        query="What is 15% of 100?",
        steps=[sample_reasoning_step],
        final_answer="15% of 100 is 15.",
        confidence=0.95,
    )


@pytest.fixture
def sample_step_verification() -> StepVerification:
    """Return a sample StepVerification for testing."""
    return StepVerification(
        step_number=1,
        is_valid=True,
        issues=[],
        severity="none",
    )


@pytest.fixture
def sample_report(sample_step_verification: StepVerification) -> VerificationReport:
    """Return a sample VerificationReport for testing."""
    return VerificationReport(
        chain_is_valid=True,
        overall_score=0.95,
        step_verifications=[sample_step_verification],
        logical_errors=[],
        potential_hallucinations=[],
        completeness_issues=[],
        summary="Reasoning chain is sound and well-supported.",
    )
