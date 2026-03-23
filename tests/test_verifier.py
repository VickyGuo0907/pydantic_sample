"""Tests for the verification agent."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent

from agents.verifier import create_verification_agent
from schemas.reasoning import ReasoningChain
from schemas.verification import VerificationReport


class TestVerificationAgent:
    """Test verification agent creation and structure."""

    def test_create_agent_returns_agent(self) -> None:
        """Test that factory returns a pydantic-ai Agent."""
        agent = create_verification_agent("test")
        assert isinstance(agent, Agent)

    def test_agent_output_type_is_verification_report(self) -> None:
        """Test that the agent's output type is VerificationReport."""
        agent = create_verification_agent("test")
        assert agent._output_type == VerificationReport

    def test_agent_has_no_tools(self) -> None:
        """Test that verifier has no tools (purely analytical)."""
        agent = create_verification_agent("test")
        assert len(agent._function_toolset.tools) == 0

    @pytest.mark.asyncio
    async def test_run_with_test_model(self, sample_chain: ReasoningChain) -> None:
        """Test running the verifier with pydantic-ai TestModel."""
        agent = create_verification_agent("test")
        prompt = f"Verify this:\n{sample_chain.model_dump_json(indent=2)}"
        result = await agent.run(prompt)
        assert isinstance(result.output, VerificationReport)
