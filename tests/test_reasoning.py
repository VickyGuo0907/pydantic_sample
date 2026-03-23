"""Tests for the multi-step reasoning agent."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent

from agents.reasoning import ReasoningDeps, create_reasoning_agent
from schemas.reasoning import ReasoningChain
from tools.retriever import VectorStore


class TestReasoningAgent:
    """Test reasoning agent creation and structure."""

    def test_create_agent_returns_agent(self) -> None:
        """Test that factory returns a pydantic-ai Agent."""
        agent = create_reasoning_agent("test")
        assert isinstance(agent, Agent)

    def test_agent_has_tools_registered(self) -> None:
        """Test that search and calculator tools are registered."""
        agent = create_reasoning_agent("test")
        tool_names = set(agent._function_toolset.tools)
        assert "search_tool" in tool_names
        assert "calculator_tool" in tool_names

    def test_agent_output_type_is_reasoning_chain(self) -> None:
        """Test that the agent's output type is ReasoningChain."""
        agent = create_reasoning_agent("test")
        assert agent._output_type == ReasoningChain

    def test_reasoning_deps_accepts_vector_store(self) -> None:
        """ReasoningDeps can hold a VectorStore instance."""
        store = VectorStore(persist_path=None)
        deps = ReasoningDeps(vector_store=store)
        assert deps.vector_store is store

    def test_reasoning_deps_vector_store_defaults_to_none(self) -> None:
        """ReasoningDeps.vector_store is None by default (RAG disabled)."""
        deps = ReasoningDeps()
        assert deps.vector_store is None

    @pytest.mark.asyncio
    async def test_run_with_test_model(self) -> None:
        """Test running the agent with pydantic-ai TestModel."""
        agent = create_reasoning_agent("test")
        deps = ReasoningDeps(search_api_key=None)
        result = await agent.run("What is 2 + 2?", deps=deps)
        # TestModel produces a result that conforms to the schema
        assert isinstance(result.output, ReasoningChain)
        assert len(result.output.steps) >= 1
        assert result.output.query or result.output.final_answer
