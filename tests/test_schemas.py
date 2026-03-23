"""Tests for Pydantic schema validation and serialization."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from schemas.reasoning import ReasoningChain, ReasoningStep, ToolCall
from schemas.verification import StepVerification, VerificationReport


class TestReasoningSchemas:
    """Test reasoning chain schemas."""

    def test_tool_call_valid(self) -> None:
        """Test valid ToolCall construction."""
        tc = ToolCall(
            tool_name="search",
            tool_input={"query": "test"},
            tool_output="result",
        )
        assert tc.tool_name == "search"

    def test_reasoning_step_valid(self) -> None:
        """Test valid ReasoningStep construction."""
        step = ReasoningStep(
            step_number=1,
            description="First step",
            reasoning="Think about it",
            conclusion="Done",
        )
        assert step.step_number == 1
        assert step.tool_calls == []

    def test_reasoning_step_invalid_number(self) -> None:
        """Test step_number must be >= 1."""
        with pytest.raises(ValidationError):
            ReasoningStep(
                step_number=0,
                description="Bad",
                reasoning="Bad",
                conclusion="Bad",
            )

    def test_chain_valid(self, sample_chain: ReasoningChain) -> None:
        """Test valid ReasoningChain construction."""
        assert sample_chain.confidence == 0.95
        assert len(sample_chain.steps) == 1

    def test_chain_confidence_too_high(self) -> None:
        """Test confidence > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            ReasoningChain(
                query="test",
                steps=[
                    ReasoningStep(
                        step_number=1,
                        description="s",
                        reasoning="r",
                        conclusion="c",
                    )
                ],
                final_answer="a",
                confidence=1.5,
            )

    def test_chain_confidence_too_low(self) -> None:
        """Test confidence < 0.0 is rejected."""
        with pytest.raises(ValidationError):
            ReasoningChain(
                query="test",
                steps=[
                    ReasoningStep(
                        step_number=1,
                        description="s",
                        reasoning="r",
                        conclusion="c",
                    )
                ],
                final_answer="a",
                confidence=-0.1,
            )

    def test_chain_requires_at_least_one_step(self) -> None:
        """Test that empty steps list is rejected."""
        with pytest.raises(ValidationError):
            ReasoningChain(
                query="test",
                steps=[],
                final_answer="a",
                confidence=0.5,
            )

    def test_chain_serialization_roundtrip(self, sample_chain: ReasoningChain) -> None:
        """Test JSON serialization and deserialization preserve data."""
        json_str = sample_chain.model_dump_json()
        restored = ReasoningChain.model_validate_json(json_str)
        assert restored == sample_chain

    def test_chain_steps_as_json_string(self) -> None:
        """Test that steps encoded as a JSON string (LM Studio quirk) are coerced."""
        import json

        step = {
            "step_number": 1,
            "description": "Calculate",
            "reasoning": "Because math",
            "tool_calls": [],
            "conclusion": "Done",
        }
        chain = ReasoningChain.model_validate(
            {
                "query": "How many seconds in a leap year?",
                "steps": json.dumps([step]),  # string instead of list
                "final_answer": "31622400",
                "confidence": 0.99,
            }
        )
        assert len(chain.steps) == 1
        assert chain.steps[0].step_number == 1


class TestVerificationSchemas:
    """Test verification report schemas."""

    def test_step_verification_valid(self) -> None:
        """Test valid StepVerification construction."""
        sv = StepVerification(
            step_number=1,
            is_valid=False,
            issues=["Unsupported claim"],
            severity="medium",
        )
        assert not sv.is_valid
        assert len(sv.issues) == 1

    def test_report_valid(self, sample_report: VerificationReport) -> None:
        """Test valid VerificationReport construction."""
        assert sample_report.chain_is_valid is True
        assert sample_report.overall_score == 0.95

    def test_report_score_too_high(self) -> None:
        """Test overall_score > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            VerificationReport(
                chain_is_valid=True,
                overall_score=1.1,
                step_verifications=[],
                summary="Bad score",
            )

    def test_report_serialization_roundtrip(
        self, sample_report: VerificationReport
    ) -> None:
        """Test JSON serialization and deserialization preserve data."""
        json_str = sample_report.model_dump_json()
        restored = VerificationReport.model_validate_json(json_str)
        assert restored == sample_report


class TestRetrievalSchemas:
    """Test RAG retrieval schemas."""

    def test_retrieved_chunk_valid(self) -> None:
        """Test valid RetrievedChunk construction."""
        from schemas.retrieval import RetrievedChunk

        chunk = RetrievedChunk(
            text="Photosynthesis converts sunlight to energy.",
            source="biology.txt",
            chunk_index=0,
            score=0.92,
        )
        assert chunk.score == 0.92
        assert chunk.source == "biology.txt"

    def test_retrieved_chunk_score_bounds(self) -> None:
        """Test score must be between 0.0 and 1.0."""
        from pydantic import ValidationError

        from schemas.retrieval import RetrievedChunk

        with pytest.raises(ValidationError):
            RetrievedChunk(
                text="x", source="f.txt", chunk_index=0, score=1.5
            )

    def test_retrieval_result_valid(self) -> None:
        """Test valid RetrievalResult with multiple chunks."""
        from schemas.retrieval import RetrievedChunk, RetrievalResult

        chunks = [
            RetrievedChunk(
                text="Fact one.", source="a.txt", chunk_index=0, score=0.9
            ),
            RetrievedChunk(
                text="Fact two.", source="b.txt", chunk_index=1, score=0.8
            ),
        ]
        result = RetrievalResult(query="test query", chunks=chunks)
        assert len(result.chunks) == 2
        assert result.query == "test query"

    def test_retrieval_result_empty_chunks(self) -> None:
        """Test RetrievalResult accepts empty chunks list."""
        from schemas.retrieval import RetrievalResult

        result = RetrievalResult(query="unknown topic", chunks=[])
        assert result.chunks == []

    def test_retrieval_result_as_context_string(self) -> None:
        """Test formatted context output for injection into prompts."""
        from schemas.retrieval import RetrievedChunk, RetrievalResult

        chunks = [
            RetrievedChunk(
                text="Water boils at 100°C.", source="science.txt",
                chunk_index=0, score=0.95
            ),
        ]
        result = RetrievalResult(query="boiling point", chunks=chunks)
        ctx = result.as_context_string()
        assert "Water boils at 100°C." in ctx
        assert "science.txt" in ctx
