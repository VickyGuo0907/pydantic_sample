"""Tests for CLI entry point functions in main.py."""

from __future__ import annotations

import sys

import pytest

from main import parse_args, print_chain, print_report
from schemas.reasoning import ReasoningChain, ReasoningStep, ToolCall
from schemas.verification import StepVerification, VerificationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain(*, verbose_step: bool = False) -> ReasoningChain:
    """Build a minimal ReasoningChain for testing print functions."""
    tool_call = ToolCall(
        tool_name="calculator",
        tool_input={"expression": "2+2"},
        tool_output="4",
    )
    step = ReasoningStep(
        step_number=1,
        description="Add two numbers",
        reasoning="Simple addition",
        tool_calls=[tool_call] if verbose_step else [],
        conclusion="2 + 2 = 4",
    )
    return ReasoningChain(
        query="What is 2+2?",
        steps=[step],
        final_answer="4",
        confidence=0.99,
    )


def _make_report(*, valid: bool = True) -> VerificationReport:
    """Build a minimal VerificationReport for testing print functions."""
    sv = StepVerification(
        step_number=1,
        is_valid=valid,
        issues=[] if valid else ["Unsupported claim"],
        severity="none" if valid else "medium",
    )
    return VerificationReport(
        chain_is_valid=valid,
        overall_score=0.95 if valid else 0.4,
        step_verifications=[sv],
        logical_errors=[] if valid else ["Non-sequitur detected"],
        potential_hallucinations=[] if valid else ["Made-up statistic"],
        completeness_issues=[] if valid else ["Query not fully addressed"],
        summary="All good." if valid else "Issues found.",
    )


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    """Test argument parsing."""

    def test_query_required(self) -> None:
        """Test that a query positional argument is captured."""
        sys.argv = ["main.py", "What is 2+2?"]
        args = parse_args()
        assert args.query == "What is 2+2?"

    def test_default_no_verify_false(self) -> None:
        """Test that --no-verify defaults to False."""
        sys.argv = ["main.py", "test"]
        args = parse_args()
        assert args.no_verify is False

    def test_no_verify_flag(self) -> None:
        """Test --no-verify sets the flag to True."""
        sys.argv = ["main.py", "test", "--no-verify"]
        args = parse_args()
        assert args.no_verify is True

    def test_verbose_flag(self) -> None:
        """Test --verbose sets the flag to True."""
        sys.argv = ["main.py", "test", "--verbose"]
        args = parse_args()
        assert args.verbose is True

    def test_provider_choice(self) -> None:
        """Test --provider accepts valid choices."""
        sys.argv = ["main.py", "test", "--provider", "lmstudio"]
        args = parse_args()
        assert args.provider == "lmstudio"

    def test_default_provider_is_none(self) -> None:
        """Test that provider defaults to None (reads from env)."""
        sys.argv = ["main.py", "test"]
        args = parse_args()
        assert args.provider is None


# ---------------------------------------------------------------------------
# print_chain
# ---------------------------------------------------------------------------

class TestPrintChain:
    """Test print_chain output (smoke tests via capsys)."""

    def test_prints_query(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that the query is printed."""
        chain = _make_chain()
        print_chain(chain)
        out = capsys.readouterr().out
        assert "What is 2+2?" in out

    def test_prints_final_answer(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that the final answer is printed."""
        chain = _make_chain()
        print_chain(chain)
        out = capsys.readouterr().out
        assert "Final Answer: 4" in out

    def test_prints_confidence(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that confidence is printed as a percentage."""
        chain = _make_chain()
        print_chain(chain)
        out = capsys.readouterr().out
        assert "99%" in out

    def test_verbose_shows_reasoning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that verbose mode shows reasoning text."""
        chain = _make_chain(verbose_step=True)
        print_chain(chain, verbose=True)
        out = capsys.readouterr().out
        assert "Simple addition" in out

    def test_verbose_shows_tool_calls(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that verbose mode shows tool call details."""
        chain = _make_chain(verbose_step=True)
        print_chain(chain, verbose=True)
        out = capsys.readouterr().out
        assert "calculator" in out

    def test_non_verbose_hides_reasoning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that non-verbose mode does not show reasoning text."""
        chain = _make_chain()
        print_chain(chain, verbose=False)
        out = capsys.readouterr().out
        assert "Simple addition" not in out


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------

class TestPrintReport:
    """Test print_report output (smoke tests via capsys)."""

    def test_prints_valid_status(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that a valid report shows VALID."""
        report = _make_report(valid=True)
        print_report(report)
        out = capsys.readouterr().out
        assert "VALID" in out

    def test_prints_issues_found_status(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that an invalid report shows ISSUES FOUND."""
        report = _make_report(valid=False)
        print_report(report)
        out = capsys.readouterr().out
        assert "ISSUES FOUND" in out

    def test_prints_score(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that the overall score is printed."""
        report = _make_report(valid=True)
        print_report(report)
        out = capsys.readouterr().out
        assert "95%" in out

    def test_prints_logical_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that logical errors section is printed when present."""
        report = _make_report(valid=False)
        print_report(report)
        out = capsys.readouterr().out
        assert "Non-sequitur detected" in out

    def test_prints_hallucinations(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that potential hallucinations are printed when present."""
        report = _make_report(valid=False)
        print_report(report)
        out = capsys.readouterr().out
        assert "Made-up statistic" in out

    def test_prints_completeness_issues(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that completeness issues are printed when present."""
        report = _make_report(valid=False)
        print_report(report)
        out = capsys.readouterr().out
        assert "Query not fully addressed" in out

    def test_prints_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that the summary line is always printed."""
        report = _make_report(valid=True)
        print_report(report)
        out = capsys.readouterr().out
        assert "All good." in out

    def test_no_logical_errors_section_when_empty(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that 'Logical Errors' header is absent when there are none."""
        report = _make_report(valid=True)
        print_report(report)
        out = capsys.readouterr().out
        assert "Logical Errors" not in out
