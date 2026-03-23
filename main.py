"""CLI entry point for the Multi-step Reasoning Agent with Verification."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from agents.reasoning import run_reasoning
from agents.verifier import run_verification
from config.settings import Settings
from schemas.reasoning import ReasoningChain
from schemas.verification import VerificationReport
from tools.retriever import VectorStore, resolve_openai_key

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Multi-step Reasoning Agent with Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "What is 15%% of France\'s GDP?"\n'
            '  python main.py "How many seconds in a leap year?" --provider lmstudio\n'
            '  python main.py "Explain quantum tunneling" --no-verify'
        ),
    )
    parser.add_argument("query", help="The question to reason about")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "lmstudio"],
        default=None,
        help="LLM provider (overrides LLM_PROVIDER env var)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the verification step",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each step",
    )
    return parser.parse_args()


def print_chain(chain: ReasoningChain, *, verbose: bool = False) -> None:
    """Print the reasoning chain in a readable format.

    Args:
        chain: The reasoning chain to display.
        verbose: If True, show tool call details.
    """
    print(f"\n{'='*60}")
    print(f"Query: {chain.query}")
    print(f"{'='*60}")

    for step in chain.steps:
        print(f"\n--- Step {step.step_number}: {step.description} ---")
        if verbose:
            print(f"  Reasoning: {step.reasoning}")
            for tc in step.tool_calls:
                print(f"  Tool [{tc.tool_name}]: {tc.tool_input} -> {tc.tool_output}")
        print(f"  Conclusion: {step.conclusion}")

    print(f"\n{'='*60}")
    print(f"Final Answer: {chain.final_answer}")
    print(f"Confidence: {chain.confidence:.0%}")
    print(f"{'='*60}")


def print_report(report: VerificationReport) -> None:
    """Print the verification report in a readable format.

    Args:
        report: The verification report to display.
    """
    status = "VALID" if report.chain_is_valid else "ISSUES FOUND"
    print(f"\n{'='*60}")
    print(f"Verification: {status} (score: {report.overall_score:.0%})")
    print(f"{'='*60}")

    for sv in report.step_verifications:
        icon = "+" if sv.is_valid else "!"
        print(f"  [{icon}] Step {sv.step_number}: {'Valid' if sv.is_valid else sv.severity.upper()}")
        for issue in sv.issues:
            print(f"      - {issue}")

    if report.logical_errors:
        print("\n  Logical Errors:")
        for err in report.logical_errors:
            print(f"    - {err}")

    if report.potential_hallucinations:
        print("\n  Potential Hallucinations:")
        for h in report.potential_hallucinations:
            print(f"    - {h}")

    if report.completeness_issues:
        print("\n  Completeness Issues:")
        for c in report.completeness_issues:
            print(f"    - {c}")

    print(f"\n  Summary: {report.summary}")
    print(f"{'='*60}")


async def run_reasoning_pipeline(args: argparse.Namespace) -> None:
    """Execute the reasoning and verification pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    settings = Settings()
    if args.provider:
        settings.llm_provider = args.provider

    # Load vector store if RAG is enabled
    vector_store: VectorStore | None = None
    if settings.rag_enabled:
        logger.info("RAG enabled — loading vector store from %s", settings.vector_store_path)
        vector_store = VectorStore(
            persist_path=settings.vector_store_path,
            openai_api_key=resolve_openai_key(settings.openai_api_key),
            embedding_model=settings.embedding_model,
        )

    # Step 1: Multi-step reasoning
    logger.info("Reasoning about: %s (provider: %s)", args.query, settings.llm_provider)
    chain = await run_reasoning(args.query, settings, vector_store=vector_store)
    print_chain(chain, verbose=args.verbose)

    # Step 2: Verification (unless skipped)
    if not args.no_verify:
        logger.info("Verifying reasoning chain...")
        report = await run_verification(chain, settings)
        print_report(report)
    else:
        logger.info("Verification skipped by user request.")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        asyncio.run(run_reasoning_pipeline(args))
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
