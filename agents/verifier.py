"""Verification agent that audits reasoning chains for correctness."""

from __future__ import annotations

import asyncio
import logging

from pydantic_ai import Agent
from pydantic_ai.models import Model

from config.settings import Settings
from schemas.reasoning import ReasoningChain
from schemas.verification import VerificationReport

logger = logging.getLogger(__name__)

VERIFICATION_SYSTEM_PROMPT = """\
You are a verification agent. Your job is to audit a reasoning chain for quality.

For each reasoning step, evaluate:
1. **Logical validity**: Does the conclusion follow from the reasoning?
2. **Evidence basis**: Are claims supported by tool results or common knowledge?
3. **Consistency**: Does this step contradict any previous step?

Then assess the overall chain for:
- **Logical errors**: Fallacies, non-sequiturs, circular reasoning
- **Potential hallucinations**: Specific facts/numbers stated without tool verification
- **Completeness**: Does the chain fully address the original query?

Scoring guidelines:
- 0.9-1.0: Sound logic, all claims verified, fully addresses query
- 0.7-0.8: Minor issues but fundamentally sound
- 0.5-0.6: Some logical gaps or unverified claims
- Below 0.5: Significant errors or missing reasoning

Be rigorous but fair. Flag real issues, not style preferences.
"""


def create_verification_agent(model: Model | str) -> Agent[None, VerificationReport]:
    """Create a verification agent for auditing reasoning chains.

    Args:
        model: A pydantic-ai Model instance or model name string.

    Returns:
        A configured Agent that produces VerificationReport output.
    """
    return Agent(
        model,
        output_type=VerificationReport,
        system_prompt=VERIFICATION_SYSTEM_PROMPT,
        retries=3,
    )


async def _run_with_backoff(
    coro_fn,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> object:
    """Run an async callable with exponential backoff on transient failures.

    Args:
        coro_fn: A zero-argument async callable to attempt.
        max_retries: Maximum total attempts before re-raising.
        base_delay: Initial delay in seconds; doubles on each retry.

    Returns:
        The return value of coro_fn on success.

    Raises:
        ValueError: Immediately — these indicate config errors, not transience.
        Exception: Re-raised after all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except ValueError:
            raise
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Verification agent attempt %d/%d failed (%s). Retrying in %.1fs...",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


async def run_verification(
    chain: ReasoningChain, settings: Settings
) -> VerificationReport:
    """Verify a reasoning chain using an independent LLM agent.

    Args:
        chain: The reasoning chain to verify.
        settings: Application settings with provider config.

    Returns:
        A structured VerificationReport with per-step analysis.
    """
    model = settings.get_model()
    agent = create_verification_agent(model)
    prompt = (
        "Verify the following reasoning chain for logical validity, "
        "hallucinations, and completeness:\n\n"
        f"{chain.model_dump_json(indent=2)}"
    )
    logger.debug("Running verification agent on chain for query: %s", chain.query)
    result = await _run_with_backoff(lambda: agent.run(prompt))
    return result.output
