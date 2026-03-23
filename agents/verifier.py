"""Verification agent that audits reasoning chains for correctness."""

from __future__ import annotations

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
    result = await agent.run(prompt)
    return result.output
