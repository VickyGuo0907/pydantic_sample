"""Multi-step reasoning agent that breaks queries into structured steps."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from config.settings import Settings
from schemas.reasoning import ReasoningChain
from schemas.retrieval import RetrievalResult
from tools.calculator import calculate
from tools.retriever import VectorStore
from tools.search import search

logger = logging.getLogger(__name__)

REASONING_SYSTEM_PROMPT = """\
You are a multi-step reasoning agent. Given a user query, you must:

1. Break the problem into numbered reasoning steps.
2. For each step, explain what you're trying to accomplish, show your reasoning,
   and state a clear conclusion.
3. Use the search tool when you need factual information you don't confidently know.
4. Use the calculator tool for ANY mathematical computation — never do math in your head.
5. Each step should build on previous steps' conclusions.
6. After all steps, synthesize a final answer.
7. Assign a confidence score (0.0 to 1.0) based on the quality of evidence:
   - 0.9-1.0: Strong evidence from tools, straightforward logic
   - 0.7-0.8: Good evidence but some uncertainty
   - 0.5-0.6: Partial evidence, reasonable inference
   - Below 0.5: Significant uncertainty, limited evidence

Your response MUST be a JSON object with exactly these top-level keys:
  "query"        – the original user question (string)
  "steps"        – a JSON array (not a string) of reasoning step objects
  "final_answer" – your synthesized answer (string)
  "confidence"   – a float between 0.0 and 1.0

Do NOT wrap "steps" in quotes or encode it as a string. It must be a real JSON array.
"""


@dataclass
class ReasoningDeps:
    """Runtime dependencies for the reasoning agent."""

    search_api_key: str | None = None
    vector_store: VectorStore | None = None  # None = RAG disabled


def create_reasoning_agent(model: Model | str) -> Agent[ReasoningDeps, ReasoningChain]:
    """Create a reasoning agent with tools registered.

    Args:
        model: A pydantic-ai Model instance or model name string.

    Returns:
        A configured Agent that produces ReasoningChain output.
    """
    agent: Agent[ReasoningDeps, ReasoningChain] = Agent(
        model,
        output_type=ReasoningChain,
        system_prompt=REASONING_SYSTEM_PROMPT,
        deps_type=ReasoningDeps,
        retries=3,
    )

    @agent.tool
    async def search_tool(ctx: RunContext[ReasoningDeps], query: str) -> str:
        """Search the web for factual information relevant to the reasoning step.

        Args:
            ctx: Run context with dependencies.
            query: The search query.

        Returns:
            Search results as a formatted string.
        """
        return await search(query, api_key=ctx.deps.search_api_key)

    @agent.tool
    async def calculator_tool(
        _ctx: RunContext[ReasoningDeps], expression: str
    ) -> str:
        """Calculate a mathematical expression safely.

        Args:
            _ctx: Run context (unused; present for pydantic-ai tool signature).
            expression: A math expression like '2 + 3 * 4'.

        Returns:
            The result as a string, or an error message.
        """
        return calculate(expression)

    @agent.tool
    async def retrieve_tool(ctx: RunContext[ReasoningDeps], query: str) -> str:
        """Retrieve relevant context from the local document corpus.

        Use this tool FIRST for questions that may be answered by domain
        documents. It searches an indexed knowledge base and returns the
        most relevant passages. If the result is empty, fall back to search.

        Args:
            ctx: Run context with dependencies.
            query: The retrieval query (can be the user question or a sub-question).

        Returns:
            Formatted context passages, or a message indicating no results.
        """
        if ctx.deps.vector_store is None:
            return "RAG not enabled. Use the search tool instead."
        result: RetrievalResult = ctx.deps.vector_store.retrieve(query, top_k=3)
        return result.as_context_string()

    return agent


async def run_reasoning(
    query: str,
    settings: Settings,
    vector_store: VectorStore | None = None,
) -> ReasoningChain:
    """Run the full reasoning pipeline for a query.

    Args:
        query: The user's question to reason about.
        settings: Application settings with provider config.
        vector_store: Optional pre-loaded vector store for RAG. None disables RAG.

    Returns:
        A structured ReasoningChain with steps and final answer.
    """
    model = settings.get_model()
    agent = create_reasoning_agent(model)
    deps = ReasoningDeps(
        search_api_key=settings.search_api_key or None,
        vector_store=vector_store,
    )
    logger.debug("Running reasoning agent for query: %s", query)
    result = await agent.run(query, deps=deps)
    return result.output
