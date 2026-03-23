"""Web search tool with real API and demo fallback modes."""

from __future__ import annotations

import httpx

# Demo responses for common query patterns (used when no API key is set).
_DEMO_RESPONSES: dict[str, str] = {
    "gdp": (
        "[Demo Mode] Search results for GDP query:\n"
        "1. World Bank Data - Global GDP figures for 2024\n"
        "2. IMF World Economic Outlook - GDP by country\n"
        "3. Wikipedia - Gross Domestic Product overview"
    ),
    "population": (
        "[Demo Mode] Search results for population query:\n"
        "1. UN Population Division - World Population Prospects 2024\n"
        "2. Census Bureau - Current population estimates\n"
        "3. Worldometer - Real-time population statistics"
    ),
}

_DEFAULT_DEMO = (
    "[Demo Mode] Search results:\n"
    "1. Wikipedia - Overview of the topic\n"
    "2. Academic sources - Research papers and studies\n"
    "3. News - Recent articles and reports\n"
    "\nNote: Set SEARCH_API_KEY for real web search results."
)


async def search(query: str, *, api_key: str | None = None) -> str:
    """Search for information on the web.

    When a non-empty SEARCH_API_KEY is provided, makes a real API call to
    DuckDuckGo Instant Answer API. Otherwise, returns demo/placeholder results.

    Args:
        query: The search query string.
        api_key: Optional API key for real search. If empty or None, uses demo mode.

    Returns:
        Formatted search results as a string.
    """
    if not api_key:
        return _demo_search(query)
    return await _real_search(query, api_key=api_key)


def _demo_search(query: str) -> str:
    """Return demo search results based on query keywords.

    Args:
        query: The search query string.

    Returns:
        A demo response string matching the query topic.
    """
    query_lower = query.lower()
    for keyword, response in _DEMO_RESPONSES.items():
        if keyword in query_lower:
            return response
    return _DEFAULT_DEMO


async def _real_search(query: str, *, api_key: str) -> str:
    """Perform a real web search using DuckDuckGo Instant Answer API.

    The ``api_key`` parameter is accepted for interface consistency and
    future authentication support; DuckDuckGo's public API currently does
    not require a key.

    Args:
        query: The search query string.
        api_key: The API key supplied by the caller (reserved for future use).

    Returns:
        Formatted search results or an error message prefixed with "Search error:".
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[str] = []

        # Abstract (main answer)
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractSource"):
                results.append(f"Source: {data['AbstractSource']}")

        # Related topics (up to 3)
        for i, topic in enumerate(data.get("RelatedTopics", [])[:3], 1):
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"{i}. {topic['Text']}")

        if not results:
            return f"No detailed results found for: {query}"

        return "\n".join(results)

    except httpx.HTTPError as exc:
        return f"Search error: {exc}"
