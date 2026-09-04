"""
Web search tool. Uses Tavily (https://tavily.com) because its API is built
for LLM agents specifically -- results come back as clean summarized
snippets rather than raw HTML, which means the researcher node doesn't need
its own scraping/cleaning logic. Free tier is enough for portfolio-scale use.
"""

from __future__ import annotations

import os
from tavily import TavilyClient

from src import config

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = config.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Get a free key at https://tavily.com "
                "and add it to your .env file."
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str, max_results: int = None) -> list[dict]:
    """
    Returns a list of {"title", "url", "content"} dicts.
    `content` is Tavily's own extracted/summarized snippet, not raw HTML.
    """
    max_results = max_results or config.WEB_SEARCH_RESULTS_PER_SUBQUESTION
    client = _get_client()
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]
