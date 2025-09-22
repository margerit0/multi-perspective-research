from __future__ import annotations

import os
from typing import Literal, List
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from langchain_google_community import GoogleSearchAPIWrapper

mcp = FastMCP(
    "WebSearch",
    instructions="Aggregate web search across Tavily and Google CSE, return structured results."
)

class SearchResult(BaseModel):
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Canonical URL")
    snippet: str = Field("", description="Short description or snippet")
    source: Literal["tavily", "google"] = Field(..., description="Which engine produced this result")
    position: int = Field(..., description="1-based rank within its engine")

class SearchResponse(BaseModel):
    query: str
    used_engines: List[str]
    results: List[SearchResult]

def _tavily_search(query: str, k: int) -> List[SearchResult]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    tvly = TavilyClient(api_key=api_key)
    resp = tvly.search(
        query=query,
        max_results=k,
        include_answer=False,
        include_raw_content=False,
    )
    items = (resp or {}).get("results", []) or []
    out: List[SearchResult] = []
    for i, it in enumerate(items, 1):
        out.append(SearchResult(
            title=it.get("title") or it.get("url") or "Untitled",
            url=it.get("url") or "",
            snippet=it.get("content") or it.get("snippet") or "",
            source="tavily",
            position=i,
        ))
    return out

def _google_search(query: str, k: int) -> List[SearchResult]:
    api_key = os.getenv("GOOGLE_SEARCH_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not (api_key and cse_id):
        return []
    g = GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cse_id)
    rows = g.results(query, num_results=k)
    out: List[SearchResult] = []
    for i, r in enumerate(rows, 1):
        out.append(SearchResult(
            title=r.get("title") or r.get("link") or "Untitled",
            url=r.get("link") or "",
            snippet=r.get("snippet") or "",
            source="google",
            position=i,
        ))
    return out

@mcp.tool()
def web_search(
    query: str,
    k: int = 5,
    engines: List[Literal["tavily", "google"]] = ["tavily", "google"],
) -> SearchResponse:
    engines = [e for e in engines if e in ("tavily", "google")]
    results: List[SearchResult] = []
    if "tavily" in engines:
        results.extend(_tavily_search(query, k))
    if "google" in engines:
        results.extend(_google_search(query, k))

    # 去重：按 URL 先来先保
    seen, deduped = set(), []
    for r in results:
        if r.url and r.url not in seen:
            deduped.append(r)
            seen.add(r.url)

    return SearchResponse(query=query, used_engines=engines, results=deduped)

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = 8765
    mcp.settings.streamable_http_path = "/mcp"
    mcp.run(transport="streamable-http")