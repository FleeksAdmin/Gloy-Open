"""Open-source retrieval: Wikipedia, arXiv, GitHub, Stack Exchange, DDG fallback."""

from __future__ import annotations

import asyncio
import gzip
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
ARXIV_API = "http://export.arxiv.org/api/query"
GITHUB_API = "https://api.github.com/search/repositories"
STACK_API = "https://api.stackexchange.com/2.3/search/advanced"


@dataclass(slots=True)
class Document:
    source: str
    title: str
    url: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "text": self.text,
        }


class OpenSources:
    def __init__(self, timeout: float = 20.0):
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            headers={"User-Agent": "GloyOpen/1.0 (+https://github.com)"},
            follow_redirects=True,
        )

    async def wikipedia(self, query: str, limit: int = 2) -> list[Document]:
        docs: list[Document] = []
        try:
            async with self._client() as client:
                search = await client.get(
                    WIKI_API,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json",
                        "srlimit": limit,
                    },
                )
                hits = search.json().get("query", {}).get("search", [])
                for hit in hits[:limit]:
                    title = hit.get("title", "")
                    summary = await client.get(WIKI_SUMMARY + title.replace(" ", "_"))
                    extract = summary.json().get("extract", "")
                    if extract:
                        docs.append(
                            Document(
                                source="wikipedia",
                                title=title,
                                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                text=extract[:1200],
                            )
                        )
        except Exception:
            return docs
        return docs

    async def arxiv(self, query: str, limit: int = 2) -> list[Document]:
        try:
            async with self._client() as client:
                response = await client.get(
                    ARXIV_API,
                    params={
                        "search_query": f"all:{query}",
                        "start": 0,
                        "max_results": limit,
                    },
                )
                body = response.text
        except Exception:
            return []
        docs: list[Document] = []
        for entry in re.findall(r"<entry>(.*?)</entry>", body, re.S)[:limit]:
            title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", entry, re.S).group(1)).strip() if "<title>" in entry else ""
            summary = re.sub(r"\s+", " ", re.search(r"<summary>(.*?)</summary>", entry, re.S).group(1)).strip() if "<summary>" in entry else ""
            link = re.search(r"<id>(.*?)</id>", entry, re.S)
            if title:
                docs.append(
                    Document(
                        source="arxiv",
                        title=title,
                        url=link.group(1).strip() if link else "",
                        text=summary[:1200],
                    )
                )
        return docs

    async def github(self, query: str, limit: int = 2) -> list[Document]:
        try:
            async with self._client() as client:
                response = await client.get(
                    GITHUB_API,
                    params={"q": query, "sort": "stars", "order": "desc", "per_page": limit},
                )
                items = response.json().get("items", [])
        except Exception:
            return []
        docs = []
        for item in items[:limit]:
            docs.append(
                Document(
                    source="github",
                    title=item.get("full_name", ""),
                    url=item.get("html_url", ""),
                    text=(item.get("description") or "")[:600],
                )
            )
        return docs

    async def stackexchange(self, query: str, limit: int = 2) -> list[Document]:
        try:
            async with self._client() as client:
                response = await client.get(
                    STACK_API,
                    params={
                        "order": "desc",
                        "sort": "relevance",
                        "q": query,
                        "site": "stackoverflow",
                        "pagesize": limit,
                        "filter": "withbody",
                    },
                )
                raw = response.content
                if response.headers.get("content-encoding") == "gzip":
                    raw = gzip.decompress(raw)
                data = json.loads(raw)
                items = data.get("items", [])
        except Exception:
            return []
        docs = []
        for item in items[:limit]:
            body = re.sub(r"<[^>]+>", " ", item.get("body", ""))
            docs.append(
                Document(
                    source="stackexchange",
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    text=re.sub(r"\s+", " ", body)[:1200],
                )
            )
        return docs

    async def search(self, query: str, per_source: int = 2) -> list[Document]:
        tasks = [
            self.wikipedia(query, per_source),
            self.arxiv(query, per_source),
            self.github(query, per_source),
            self.stackexchange(query, per_source),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        docs: list[Document] = []
        for group in results:
            if isinstance(group, list):
                docs.extend(group)
        return docs

    async def evidence(self, query: str, per_source: int = 2) -> tuple[list[str], list[str]]:
        docs = await self.search(query, per_source)
        texts = [f"[{d.source}] {d.title}: {d.text}" for d in docs if d.text]
        urls = [d.url for d in docs if d.url]
        return texts, urls