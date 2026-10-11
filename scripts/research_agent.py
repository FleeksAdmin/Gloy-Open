"""Research agent: gather from open sources, classify reliability, build consensus."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from continuous_learner import Fact, KnowledgeStore, LOGS_DIR
from open_sources import OpenSources
from source_reliability import rank

RESEARCH_DIR = LOGS_DIR / "research"
KB = KnowledgeStore(LOGS_DIR / "continuous" / "knowledge.jsonl")


async def run(query: str, per_source: int = 2) -> dict[str, object]:
    docs = await OpenSources().search(query, per_source=per_source)
    ranked = rank(docs)
    trusted = [r for r in ranked if r["verdict"].tier in ("official", "authoritative", "media")]
    consensus_texts = [f"[{r['doc'].source}] {r['doc'].title}: {r['doc'].text[:1200]}" for r in trusted[:4]]
    consensus_urls = [r["url"] for r in trusted if r["url"]]
    if consensus_texts:
        KB.add(
            Fact(
                text=consensus_texts[0][:1500],
                source=f"research:{trusted[0]['verdict'].tier}",
                confidence=0.9,
                verified=True,
                topic=query[:120],
                urls=consensus_urls,
            )
        )
    result = {
        "query": query,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "docs_checked": len(docs),
        "trusted": [
            {"url": r["url"], "tier": r["verdict"].tier, "label": r["verdict"].label,
             "title": r["doc"].title}
            for r in trusted
        ],
        "distrusted": [
            {"url": r["url"], "tier": r["verdict"].tier, "label": r["verdict"].label}
            for r in ranked if r["verdict"].tier not in ("official", "authoritative", "media")
        ],
        "consensus_texts": consensus_texts,
        "consensus_urls": consensus_urls,
        "knowledge_added": len(consensus_texts) > 0,
    }
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    with (RESEARCH_DIR / f"{ts}.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="research agent")
    parser.add_argument("query")
    parser.add_argument("--per-source", type=int, default=2)
    args = parser.parse_args(argv)
    result = asyncio.run(run(args.query, args.per_source))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())