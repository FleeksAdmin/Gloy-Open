"""Live-learn session: pause 24/7 loop, research, answer, save secret-pattern, resume."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
STATE_DIR = LOGS_DIR / "continuous"
PAUSE_FILE = STATE_DIR / "pause.flag"
DATASET_FILE = LOGS_DIR / "lora_dataset.jsonl"
RUNS_FILE = STATE_DIR / "runs.jsonl"
SUCCESS_FILE = LOGS_DIR / "success.jsonl"
TRAINING_FILE = LOGS_DIR / "training_queue.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def pause_user() -> None:
    PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_FILE.touch(exist_ok=True)


def resume_user() -> None:
    try:
        PAUSE_FILE.unlink(missing_ok=True)
    except OSError:
        pass


@asynccontextmanager
async def user_turn() -> AsyncIterator[None]:
    pause_user()
    try:
        yield
    finally:
        resume_user()


def format_secret_output(secret: str, thinking: str, answer: str) -> str:
    return (
        f"<secret>{(secret or '').strip()[:2000]}</secret>\n"
        f"<thinking>{(thinking or '').strip()[:1200]}</thinking>\n"
        f"<answer>{(answer or '').strip()[:1500]}</answer>"
    )


def save_example(
    prompt: str,
    answer: str,
    *,
    secret: str = "",
    thinking: str = "",
    sources: list[str] | None = None,
    status: str = "acceptable",
    score: float | None = None,
    origin: str = "user",
) -> dict[str, Any]:
    prompt = (prompt or "").strip()
    answer = (answer or "").strip()
    if len(prompt) < 2 or len(answer) < 1:
        return {"saved": False}
    thinking_text = thinking or ""
    if isinstance(thinking, list):
        thinking_text = "\n".join(str(item) for item in thinking)
    row = {
        "instruction": prompt,
        "output": format_secret_output(secret, thinking_text, answer),
        "prompt": prompt,
        "answer": answer,
        "scratchpad": secret,
        "thinking": thinking_text,
        "sources": sources or [],
        "status": status,
        "score": score,
        "origin": origin,
        "time": _now(),
    }
    _append(DATASET_FILE, row)
    _append(
        RUNS_FILE,
        {
            "run_id": f"{origin}-{_now()}",
            "prompt": prompt,
            "answer": answer,
            "status": status,
            "score": score,
            "scratchpad": secret,
            "thinking": thinking_text,
            "sources": sources or [],
            "time": _now(),
        },
    )
    if status in ("perfect", "acceptable"):
        _append(SUCCESS_FILE, {"prompt": prompt, "answer": answer, "time": _now()})
    else:
        _append(TRAINING_FILE, {"prompt": prompt, "flaws": ["user_or_judge_failed"], "time": _now()})
    return {"saved": True, "dataset": str(DATASET_FILE)}


def needs_web(prompt: str) -> bool:
    text = (prompt or "").lower()
    forced = (
        "найди", "погугли", "search", "в интернете", "look up", "latest",
        "новост", "current event", "сегодня в",
    )
    factual = (
        "who is ", "who was ", "what is ", "what's ", "кто так", "что такое",
        "when did", "where is", "столиц", "capital of",
    )
    return any(token in text for token in forced + factual)


async def research_if_needed(query: str, per_source: int = 1) -> dict[str, Any]:
    if not needs_web(query):
        return {"digest": "", "urls": [], "n": 0, "skipped": True}
    found = await research(query, per_source=per_source)
    found["skipped"] = False
    return found


async def research(query: str, per_source: int = 1) -> dict[str, Any]:
    texts: list[str] = []
    urls: list[str] = []
    try:
        from open_sources import OpenSources
        from source_reliability import rank

        docs = await OpenSources().search(query, per_source=per_source)
        for item in rank(docs):
            verdict = item["verdict"]
            doc = item["doc"]
            if verdict.tier in ("official", "authoritative", "media", "community"):
                texts.append(f"[{doc.source}:{verdict.tier}] {doc.title}: {doc.text[:400]}")
                if doc.url:
                    urls.append(doc.url)
    except Exception:
        pass
    try:
        from ddgs import DDGS

        with DDGS() as client:
            for hit in client.text(query, max_results=3) or []:
                body = (hit.get("body") or "").strip()
                href = hit.get("href") or ""
                if body:
                    texts.append(body[:400])
                if href:
                    urls.append(href)
    except Exception:
        pass
    digest = "\n".join(texts[:6])
    return {"digest": digest, "urls": urls[:8], "n": len(texts)}
