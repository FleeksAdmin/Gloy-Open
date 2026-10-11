"""Gloy 4.0 debugging and monitoring dashboard (FastAPI)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from orchestrator import (
    BUFFER_FILE,
    SUCCESS_FILE,
    TRAINING_FILE,
    Buffer,
    Config,
    Evaluation,
    Evaluator,
    Status,
    StreamAnomalyDetector,
    VLLMClient,
    _read_jsonl,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = SCRIPT_DIR / "dashboard.html"
PAUSE_FLAG = SCRIPT_DIR.parent / "logs" / "continuous" / "pause.flag"

_client_class = VLLMClient


class GenerateRequest(BaseModel):
    prompt: str
    temperatures: list[float] = Field(default_factory=lambda: [0.7])
    stream: bool = True
    mode: str = "normal"


app = FastAPI(title="Gloy AI Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_config(temperature: float) -> Config:
    return Config(temperature=temperature)


def _make_client(config: Config):
    return _client_class(config)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _record(
    config: Config,
    prompt: str,
    text: str,
    evaluation,
    anomaly,
    temperature: float,
    tokens: int,
    elapsed: float,
    judge_elapsed: float | None,
) -> dict[str, Any]:
    record = {
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "temperature": temperature,
        "attempt": 0,
        "prompt": prompt,
        "answer": text,
        "status": evaluation.status.value,
        "score": round(evaluation.score, 3),
        "flaws": list(evaluation.flaws),
        "critique_prompt": evaluation.critique_prompt,
        "anomaly": anomaly.to_dict() if anomaly else None,
        "tokens": tokens,
        "elapsed": round(elapsed, 3),
        "judge_elapsed": round(judge_elapsed, 3) if judge_elapsed is not None else None,
    }
    try:
        from learn_session import save_example

        save_example(
            prompt,
            text,
            status=evaluation.status.value,
            score=round(evaluation.score, 3),
            origin="dashboard",
        )
    except Exception:
        pass
    return record


async def _stream_temperature(
    prompt: str,
    temperature: float,
    cfg: Config,
) -> AsyncIterator[str]:
    client = _make_client(cfg)
    evaluator = Evaluator(client, cfg)
    messages = [{"role": "user", "content": prompt}]
    detector = StreamAnomalyDetector(cfg)
    yield _sse({"type": "temp_start", "temperature": temperature})
    started = time.perf_counter()
    first_at: float | None = None
    tokens = 0
    chunks: list[str] = []
    anomaly = None
    try:
        async for chunk in client.stream_chat(
            messages,
            temperature=temperature,
            max_tokens=cfg.max_tokens,
        ):
            if first_at is None:
                first_at = time.perf_counter()
            tokens += 1
            chunks.append(chunk)
            anomaly = detector.update(chunk)
            now = time.perf_counter() - started
            ttft_ms = (first_at - started) * 1000
            decode_s = max(1e-6, now - (first_at - started))
            tps = tokens / decode_s
            yield _sse(
                {
                    "type": "token",
                    "temperature": temperature,
                    "token": chunk,
                    "ttft_ms": round(ttft_ms, 1),
                    "tps": round(tps, 1),
                    "latency_ms": round(now * 1000, 1),
                    "tokens": tokens,
                }
            )
            if anomaly is not None:
                break
    except Exception as exc:
        yield _sse(
            {
                "type": "error",
                "temperature": temperature,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return
    text = "".join(chunks)
    if anomaly is None:
        anomaly = detector.finalize(text)
    latency = time.perf_counter() - started
    ttft_ms = (first_at - started) * 1000 if first_at else latency * 1000
    decode_s = max(1e-6, latency - ((first_at - started) if first_at else 0))
    tps = tokens / decode_s
    yield _sse(
        {
            "type": "generated",
            "temperature": temperature,
            "answer": text,
            "tokens": tokens,
            "ttft_ms": round(ttft_ms, 1),
            "tps": round(tps, 1),
            "latency_ms": round(latency * 1000, 1),
            "anomaly": anomaly.to_dict() if anomaly else None,
        }
    )
    judge_elapsed = None
    if anomaly is not None:
        evaluation = Evaluation(
            score=0.0,
            status=Status.FAILED,
            flaws=[f"stream anomaly: {anomaly.kind} ({anomaly.detail})"],
            critique_prompt="Regenerate and avoid repetition.",
        )
    else:
        judge_start = time.perf_counter()
        evaluation = await evaluator.evaluate(prompt, text)
        judge_elapsed = time.perf_counter() - judge_start
    record = _record(
        cfg, prompt, text, evaluation, anomaly, temperature, tokens, latency, judge_elapsed
    )
    await Buffer(BUFFER_FILE).append(record)
    if evaluation.status == Status.PERFECT:
        await Buffer(SUCCESS_FILE).append(record)
    else:
        await Buffer(TRAINING_FILE).append(record)
    yield _sse(
        {
            "type": "done",
            "temperature": temperature,
            "evaluation": evaluation.to_dict(),
            "anomaly": anomaly.to_dict() if anomaly else None,
            "answer": text,
            "ttft_ms": round(ttft_ms, 1),
            "tps": round(tps, 1),
            "latency_ms": round(latency * 1000, 1),
            "tokens": tokens,
            "judge_elapsed_ms": round(judge_elapsed * 1000, 1)
            if judge_elapsed is not None
            else None,
        }
    )


@app.get("/")
def index() -> FileResponse:
    if not DASHBOARD_HTML.exists():
        raise HTTPException(status_code=500, detail="dashboard.html not found")
    return FileResponse(DASHBOARD_HTML)


@app.post("/api/generate")
async def generate_api(req: GenerateRequest):
    if not req.temperatures:
        raise HTTPException(status_code=400, detail="at least one temperature required")
    PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_FLAG.touch(exist_ok=True)
    try:
        return await _do_generate(req)
    finally:
        PAUSE_FLAG.unlink(missing_ok=True)


async def _do_generate(req: GenerateRequest):
    if not req.temperatures:
        raise HTTPException(status_code=400, detail="at least one temperature required")
    if req.stream:
        temps = req.temperatures or [0.7]
        cfg = _make_config(temps[0])

        async def sse_stream() -> AsyncIterator[str]:
            for temperature in temps:
                cfg = _make_config(temperature)
                async for event in _stream_temperature(req.prompt, temperature, cfg):
                    yield event

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    temperatures = req.temperatures or (
        [0.2, 0.7, 1.1] if req.mode == "gloy_code" else [0.7]
    )
    client = _make_client(_make_config(temperatures[0]))
    orchestrator = _build_orchestrator(client, _make_config(temperatures[0]))
    result = await orchestrator.run_best_of_n(
        req.prompt, temperatures=temperatures, mode=req.mode
    )
    return result.to_dict()


def _build_orchestrator(client, cfg):
    from orchestrator import Orchestrator

    return Orchestrator(client, cfg)


@app.get("/api/logs/{log_type}")
def logs(log_type: str):
    files = {
        "buffer": BUFFER_FILE,
        "training_queue": TRAINING_FILE,
        "success": SUCCESS_FILE,
    }
    if log_type not in files:
        raise HTTPException(status_code=404, detail="unknown log type")
    path = files[log_type]
    records = _read_jsonl(path)
    records.reverse()
    return {"type": log_type, "records": records}


@app.get("/api/learner")
def learner_metrics():
    continuous_dir = SCRIPT_DIR.parent / "logs" / "continuous"
    state = {}
    checkpoint_file = continuous_dir / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            state = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}

    def tail_jsonl(path, limit):
        if not path.exists():
            return []
        rows = _read_jsonl(path)
        items = [dict(item) for item in rows]
        items.reverse()
        return items[:limit]

    all_runs = _read_jsonl(continuous_dir / "runs.jsonl")
    all_runs.reverse()
    session_file = continuous_dir / "session.json"
    session_started = ""
    if session_file.exists():
        try:
            session_started = str(
                json.loads(session_file.read_text(encoding="utf-8")).get(
                    "started_at", ""
                )
            )
        except (json.JSONDecodeError, OSError):
            session_started = ""
    runs = [
        item for item in all_runs if not session_started or item.get("time", "") >= session_started
    ][:15]
    scratch = tail_jsonl(continuous_dir / "scratch_steps.jsonl", 10)
    hypotheses = tail_jsonl(continuous_dir / "hypotheses.jsonl", 8)
    cot = tail_jsonl(continuous_dir / "cot.log", 6)
    experiments = tail_jsonl(continuous_dir / "experiments.jsonl", 5)
    knowledge = tail_jsonl(continuous_dir / "knowledge.jsonl", 4)
    anomalies = [
        item
        for item in _read_jsonl(SCRIPT_DIR.parent / "logs" / "anomalies.jsonl")
        if item.get("event") == "learner_tick_error"
        or item.get("event") == "lora_failed"
    ]
    anomalies.reverse()
    anomalies = anomalies[:6]
    latest = runs[0] if runs else {}
    latest_experiment = experiments[0] if experiments else {}
    latest_scratch = scratch[0] if scratch else {}
    latest_knowledge = knowledge[0] if knowledge else {}
    pending = (state.get("state") or {}).get("pending", [])
    chain = [
        {
            "id": "queue",
            "title": "Queue",
            "status": "active" if pending else "done",
            "detail": f"{len(pending)} pending topics",
        },
        {
            "id": "scratchpad",
            "title": "Private scratchpad",
            "status": "done" if latest_scratch or latest.get("scratchpad") else "waiting",
            "detail": f"{latest.get('steps', 0)} steps on latest run",
        },
        {
            "id": "web_rag",
            "title": "Web / RAG",
            "status": "done" if latest_knowledge else "waiting",
            "detail": latest_knowledge.get("source", "awaiting verified consensus"),
        },
        {
            "id": "sandbox",
            "title": "Sandbox",
            "status": "done" if latest_experiment else "idle",
            "detail": latest_experiment.get("stdout", "no experiment in latest run"),
        },
        {
            "id": "evaluator",
            "title": "Evaluator T=0.0",
            "status": "done" if latest.get("score") is not None else "waiting",
            "detail": f"score {latest.get('score', '—')} · factuality {latest.get('factuality', '—')}",
        },
        {
            "id": "result",
            "title": "Result",
            "status": latest.get("status", "waiting"),
            "detail": latest.get("prompt", "no completed run"),
        },
    ]
    return {
        "state": state,
        "runs": runs,
        "hypotheses": hypotheses,
        "history_count": max(0, len(all_runs) - len(runs)),
        "scratch": scratch,
        "cot": cot,
        "experiments": experiments,
        "knowledge": knowledge,
        "anomalies": anomalies,
        "chain": chain,
    }


@app.post("/api/lora/run")
def lora_run():
    LORA_FLAG = PAUSE_FLAG.parent / "lora.flag"
    LORA_FLAG.touch(exist_ok=True)
    return {"requested": True, "note": "learner will run LoRA kick on next tick"}


@app.get("/api/bench")
def bench_history():
    bench_dir = SCRIPT_DIR.parent / "logs" / "bench"
    history_file = bench_dir / "solved_history.jsonl"
    points = _read_jsonl(history_file)
    return {
        "points": [
            {
                "ran_at": p.get("ran_at", ""),
                "solved": p.get("solved", 0),
                "total": p.get("total", 0),
            }
            for p in points
        ]
    }


@app.get("/api/metrics")
def metrics():
    records = _read_jsonl(BUFFER_FILE)
    timeline = [r for r in records if r.get("score") is not None]
    timeline = timeline[-30:]
    status_counts = Counter(r.get("status", "unknown") for r in records)
    temp_scores: dict[float, list[float]] = defaultdict(list)
    for r in records:
        temp = r.get("temperature")
        score = r.get("score")
        if temp is None or score is None:
            continue
        temp_scores[temp].append(score)
    gen_latencies = [r["elapsed"] for r in records if r.get("elapsed") is not None]
    judge_latencies = [
        r["judge_elapsed"]
        for r in records
        if r.get("judge_elapsed") is not None
    ]
    return {
        "timeline": [
            {
                "x": r.get("timestamp", ""),
                "score": float(r["score"]),
                "status": r.get("status", ""),
            }
            for r in timeline
        ],
        "temp_vs_score": [
            {
                "temperature": temp,
                "avg_score": round(sum(scores) / len(scores), 3),
                "count": len(scores),
            }
            for temp, scores in sorted(temp_scores.items())
        ],
        "status_counts": {
            "perfect": status_counts.get("perfect", 0),
            "acceptable": status_counts.get("acceptable", 0),
            "failed": status_counts.get("failed", 0),
        },
        "latency": {
            "avg_generation_ms": round(
                (sum(gen_latencies) / len(gen_latencies) * 1000)
                if gen_latencies
                else 0.0,
                1,
            ),
            "avg_judge_ms": round(
                (sum(judge_latencies) / len(judge_latencies) * 1000)
                if judge_latencies
                else 0.0,
                1,
            ),
            "runs": len(records),
        },
    }


def main(argv=None):
    global _client_class
    parser = argparse.ArgumentParser(description="Gloy AI dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mock", action="store_true", help="use mock LLM client")
    args = parser.parse_args(argv)
    if args.mock:
        from orchestrator import MockClient

        _client_class = MockClient

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
