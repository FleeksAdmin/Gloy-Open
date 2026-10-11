"""Gloy Agent — browser voice agent: Whisper STT, TTS, research + LLM actions."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse

SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_HTML = SCRIPT_DIR / "agent.html"
LOGS_DIR = SCRIPT_DIR.parent / "logs"

app = FastAPI(title="Gloy Agent")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
ENDPOINT = os.environ.get("GLOW_ENDPOINT", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("GLOW_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")

AGENT_SYSTEM = (
    "You are Gloy, a general autonomous agent. Use evidence, reason carefully, "
    "admit uncertainty, and answer in the user's language. If a task needs "
    "research, use the provided research digest; if it needs code, provide "
    "executable Python."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _have_llm() -> bool:
    try:
        response = httpx.get(ENDPOINT.replace("/chat/completions", "/models"), timeout=4)
        return response.status_code == 200
    except Exception:
        return False


async def _llm_chat(messages: list[dict[str, str]]) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            ENDPOINT,
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 1024,
                "extra_body": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@app.get("/", response_class=HTMLResponse)
def index():
    if AGENT_HTML.exists():
        return AGENT_HTML.read_text(encoding="utf-8")
    return HTMLResponse("<h1>agent.html missing</h1>")


@app.post("/api/research")
async def research(request: Request):
    body = await request.json()
    query = str(body.get("query", "")).strip()
    if not query:
        return JSONResponse({"error": "empty query"}, status_code=400)
    try:
        from open_sources import OpenSources
        from source_reliability import rank

        docs = await OpenSources().search(query, per_source=1)
        ranked = rank(docs)
        trusted = [r for r in ranked if r["verdict"].tier in ("official", "authoritative", "media")]
        digest = "\n".join(
            f"[{r['doc'].source}:{r['verdict'].tier}] {r['doc'].title}: {r['doc'].text[:900]}"
            for r in trusted[:4]
        ) or "no trusted open-source evidence found"
        return {
            "digest": digest,
            "sources": [{"url": r["url"], "tier": r["verdict"].tier} for r in trusted[:6]],
            "checked": len(docs),
            "time": _now(),
        }
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


@app.post("/api/act")
async def act(request: Request):
    body = await request.json()
    prompt = str(body.get("prompt", "")).strip()
    research_digest = str(body.get("research_digest", ""))
    if not prompt:
        return JSONResponse({"error": "empty prompt"}, status_code=400)
    if not _have_llm():
        return JSONResponse(
            {"error": "no LLM endpoint. Start vLLM or set GLOW_ENDPOINT (the 4090 tunnel URL)."},
            status_code=503,
        )
    user = prompt
    if research_digest:
        user = f"Research:\n{research_digest}\n\nTask: {prompt}"
    tool_mode = str(body.get("mode", "agent")).lower()
    effort = str(body.get("effort", "low")).lower()
    if effort != "low" or tool_mode == "deepthink":
        try:
            from deepthink import DeepThink, effort as _effort

            async def _talk(messages, **kwargs):
                return await _llm_chat(list(messages))

            result = await DeepThink(_talk, evaluator=None).solve(user, effort if effort != "low" else "high")
            return {"answer": result["answer"] or _llm_text(user), "passes": result["passes"],
                    "effort": result["effort"], "time": _now()}
        except Exception as exc:
            return JSONResponse({"error": f"deepthink: {type(exc).__name__}: {exc}"}, status_code=502)
    try:
        answer = await _llm_chat(
            [
                {"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": user},
            ]
        )
        return {"answer": answer, "effort": "low", "time": _now()}
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)


async def _llm_text(prompt: str) -> str:
    return await _llm_chat([{"role": "user", "content": prompt}])


@app.post("/api/stt")
async def stt(file: UploadFile):
    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty audio"}, status_code=400)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return JSONResponse(
            {"error": "whisper needs ffmpeg to decode audio; install ffmpeg or use the browser's native STT"},
            status_code=501,
        )
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    except Exception as exc:
        return JSONResponse({"error": f"whisper load failed: {exc}"}, status_code=500)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as handle:
        handle.write(data)
        name = handle.name
    wav = name.replace(".webm", ".wav")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", name, "-ar", "16000", "-ac", "1", wav],
            capture_output=True,
            check=True,
        )
        segments, _info = model.transcribe(wav, language=None)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        text = ""
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for path in (name, wav):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
    if not text:
        return JSONResponse({"error": error}, status_code=500)
    return {"text": text, "model": WHISPER_MODEL}


@app.post("/api/tts")
async def tts(request: Request):
    body = await request.json()
    text = str(body.get("text", "")).strip()
    voice = str(body.get("voice", "ru-RU-DmitryNeural"))
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return Response(b"".join(chunks), media_type="audio/mpeg")
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)