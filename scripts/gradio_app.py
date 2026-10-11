"""Gloy Agent — Gradio console with DeepThink (effort), secret/thinking, full debug."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any

import gradio as gr

from deepthink import DeepThink, EFFORT_LEVELS

ENDPOINT = os.environ.get("GLOW_ENDPOINT", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("GLOW_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")

JUDGE_SYSTEM = (
    "You are a strict evaluator. Respond with JSON: {\"score\":0..1,\"status\":"
    "\"perfect|acceptable|failed\",\"flaws\":[],\"critique_prompt\":\"\"} "
    "for the assistant answer."
)


def _have_llm() -> bool:
    try:
        import httpx

        response = httpx.get(ENDPOINT.replace("/chat/completions", "/models"), timeout=4)
        return response.status_code == 200
    except Exception:
        return False


class GradioLLM:
    def __init__(self):
        self.latency_ms = 0.0
        self.tokens_last = 0

    def _payload(self, messages, temperature, max_tokens, stream, extra_body=None):
        payload = {
            "model": MODEL,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if extra_body:
            payload.update(extra_body)
        return payload

    async def stream_chat(self, messages, *, temperature, max_tokens, extra_body=None):
        import httpx

        payload = self._payload(messages, temperature, max_tokens, True, extra_body)
        started = time.perf_counter()
        tokens = 0
        acc: list[str] = []
        timeout = httpx.Timeout(180, read=25.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", ENDPOINT, json=payload) as response:
                response.raise_for_status()
                lines = response.aiter_lines()
                while True:
                    try:
                        line = await asyncio.wait_for(lines.__anext__(), timeout=20.0)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        acc.append("\n[stream stalled — cut]")
                        yield "\n[stream stalled — cut]"
                        break
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = (
                        (delta.get("content") or "")
                        + (delta.get("reasoning") or "")
                        + (delta.get("reasoning_content") or "")
                    )
                    if piece:
                        tokens += 1
                        acc.append(piece)
                        yield piece
        elapsed = max(0.001, time.perf_counter() - started)
        self.latency_ms = round(elapsed * 1000, 1)
        self.tokens_last = tokens or max(1, len("".join(acc)) // 4)

    async def chat(self, messages, *, temperature, max_tokens, extra_body=None) -> str:
        parts: list[str] = []
        async for token in self.stream_chat(
            messages, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body
        ):
            parts.append(token)
        return "".join(parts)


class GradioJudge:
    def __init__(self, llm: GradioLLM):
        self._llm = llm

    def heuristic(self, answer: str):
        length = max(1, len(answer.split()))
        score = min(1.0, 0.35 + 0.04 * min(length, 14))
        return score, [], "check reasoning depth"

    async def __call__(self, prompt: str, answer: str):
        if not _have_llm():
            score, flaws, critique = self.heuristic(answer)
            return type("E", (), {"score": score, "critique_prompt": critique, "flaws": flaws})()
        try:
            raw = await self._llm.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": f"Task: {prompt}\nAnswer: {answer}"},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            score = float(data.get("score", 0.0))
            return type(
                "E", (), {
                    "score": score,
                    "critique_prompt": data.get("critique_prompt", ""),
                    "flaws": data.get("flaws", []),
                }
            )()
        except Exception:
            score, flaws, critique = self.heuristic(answer)
            return type("E", (), {"score": score, "critique_prompt": critique, "flaws": flaws})()


def is_smalltalk(prompt: str) -> bool:
    text = re.sub(r"[^\w\sа-яё]+", "", (prompt or "").lower(), flags=re.I).strip()
    if not text or len(text) > 40:
        return False
    greetings = (
        "привет", "прив", "хай", "ку", "йо", "здарова", "здравствуй",
        "здравствуйте", "добрый день", "доброе утро", "добрый вечер",
        "hi", "hello", "hey", "yo", "sup", "good morning", "good evening",
        "как дела", "как ты", "что как", "how are you", "whats up", "what's up",
    )
    return text in greetings or any(text == g or text.startswith(g + " ") for g in greetings)


async def _fast_hello(prompt: str, llm: GradioLLM) -> dict[str, Any]:
    raw = await llm.chat(
        [
            {
                "role": "system",
                "content": "Reply in one short friendly sentence. No lists, no tools, no essay.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=64,
    )
    answer = raw.split("</think>")[-1].strip() if "</think>" in raw else raw.strip()
    return {
        "answer": answer or "Привет.",
        "score": -1.0,
        "passes": 0,
        "effort": "fast",
        "scratchpad": "(skipped — smalltalk)",
        "thinking_steps": ["short reply, no DeepThink"],
        "secret_log": [],
        "fast_path": True,
    }


def _demo_secret(prompt: str, idx: int, summary: str) -> tuple[str, str]:
    digest = hashlib.md5(f"{prompt}|{idx}".encode()).hexdigest()[:6]
    notes = [
        "parse the goal, list constraints, choose approach",
        "verify against known facts, estimate confidence",
        "refine for precision, catch contradictions",
        "final pass: simplify, make executable and clear",
    ]
    draft = f"draft pass {idx + 1} for '{prompt[:40]}…' (hash {digest})"
    return notes[idx % len(notes)], draft


async def _live_solve(prompt: str, effort_name: str, llm: GradioLLM, judge: GradioJudge, goal: str = ""):
    from goals import add_goal, ingest_from_text
    from learn_session import research_if_needed, save_example, user_turn

    if is_smalltalk(prompt) and not (goal or "").strip():
        return await _fast_hello(prompt, llm)

    stored = add_goal(goal, source="gradio") if (goal or "").strip() else None
    task = prompt
    if stored:
        task = (
            f"Long-term goal: {stored['text']}\n"
            f"User prompt: {prompt}\n"
            "Decompose if needed, do the next executable slice, do not dump a fake 1000-step plan."
        )
    async with user_turn():
        found = await research_if_needed(task)
        output = await DeepThink(llm, judge, max_tokens=1024).solve(
            task, effort_name, web_context=found.get("digest", "")
        )
        score = output.get("score", -1.0)
        status = "perfect" if score >= 0.9 else "acceptable" if score >= 0.75 else "failed"
        if score < 0:
            status = "acceptable"
        save_example(
            prompt,
            output.get("answer") or "",
            secret=output.get("scratchpad") or "",
            thinking="\n".join(output.get("thinking_steps") or []),
            sources=found.get("urls") or [],
            status=status,
            score=None if score < 0 else score,
            origin="gradio",
        )
        if stored:
            extra_steps = ingest_from_text(
                (output.get("scratchpad") or "") + "\n" + "\n".join(output.get("thinking_steps") or []),
                stored["id"],
            )
            output["goal_id"] = stored["id"]
            output["goal_steps_added"] = extra_steps
        output["web_urls"] = found.get("urls") or []
        output["web_n"] = found.get("n", 0)
        return output


def _pack(thinking, secret, answer, score, tps, effort_name, debug):
    return [thinking, secret, answer, score, str(tps), effort_name, debug]


async def run(prompt: str, effort_name: str, goal: str = ""):
    if not prompt.strip() and not (goal or "").strip():
        yield _pack("", "", "empty prompt", "0 tok/s", "low", effort_name, "{}")
        return
    live = _have_llm()
    llm = GradioLLM()
    judge = GradioJudge(llm)
    if not live:
        yield _pack(
            "offline",
            "",
            f"demo answer for '{(prompt or '')[:60]}…'",
            "demo",
            "n/a (no LLM)",
            effort_name,
            '{"mode":"demo"}',
        )
        return

    thinking = "_streaming…_"
    secret = ""
    answer = ""
    yield _pack(thinking, secret, "…", "…", "0", effort_name, "{}")

    if is_smalltalk(prompt) and not (goal or "").strip():
        thinking = "_fast path_"
        started = time.perf_counter()
        async for token in llm.stream_chat(
            [
                {
                    "role": "system",
                    "content": "Reply in one short friendly sentence. No lists, no tools, no essay.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=64,
        ):
            answer += token
            yield _pack(thinking, "(skipped — smalltalk)", answer, "…", "…", "fast", "{}")
        clean = answer.split("</think>")[-1].strip() if "</think>" in answer else answer.strip()
        tps = round(llm.tokens_last / max(0.001, llm.latency_ms / 1000), 1)
        yield _pack(thinking, "(skipped — smalltalk)", clean or "Привет.", "n/a", tps, "fast", "{}")
        return

    from goals import add_goal, ingest_from_text
    from learn_session import research_if_needed, save_example, user_turn

    stored = add_goal(goal, source="gradio") if (goal or "").strip() else None
    task = prompt
    if stored:
        task = (
            f"Long-term goal: {stored['text']}\n"
            f"User prompt: {prompt}\n"
            "Decompose if needed, do the next executable slice, do not dump a fake 1000-step plan."
        )

    box = {"thinking": "", "secret": "", "answer": "", "last": 0.0, "phase": "start"}
    queue: asyncio.Queue = asyncio.Queue()

    def snapshot(status="…"):
        tps = round(llm.tokens_last / max(0.001, (llm.latency_ms or 1) / 1000), 1)
        live = f"{box['phase']} · {len(box['secret'])}c secret · {len(box['answer'])}c out"
        think = f"_{live}_\n\n{box['thinking']}"
        secret_view = box["secret"][-1600:]
        if len(box["secret"]) > 1600:
            secret_view = "…\n" + secret_view
        return _pack(think, f"```\n{secret_view}\n```", box["answer"] or "…", status, tps, effort_name, "{}")

    async def on_event(kind: str, text: str):
        if kind == "status":
            box["phase"] = text
            if "public" in text:
                box["answer"] = ""
        elif kind == "secret":
            box["secret"] += text
        elif kind == "public":
            box["answer"] += text
        now = time.perf_counter()
        if kind == "status" or now - box["last"] >= 0.08:
            box["last"] = now
            await queue.put(snapshot("…"))

    async def _work():
        try:
            async with user_turn():
                found = await research_if_needed(task)
                if found.get("n"):
                    box["thinking"] = f"_web: {found.get('n')} hits_"
                    await queue.put(snapshot("…"))
                result = await DeepThink(llm, judge, max_tokens=1024).solve(
                    task, effort_name, web_context=found.get("digest", ""), on_event=on_event
                )
                result["web_urls"] = found.get("urls") or []
                result["web_n"] = found.get("n", 0)
                result["_found"] = found
                await queue.put(("done", result))
        except Exception as exc:
            await queue.put(
                (
                    "done",
                    {
                        "answer": f"error: {type(exc).__name__}: {exc}",
                        "score": -1.0,
                        "scratchpad": box["secret"],
                        "thinking_steps": [box["thinking"]],
                        "_found": {},
                    },
                )
            )

    worker = asyncio.create_task(_work())
    output = None
    while True:
        item = await queue.get()
        if isinstance(item, tuple) and item[0] == "done":
            output = item[1]
            break
        yield item
    await worker
    found = output.pop("_found", {})
    score_n = output.get("score", -1.0)
    status = "perfect" if score_n >= 0.9 else "acceptable" if score_n >= 0.75 else "failed"
    if score_n < 0:
        status = "acceptable"
    save_example(
        prompt,
        output.get("answer") or "",
        secret=output.get("scratchpad") or "",
        thinking="\n".join(output.get("thinking_steps") or []),
        sources=found.get("urls") or [],
        status=status,
        score=None if score_n < 0 else score_n,
        origin="gradio",
    )
    if stored:
        extra_steps = ingest_from_text(
            (output.get("scratchpad") or "") + "\n" + "\n".join(output.get("thinking_steps") or []),
            stored["id"],
        )
        output["goal_id"] = stored["id"]
        output["goal_steps_added"] = extra_steps

    steps = output.get("thinking_steps") or []
    thinking = "\n".join(
        f"**step {i + 1}**\n```\n{' '.join(s.split())}\n```" for i, s in enumerate(steps)
    ) or box["thinking"]
    secret = output.get("scratchpad") or box["secret"]
    answer = output.get("answer") or box["answer"] or "(none)"
    score = str(round(output["score"], 3)) if output["score"] >= 0 else "n/a"
    tps = round(llm.tokens_last / max(0.001, llm.latency_ms / 1000), 1)
    full = json.dumps(output, ensure_ascii=False, indent=2, default=str)
    yield _pack(thinking, secret, answer, score, tps, output.get("effort") or effort_name, full)


with gr.Blocks(title="Gloy Agent — DeepThink") as demo:
    have = _have_llm()
    gr.Markdown(
        f"# Gloy Agent · Agentic DeepThink\nModel: `{MODEL}` · LLM: "
        f"**{'LIVE' if have else 'OFFLINE (demo)'}**"
    )
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="ask anything…", lines=3, scale=3)
        effort = gr.Radio(["low", "medium", "high", "max"], label="Effort", value="medium")
    goal = gr.Textbox(
        label="Long-term goal (optional)",
        placeholder="e.g. open a restaurant — stored on disk, learner picks next microtask",
        lines=2,
    )
    run_btn = gr.Button("Run Agentic DeepThink", variant="primary")
    with gr.Accordion("Thinking · what the user sees (built from secret)", open=True):
        thinking = gr.Markdown("_waiting…_")
    with gr.Accordion("Secret · private mind first (user never needs this)", open=False):
        secret = gr.Markdown("_waiting…_")
    with gr.Accordion("Answer", open=True):
        answer = gr.Markdown("")
    with gr.Row():
        tok_per_s = gr.Textbox(label="Tokens / sec", interactive=False)
        score_out = gr.Textbox(label="Score", interactive=False)
        effort_out = gr.Textbox(label="Effort", interactive=False)
    with gr.Accordion("Full JSON debug"):
        debug = gr.Code(language="json")

    run_btn.click(
        run,
        [prompt, effort, goal],
        [thinking, secret, answer, score_out, tok_per_s, effort_out, debug],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)