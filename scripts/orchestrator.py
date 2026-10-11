"""Async self-checking generation pipeline for Gloy AGI."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, Sequence

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
RUNS_DIR = LOGS_DIR / "runs"
BUFFER_FILE = LOGS_DIR / "buffer.jsonl"
SUCCESS_FILE = LOGS_DIR / "success.jsonl"
TRAINING_FILE = LOGS_DIR / "training_queue.jsonl"
ERROR_FILE = LOGS_DIR / "error_buffer.json"
LORA_FILE = LOGS_DIR / "lora_examples.jsonl"

JUDGE_SYSTEM = (
    "You are a strict, technically precise evaluator. Style, analogy and "
    "storytelling must never inflate a score: grade truth over prose.\n"
    "Reason in this fixed order and reflect it in the JSON: "
    "step_1_extract_claims, step_2_verify_facts_against_whitelist or ground truth, "
    "step_3_check_ambiguity_warnings, step_4_generate_critique, "
    "step_5_calculate_weighted_score - the final score is computed last, "
    "after the critique text.\n"
    "Respond with a single JSON object with exactly these keys:\n"
    '"claims": [{"claim": string, "factuality": number 0..1}],\n'
    '"factuality": number 0..1 (fraction of claims that are true),\n'
    '"rubric_marks": {"ambiguity_flag": bool, "unwarned_hallucination": bool, '
    '"purely_cosmetic_analogy": bool, "technical_accuracy": bool},\n'
    '"flaws": [string],\n'
    '"critique": string (concrete correction guidance, written before the score),\n'
    '"critique_prompt": string (same concrete guidance),\n'
    '"score": number 0..1,\n'
    '"status": "perfect" | "acceptable" | "failed".\n'
    'Ambiguity rule: if a term is ambiguous or unknown and the answer does NOT '
    'warn "this depends / several meanings", set rubric_marks.ambiguity_flag=false '
    "and rubric_marks.unwarned_hallucination=true. "
    "A purely cosmetic analogy that adds no factual value sets "
    "rubric_marks.purely_cosmetic_analogy=true. "
    "Perfect means score >= 0.9, acceptable 0.75 <= score < 0.9, failed below 0.75."
)


class Status(str, Enum):
    PERFECT = "perfect"
    ACCEPTABLE = "acceptable"
    FAILED = "failed"


ALLOWED_STATUSES = {status.value for status in Status}


@dataclass(slots=True)
class Config:
    model: str = os.environ.get("GLOW_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")
    chat_endpoint: str = os.environ.get(
        "GLOW_ENDPOINT", "http://localhost:8000/v1/chat/completions"
    )
    models_endpoint: str = os.environ.get(
        "GLOW_MODELS_ENDPOINT", "http://localhost:8000/v1/models"
    )
    request_timeout: float = float(os.environ.get("GLOW_TIMEOUT", "300"))
    temperature: float = 0.7
    judge_temperature: float = 0.0
    max_tokens: int = 1024
    judge_max_tokens: int = 512
    perfect_threshold: float = 0.9
    acceptable_threshold: float = 0.75
    effort: str = os.environ.get("GLOW_EFFORT", "low")
    factuality_minimum: float = 0.70
    factuality_cap: float = 0.65
    judge_penalties: tuple[tuple[str, float], ...] = (
        ("unwarned_hallucination", 0.30),
        ("missing_ambiguity_flag", 0.20),
        ("purely_cosmetic_analogy", 0.10),
    )
    max_retries: int = 2
    stream: bool = True
    repeat_line_limit: int = 4
    repeat_ngram_size: int = 10
    repeat_ngram_limit: int = 3
    forbidden_patterns: tuple[str, ...] = (
        r"<\|endoftext\|>",
        r"<\|eot_id\|>",
        r"<\|im_end\|>",
        r"<\|start_header_id\|>",
    )


@dataclass(slots=True)
class AnomalySignal:
    kind: str
    detail: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "position": self.position}


@dataclass(slots=True)
class Generation:
    text: str
    tokens: int
    elapsed: float
    anomaly: AnomalySignal | None
    aborted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "elapsed": round(self.elapsed, 3),
            "aborted": self.aborted,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None,
        }


@dataclass(slots=True)
class Evaluation:
    score: float
    status: Status
    flaws: list[str] = field(default_factory=list)
    critique_prompt: str = ""
    factuality: float | None = None
    applied_penalties: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "status": self.status.value,
            "flaws": list(self.flaws),
            "critique_prompt": self.critique_prompt,
            "factuality": self.factuality,
            "applied_penalties": list(self.applied_penalties),
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class Attempt:
    index: int
    prompt: str
    answer: str
    evaluation: Evaluation
    generation: Generation
    judge_elapsed: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "prompt": self.prompt,
            "answer": self.answer,
            "evaluation": self.evaluation.to_dict(),
            "generation": self.generation.to_dict(),
            "judge_elapsed": self.judge_elapsed,
        }


@dataclass(slots=True)
class RunResult:
    run_id: str
    prompt: str
    delivered_status: Status
    delivered_score: float
    answer: str
    note: str
    attempts: list[Attempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "prompt": self.prompt,
            "delivered_status": self.delivered_status.value,
            "delivered_score": round(self.delivered_score, 3),
            "answer": self.answer,
            "note": self.note,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(slots=True)
class BestOfResult:
    prompt: str
    mode: str
    best: dict[str, Any] | None
    variants: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "mode": self.mode,
            "best": self.best,
            "variants": self.variants,
        }


class LLMClient(Protocol):
    def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]: ...

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> str: ...


def _message_text(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    reason = (
        message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )
    content = message.get("content") or ""
    if reason and f"<think>" not in content:
        return f"<think>{reason}</think>\n{content}"
    return content


class VLLMClient:
    def __init__(self, config: Config):
        self._config = config

    def _payload(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        response_format: dict[str, str] | None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)
        return payload

    async def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, temperature, max_tokens, True, None)
        timeout = httpx.Timeout(self._config.request_timeout, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", self._config.chat_endpoint, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
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
                    content = delta.get("content")
                    if content:
                        yield content

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        payload = self._payload(
            messages, temperature, max_tokens, False, response_format, extra_body
        )
        timeout = httpx.Timeout(self._config.request_timeout, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self._config.chat_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
        return _message_text(data["choices"][0]["message"])


class MockClient:
    def __init__(
        self,
        config: Config | None = None,
        answers: Sequence[str] | None = None,
        judge_scores: Sequence[float] | None = None,
    ):
        self._answers = list(answers or [])
        self._judge_scores = list(judge_scores or [])

    async def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        if self._answers:
            text = self._answers.pop(0)
        else:
            text = f"Mock answer for: {messages[-1]['content'][:60]}"
        for token in text.split(" "):
            await asyncio.sleep(0)
            yield token + " "

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> str:
        score = self._judge_scores.pop(0) if self._judge_scores else 0.95
        if score >= 0.9:
            status, flaws, critique = "perfect", [], ""
        elif score >= 0.75:
            status, flaws, critique = "acceptable", ["minor issue"], "tighten wording"
        else:
            status, flaws, critique = (
                "failed",
                ["incorrect result"],
                "recompute step by step",
            )
        return json.dumps(
            {
                "score": score,
                "status": status,
                "flaws": flaws,
                "critique_prompt": critique,
            }
        )


class StreamAnomalyDetector:
    def __init__(self, config: Config):
        self._config = config
        self._patterns = [re.compile(pattern) for pattern in config.forbidden_patterns]
        self._current_line: list[str] = []
        self._last_line: str | None = None
        self._line_repeats = 0
        self._tokens: deque[str] = deque(
            maxlen=config.repeat_ngram_size * (config.repeat_ngram_limit + 1)
        )
        self._token_carry = ""
        self._fence_open = False
        self._length = 0

    def update(self, chunk: str) -> AnomalySignal | None:
        self._length += len(chunk)
        for pattern in self._patterns:
            match = pattern.search(chunk)
            if match:
                return AnomalySignal(
                    "forbidden_token",
                    f"matched {match.group(0)!r}",
                    self._length,
                )
        if chunk.count("```") % 2 == 1:
            self._fence_open = not self._fence_open
        signal = self._update_lines(chunk)
        if signal is not None:
            return signal
        return self._update_ngrams(chunk)

    def _update_lines(self, chunk: str) -> AnomalySignal | None:
        for char in chunk:
            if char != "\n":
                self._current_line.append(char)
                continue
            line = "".join(self._current_line).rstrip()
            self._current_line.clear()
            if not line:
                continue
            if line == self._last_line:
                self._line_repeats += 1
            else:
                self._last_line = line
                self._line_repeats = 1
            if self._line_repeats >= self._config.repeat_line_limit:
                return AnomalySignal(
                    "repeated_line",
                    f"line repeated {self._line_repeats}x: {line[:80]!r}",
                    self._length,
                )
        return None

    def _update_ngrams(self, chunk: str) -> AnomalySignal | None:
        text = self._token_carry + chunk
        parts = text.split()
        if text and not text[-1].isspace():
            self._token_carry = parts[-1] if parts else ""
            parts = parts[:-1]
        else:
            self._token_carry = ""
        for token in parts:
            self._tokens.append(token)
        size = self._config.repeat_ngram_size
        limit = self._config.repeat_ngram_limit
        needed = size * limit
        if len(self._tokens) < needed:
            return None
        window = list(self._tokens)[-needed:]
        block = window[:size]
        if not any(block):
            return None
        if all(window[i : i + size] == block for i in range(0, needed, size)):
            return AnomalySignal(
                "repeated_sequence",
                f"token block repeated {limit}x",
                self._length,
            )
        return None

    def finalize(self, text: str) -> AnomalySignal | None:
        if self._fence_open:
            return AnomalySignal(
                "unterminated_code_block",
                "code fence opened but never closed",
                len(text),
            )
        return None


class Generator:
    def __init__(self, client: LLMClient, config: Config):
        self._client = client
        self._config = config

    async def generate(self, messages: Sequence[dict[str, str]]) -> Generation:
        chunks: list[str] = []
        detector = StreamAnomalyDetector(self._config)
        anomaly: AnomalySignal | None = None
        started = time.perf_counter()
        try:
            async for chunk in self._client.stream_chat(
                messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            ):
                chunks.append(chunk)
                anomaly = detector.update(chunk)
                if anomaly is not None:
                    break
        except Exception as exc:
            anomaly = AnomalySignal(
                "stream_error",
                f"{type(exc).__name__}: {exc}",
                sum(len(chunk) for chunk in chunks),
            )
        elapsed = time.perf_counter() - started
        text = "".join(chunks)
        if anomaly is None:
            anomaly = detector.finalize(text)
        return Generation(
            text=text,
            tokens=len(chunks),
            elapsed=elapsed,
            anomaly=anomaly,
            aborted=anomaly is not None,
        )


class Evaluator:
    def __init__(self, client: LLMClient, config: Config):
        self._client = client
        self._config = config

    async def evaluate(
        self,
        prompt: str,
        answer: str,
        is_code: bool | None = None,
        context: Sequence[str] | None = None,
        ground_truth: dict[str, Any] | None = None,
    ) -> Evaluation:
        if is_code is None:
            is_code = looks_like_code(prompt, answer)
        system = JUDGE_SYSTEM
        if is_code:
            system += (
                " The task involves code: mentally execute it and check the"
                " returned result before scoring."
            )
        parts = [f"Task:\n{prompt}", f"\n\nAssistant answer:\n{answer}"]
        if ground_truth:
            parts.append(
                "\n\nGround truth context (validate against it):\n"
                + json.dumps(ground_truth, ensure_ascii=False)
            )
        if context:
            joined = "\n".join(context) if not isinstance(context, str) else context
            parts.append(
                "\n\nVerified knowledge (use only to check facts):\n" + joined
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "".join(parts)},
        ]
        try:
            raw = await self._client.chat(
                messages,
                temperature=self._config.judge_temperature,
                max_tokens=self._config.judge_max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            raw = await self._client.chat(
                messages,
                temperature=self._config.judge_temperature,
                max_tokens=self._config.judge_max_tokens,
            )
        ground_truth_known = ground_truth is not None
        ambiguity_known = bool(
            ground_truth and ground_truth.get("is_ambiguous") is True
        )
        return self._parse(
            raw,
            ambiguity_known=ambiguity_known,
            ground_truth_known=ground_truth_known,
        )

    def _parse(
        self,
        raw: str,
        ambiguity_known: bool = False,
        ground_truth_known: bool = False,
    ) -> Evaluation:
        data = extract_json(raw)
        if not isinstance(data, dict):
            return Evaluation(
                score=0.0,
                status=Status.FAILED,
                flaws=["evaluator returned invalid JSON"],
                critique_prompt="Return the answer as a strict JSON object.",
                raw=raw,
            )
        base_hint = _as_float(data.get("score"), _status_to_score(data.get("status")))
        base = clamp(base_hint) if base_hint is not None else 0.0
        factuality = _as_float(data.get("factuality"), None)
        rubric = data.get("rubric_marks")
        rubric = rubric if isinstance(rubric, dict) else {}
        penalty_map = dict(self._config.judge_penalties)
        applied: list[str] = []
        subtract = 0.0
        if ground_truth_known and rubric.get("unwarned_hallucination"):
            applied.append("unwarned_hallucination")
            subtract += penalty_map.get("unwarned_hallucination", 0.0)
        if rubric.get("ambiguity_flag") is False and ambiguity_known:
            applied.append("missing_ambiguity_flag")
            subtract += penalty_map.get("missing_ambiguity_flag", 0.0)
        if ground_truth_known and rubric.get("purely_cosmetic_analogy"):
            applied.append("purely_cosmetic_analogy")
            subtract += penalty_map.get("purely_cosmetic_analogy", 0.0)
        score = clamp(base - subtract)
        if factuality is not None and factuality < self._config.factuality_minimum:
            score = min(score, self._config.factuality_cap)
        status = derive_status(score, self._config)
        flaws = data.get("flaws")
        flaws = [str(item) for item in flaws] if isinstance(flaws, list) else []
        critique = str(data.get("critique_prompt") or data.get("critique") or "")
        reasoning = str(data.get("critique") or "")
        return Evaluation(
            score=score,
            status=status,
            flaws=flaws,
            critique_prompt=critique,
            factuality=round(factuality, 3) if factuality is not None else None,
            applied_penalties=applied,
            reasoning=reasoning,
            raw=raw,
        )


class Buffer:
    _lock = asyncio.Lock()

    def __init__(self, path: Path):
        self._path = path

    async def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        async with Buffer._lock:
            await asyncio.to_thread(self._write, line)

    def _write(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class ReportWriter:
    def __init__(self, reports_dir: Path = REPORTS_DIR, runs_dir: Path = RUNS_DIR):
        self._reports_dir = reports_dir
        self._runs_dir = runs_dir

    def write(self, result: RunResult) -> Path:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        run_file = self._runs_dir / f"{result.run_id}.json"
        run_file.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = self._reports_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        with report_file.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(self._markdown(result)) + "\n")
        return run_file

    def _markdown(self, result: RunResult) -> list[str]:
        lines = [
            "",
            f"## Run {result.run_id}",
            "",
            f"- Delivered status: {result.delivered_status.value} "
            f"(score {result.delivered_score:.2f})",
            f"- Attempts: {len(result.attempts)}",
            f"- Note: {result.note}",
            f"- Prompt: {result.prompt}",
            "",
        ]
        for attempt in result.attempts:
            evaluation = attempt.evaluation
            anomaly = attempt.generation.anomaly
            lines.append(
                f"### Attempt {attempt.index + 1}: {evaluation.status.value} "
                f"(score {evaluation.score:.2f})"
            )
            lines.append(f"- Flaws: {'; '.join(evaluation.flaws) or 'none'}")
            if evaluation.critique_prompt:
                lines.append(f"- Critique: {evaluation.critique_prompt}")
            if anomaly is not None:
                lines.append(f"- Anomaly: {anomaly.kind} ({anomaly.detail})")
            lines.append(
                f"- Tokens: {attempt.generation.tokens}, "
                f"elapsed: {attempt.generation.elapsed:.2f}s"
            )
            lines.append("")
        lines.extend(["### Delivered answer", "", result.answer, ""])
        return lines


class Orchestrator:
    def __init__(
        self,
        client: LLMClient,
        config: Config | None = None,
        work_dir: Path | None = None,
    ):
        self._config = config or Config()
        self._client = client
        self._generator = Generator(client, self._config)
        self._evaluator = Evaluator(client, self._config)
        if work_dir is None:
            buffer_file = BUFFER_FILE
            success_file = SUCCESS_FILE
            training_file = TRAINING_FILE
            reporter = ReportWriter()
        else:
            work_dir = Path(work_dir)
            buffer_file = work_dir / "buffer.jsonl"
            success_file = work_dir / "success.jsonl"
            training_file = work_dir / "training_queue.jsonl"
            reporter = ReportWriter(
                reports_dir=work_dir / "reports",
                runs_dir=work_dir / "runs",
            )
        self._buffer = Buffer(buffer_file)
        self._success = Buffer(success_file)
        self._training = Buffer(training_file)
        self._reporter = reporter

    async def run(
        self,
        prompt: str,
        system: str | None = None,
        is_code: bool | None = None,
    ) -> RunResult:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        attempts: list[Attempt] = []
        best: Attempt | None = None
        critique: str | None = None
        delivered: Attempt | None = None
        note = ""
        for index in range(self._config.max_retries + 1):
            messages = self._build_messages(prompt, system, attempts, critique)
            generation = await self._generator.generate(messages)
            judge_elapsed: float | None = None
            if generation.anomaly is not None:
                evaluation = Evaluation(
                    score=0.0,
                    status=Status.FAILED,
                    flaws=[
                        f"stream anomaly: {generation.anomaly.kind} "
                        f"({generation.anomaly.detail})"
                    ],
                    critique_prompt=(
                        "Regenerate without repetition and close every code block."
                    ),
                )
            else:
                judge_start = time.perf_counter()
                evaluation = await self._evaluator.evaluate(
                    prompt, generation.text, is_code=is_code
                )
                judge_elapsed = time.perf_counter() - judge_start
            attempt = Attempt(
                index=index,
                prompt=prompt,
                answer=generation.text,
                evaluation=evaluation,
                generation=generation,
                judge_elapsed=judge_elapsed,
            )
            attempts.append(attempt)
            await self._buffer.append(self._record(run_id, attempt))
            if best is None or evaluation.score > best.evaluation.score:
                best = attempt
            if evaluation.status == Status.PERFECT:
                delivered = attempt
                note = "delivered perfect answer"
                await self._success.append(self._record(run_id, attempt))
                break
            if evaluation.status == Status.ACCEPTABLE:
                delivered = attempt
                note = "delivered acceptable answer; flaws queued for training"
                await self._training.append(self._record(run_id, attempt))
                break
            critique = (
                evaluation.critique_prompt
                or "; ".join(evaluation.flaws)
                or "Improve correctness and completeness."
            )
            if index == self._config.max_retries:
                delivered = best
                note = (
                    f"delivered best-effort answer after {len(attempts)} attempts "
                    f"(best score {best.evaluation.score:.2f})"
                )
                await self._training.append(self._record(run_id, best))
        result = RunResult(
            run_id=run_id,
            prompt=prompt,
            delivered_status=delivered.evaluation.status,
            delivered_score=delivered.evaluation.score,
            answer=delivered.answer,
            note=note,
            attempts=attempts,
        )
        await asyncio.to_thread(self._reporter.write, result)
        return result

    def _build_messages(
        self,
        prompt: str,
        system: str | None,
        attempts: Sequence[Attempt],
        critique: str | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if attempts and critique:
            last = attempts[-1]
            flaws = "; ".join(last.evaluation.flaws) or "not specified"
            messages.append({"role": "assistant", "content": last.answer})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous answer scored "
                        f"{last.evaluation.score:.2f}/1.00 and was rejected. "
                        f"Flaws: {flaws}. Revision guidance: {critique} "
                        "Return a corrected answer."
                    ),
                }
            )
        return messages

    async def run_best_of_n(
        self,
        prompt: str,
        temperatures: Sequence[float] = (0.2, 0.7, 1.1),
        system: str | None = None,
        is_code: bool | None = None,
        mode: str = "gloy_code",
    ) -> BestOfResult:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        variants: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for index, temperature in enumerate(temperatures):
            cfg = Config(
                model=self._config.model,
                chat_endpoint=self._config.chat_endpoint,
                models_endpoint=self._config.models_endpoint,
                temperature=temperature,
                max_retries=0,
            )
            generator = Generator(self._client, cfg)
            evaluator = Evaluator(self._client, cfg)
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            generation = await generator.generate(messages)
            judge_elapsed = None
            if generation.anomaly is not None:
                evaluation = Evaluation(
                    score=0.0,
                    status=Status.FAILED,
                    flaws=[
                        f"stream anomaly: {generation.anomaly.kind} "
                        f"({generation.anomaly.detail})"
                    ],
                    critique_prompt="Regenerate without repetition.",
                )
            else:
                judge_start = time.perf_counter()
                evaluation = await evaluator.evaluate(
                    prompt, generation.text, is_code=is_code
                )
                judge_elapsed = time.perf_counter() - judge_start
            record = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": cfg.model,
                "temperature": temperature,
                "judge_temperature": cfg.judge_temperature,
                "mode": mode,
                "attempt": index,
                "prompt": prompt,
                "answer": generation.text,
                "status": evaluation.status.value,
                "score": round(evaluation.score, 3),
                "factuality": evaluation.factuality,
                "penalties": evaluation.applied_penalties,
                "flaws": list(evaluation.flaws),
                "critique_prompt": evaluation.critique_prompt,
                "anomaly": generation.anomaly.to_dict()
                if generation.anomaly
                else None,
                "tokens": generation.tokens,
                "elapsed": round(generation.elapsed, 3),
                "judge_elapsed": (
                    round(judge_elapsed, 3) if judge_elapsed is not None else None
                ),
            }
            variants.append(record)
            await self._buffer.append(record)
            if evaluation.status == Status.PERFECT:
                await self._success.append(record)
            else:
                await self._training.append(record)
            if best is None or evaluation.score > best["score"]:
                best = record
        return BestOfResult(prompt=prompt, mode=mode, best=best, variants=variants)

    def _record(self, run_id: str, attempt: Attempt) -> dict[str, Any]:
        evaluation = attempt.evaluation
        anomaly = attempt.generation.anomaly
        return {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self._config.model,
            "temperature": self._config.temperature,
            "attempt": attempt.index,
            "prompt": attempt.prompt,
            "answer": attempt.answer,
            "status": evaluation.status.value,
            "score": round(evaluation.score, 3),
            "flaws": list(evaluation.flaws),
            "critique_prompt": evaluation.critique_prompt,
            "anomaly": anomaly.to_dict() if anomaly else None,
            "tokens": attempt.generation.tokens,
            "elapsed": round(attempt.generation.elapsed, 3),
            "judge_elapsed": (
                round(attempt.judge_elapsed, 3)
                if attempt.judge_elapsed is not None
                else None
            ),
        }


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def derive_status(score: float, config: Config) -> Status:
    if score >= config.perfect_threshold:
        return Status.PERFECT
    if score >= config.acceptable_threshold:
        return Status.ACCEPTABLE
    return Status.FAILED


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return clamp(float(value))
    except (TypeError, ValueError):
        return default


def _status_to_score(status: Any) -> float | None:
    if not isinstance(status, str):
        return None
    if status == "perfect":
        return 0.95
    if status == "acceptable":
        return 0.82
    if status == "failed":
        return 0.4
    return None


def extract_json(text: str) -> Any:
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(text[start:])
        return data
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


CODE_HINTS = ("```", "def ", "class ", "function ", "import ", "traceback", "compile")


def looks_like_code(prompt: str, answer: str) -> bool:
    haystack = f"{prompt}\n{answer}".lower()
    return any(hint in haystack for hint in CODE_HINTS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_model() -> Config:
    config = Config()
    response = httpx.get(config.models_endpoint, timeout=30)
    response.raise_for_status()
    return config


def ask_model(
    prompt: str,
    session: Config | None = None,
    config: Config | None = None,
) -> str:
    resolved = config or session or Config()
    client = VLLMClient(resolved)
    return asyncio.run(
        client.chat(
            [{"role": "user", "content": prompt}],
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
        )
    )


def evaluate_response(
    prompt: str,
    answer: str,
    session: Config | None = None,
    config: Config | None = None,
) -> int:
    resolved = config or session or Config()
    evaluation = asyncio.run(
        Evaluator(VLLMClient(resolved), resolved).evaluate(prompt, answer)
    )
    return int(round(evaluation.score * 10))


def save_error(prompt: str, answer: str) -> None:
    _append_jsonl(
        ERROR_FILE,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "answer": answer,
        },
    )


def background_learn(iterations: int = 100, sleep_seconds: float = 5.0) -> None:
    config = Config()
    client = VLLMClient(config)
    for _ in range(iterations):
        records = _read_jsonl(TRAINING_FILE)
        if not records:
            time.sleep(sleep_seconds)
            continue
        record = records[0]
        improved = asyncio.run(
            client.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the answer to remove the listed flaws. "
                            f"Question: {record['prompt']}\n"
                            f"Answer: {record['answer']}\n"
                            f"Flaws: {'; '.join(record.get('flaws', []))}"
                        ),
                    }
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )
        _append_jsonl(
            LORA_FILE,
            {
                "prompt": record["prompt"],
                "weak_answer": record["answer"],
                "strengthened": improved,
                "score": record.get("score"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Gloy AGI self-checking orchestrator")
    parser.add_argument("prompt", nargs="?", default="What is 2+2?")
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--endpoint", default=defaults.chat_endpoint)
    parser.add_argument("--max-retries", type=int, default=defaults.max_retries)
    parser.add_argument("--mock", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(
        model=args.model,
        chat_endpoint=args.endpoint,
        max_retries=args.max_retries,
    )
    client: LLMClient = MockClient() if args.mock else VLLMClient(config)
    result = asyncio.run(Orchestrator(client, config).run(args.prompt))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
