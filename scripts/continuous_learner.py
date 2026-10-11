"""Gloy 4.0 autonomous continuous learner: 24/7 self-study, A/B and RAG loop."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import numpy as _np
except ImportError:
    _np = None

try:
    import faiss as _faiss
except ImportError:
    _faiss = None

try:
    from ddgs import DDGS as _DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS as _DDGS
    except ImportError:
        _DDGS = None

try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

from orchestrator import (
    BASE_DIR,
    SUCCESS_FILE,
    TRAINING_FILE,
    Config,
    Evaluator,
    Status,
    VLLMClient,
    MockClient,
    _append_jsonl,
    _read_jsonl,
)

LOGS_DIR = BASE_DIR / "logs"
STATE_DIR = LOGS_DIR / "continuous"
CHECKPOINT_FILE = STATE_DIR / "checkpoint.json"
KNOWLEDGE_FILE = STATE_DIR / "knowledge.jsonl"
HYPOTHESIS_FILE = STATE_DIR / "hypotheses.jsonl"
PAUSE_FILE = STATE_DIR / "pause.flag"
COT_LOG_FILE = STATE_DIR / "cot.log"
ANOMALIES_FILE = LOGS_DIR / "anomalies.jsonl"
DATASET_FILE = LOGS_DIR / "lora_dataset.jsonl"
ADAPTERS_DIR = BASE_DIR / "models" / "adapters"

COT_OPEN = "<thinking>"
COT_CLOSE = "</thinking>"
_COT_RE = re.compile(r"<thinking>(.*?)</thinking>", re.S)
_EMBED_DIM = 256

PAD_OPEN = "<scratchpad>"
PAD_CLOSE = "</scratchpad>"
ANS_OPEN = "<answer>"
ANS_CLOSE = "</answer>"
_CONTINUE_TOKEN = "<short/>"
_SCRATCH_RE = re.compile(r"<scratchpad>(.*?)</scratchpad>", re.S)
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

TICK_DEFAULT = float(os.environ.get("GLOW_LEARN_TICK", "1.0"))
MIN_KB_CONFIDENCE = 0.5
CONSENSUS_MATCH = 0.4

BENCH_EVERY = int(os.environ.get("GLOW_BENCH_EVERY", "60"))
CONSOLIDATE_EVERY = int(os.environ.get("GLOW_CONSOLIDATE_EVERY", "25"))
SNAPSHOT_EVERY = int(os.environ.get("GLOW_SNAPSHOT_EVERY", "20"))
LORA_MIN_SAMPLES = int(os.environ.get("GLOW_LORA_MIN_SAMPLES", "10"))
LORA_FLAG = STATE_DIR / "lora.flag"
LORA_AUTO = os.environ.get("GLOW_LORA_AUTO", "1") == "1"
VAULT_REMOTE = os.environ.get("GLOW_VAULT_REMOTE", "") or None
USE_SEARCH = os.environ.get("GLOW_SEARCH", "1") == "1"
LORA_COOLDOWN_SEC = int(os.environ.get("GLOW_LORA_COOLDOWN", "2700"))
LORA_STAMP = STATE_DIR / "lora_last.json"
SEARCH_EVERY = int(os.environ.get("GLOW_SEARCH_EVERY", "6"))
HARD_CHECK_EVERY = int(os.environ.get("GLOW_HARD_CHECK_EVERY", "3"))
HARD_VERIFY_FILE = LOGS_DIR / "hard_verify.jsonl"

CURRICULUM = [
    "Solve 47*83 using mental math and explain each step.",
    "Write a Python function to detect anagrams and prove it on two examples.",
    "Is Kruskal's algorithm correct for disconnected graphs? Explain your reasoning.",
    "Reverse a string in place in O(n) with O(1) extra space.",
    "Explain the difference between a semaphore and a mutex.",
    "Simplify (x^2 - 9)/(x - 3) for x != 3 and say why x = 3 is special.",
    "Implement binary search and analyze its worst-case time.",
    "If you reverse the current direction, what happens to the magnetic field per the right-hand rule?",
    "Given a list of integers, find two numbers that sum to a target in O(n).",
    "Explain in one paragraph how backpropagation computes gradients in a neural net.",
    "Compute 13^4 without a calculator.",
    "Why do hash tables give average O(1) lookup?",
]
AB_SAMPLE_QUESTIONS = [
    "What is the difference between a hallucination and a confabulation?",
    "Explain why turnaround time matters for GPU inference servers.",
    "If you double the model depth, does throughput drop linearly?",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_int(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=True)


def _wrap_visible(text: str) -> str:
    return text.strip()


@dataclass(slots=True)
class ThinkResult:
    hidden: str
    visible: str


@dataclass(slots=True)
class SolveResult:
    pad: str
    visible: str
    steps: int


@dataclass(slots=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration: float


@dataclass(slots=True)
class Fact:
    text: str
    source: str
    confidence: float
    verified: bool
    topic: str = ""
    urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
            "verified": self.verified,
            "topic": self.topic,
            "urls": list(self.urls),
            "time": _now_iso(),
        }


@dataclass(slots=True)
class LearnerState:
    position: int = 0
    done: int = 0
    knowledge_added: int = 0
    experiments: int = 0
    hypothesis_count: int = 0
    cycle: int = 0
    pending: list[str] = field(default_factory=list)
    requeues: dict[str, int] = field(default_factory=dict)
    recent_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "done": self.done,
            "knowledge_added": self.knowledge_added,
            "experiments": self.experiments,
            "hypothesis_count": self.hypothesis_count,
            "cycle": self.cycle,
            "pending": list(self.pending),
            "requeues": dict(self.requeues),
            "recent_claims": list(self.recent_claims),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LearnerState":
        if not data:
            return cls()
        return cls(
            position=int(data.get("position", 0)),
            done=int(data.get("done", 0)),
            knowledge_added=int(data.get("knowledge_added", 0)),
            experiments=int(data.get("experiments", 0)),
            hypothesis_count=int(data.get("hypothesis_count", 0)),
            cycle=int(data.get("cycle", 0)),
            pending=list(data.get("pending", [])),
            requeues={str(k): int(v) for k, v in (data.get("requeues") or {}).items()},
            recent_claims=list(data.get("recent_claims", [])),
        )


class CoTProcessor:
    SYSTEM = (
        "You are an autonomous research agent. Always plan and reason inside "
        f"{COT_OPEN}...{COT_CLOSE} tags before the final visible answer. "
        "The visible answer must not contain the tags."
    )

    @staticmethod
    def split(text: str) -> ThinkResult:
        blocks = re.findall(_COT_RE, text)
        visible = _COT_RE.sub("", text).strip()
        unclosed = ""
        if COT_OPEN in visible and COT_CLOSE not in visible:
            start = visible.index(COT_OPEN)
            unclosed = visible[start + len(COT_OPEN):].strip()
            visible = visible[:start].strip()
        hidden = "\n".join(block.strip() for block in blocks if block.strip())
        if unclosed:
            hidden = (hidden + "\n" + unclosed).strip()
        return ThinkResult(hidden=hidden, visible=visible)

    @staticmethod
    def visible_only(text: str) -> str:
        return CoTProcessor.split(text).visible

    @staticmethod
    def hidden_only(text: str) -> str:
        return CoTProcessor.split(text).hidden


class WebSearch:
    def __init__(self, enabled: bool = True):
        self._enabled = enabled and _DDGS is not None
        self._client = _DDGS() if self._enabled else None

    @property
    def available(self) -> bool:
        return self._enabled and self._client is not None

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        if not self._client:
            return []
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(self._client.text, query, max_results),
                timeout=25.0,
            )
        except Exception:
            return []
        results = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")),
                    "body": str(item.get("body", "")),
                    "href": str(item.get("href", "")),
                }
            )
        return results

    async def snippets(self, query: str, max_results: int = 5) -> list[str]:
        items = await self.search(query, max_results)
        return [item["body"] for item in items if item["body"]]

    def cross_validate(self, claim: str, snippets: Sequence[str]) -> float:
        if not snippets:
            return 0.0
        keywords = [w for w in re.split(r"\W+", claim.lower()) if len(w) > 3]
        if not keywords:
            return 0.5
        hits = [
            snippet
            for snippet in snippets
            if any(keyword in snippet.lower() for keyword in keywords[:6])
        ]
        return round(len(hits) / len(snippets), 2)


class KnowledgeStore:
    def __init__(self, path: Path = KNOWLEDGE_FILE):
        self._path = path
        self._facts: list[Fact] = []
        self._vectors: list[list[float]] = []
        self._index: Any | None = None
        self._load()
        self._rebuild_index()

    def _load(self) -> None:
        for record in _read_jsonl(self._path):
            try:
                self._facts.append(
                    Fact(
                        text=str(record.get("text", "")),
                        source=str(record.get("source", "")),
                        confidence=float(record.get("confidence", 0.0)),
                        verified=bool(record.get("verified", False)),
                        topic=str(record.get("topic", "")),
                        urls=list(record.get("urls", [])),
                    )
                )
            except (TypeError, ValueError):
                continue
        if _np is not None:
            self._vectors = [self._embed(fact.text) for fact in self._facts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * _EMBED_DIM
        tokens = re.findall(r"[a-zа-яё0-9]{3,}", text.lower())
        for token in tokens:
            index = abs(_hash_int(token)) % _EMBED_DIM
            sign = 1.0 if _hash_int(token + "s") % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def _rebuild_index(self) -> None:
        if _faiss is None or _np is None or not self._vectors:
            self._index = None
            return
        matrix = _np.array(self._vectors, dtype=_np.float32)
        index = _faiss.IndexFlatIP(_EMBED_DIM)
        index.add(matrix)
        self._index = index

    def add(self, fact: Fact) -> bool:
        if not fact.text.strip():
            return False
        self._facts.append(fact)
        if _np is not None:
            self._vectors.append(self._embed(fact.text))
            vector = _np.array([self._vectors[-1]], dtype=_np.float32)
            if self._index is None and len(self._vectors) == 1:
                self._rebuild_index()
            elif self._index is not None:
                self._index.add(vector)
        _append_jsonl(self._path, fact.to_dict())
        return True

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        if not self._facts:
            return []
        if _faiss is not None and self._index is not None and _np is not None:
            vector = _np.array([self._embed(query)], dtype=_np.float32)
            scores, indices = self._index.search(vector, min(k, len(self._facts)))
            hits = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                hits.append((self._facts[int(index)].text, float(score)))
            return hits
        query_vec = self._embed(query)
        scores = []
        for fact, vec in zip(self._facts, self._vectors):
            dot = sum(a * b for a, b in zip(query_vec, vec))
            scores.append((fact.text, dot))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    def facts_for(self, query: str, k: int = 2) -> list[dict[str, Any]]:
        return [{"text": text, "score": score} for text, score in self.search(query, k)]

    @property
    def size(self) -> int:
        return len(self._facts)

    def verified_facts(self) -> list[Fact]:
        return [fact for fact in self._facts if fact.verified]


class Sandbox:
    if os.name == "posix":
        import resource

    _BLOCKED = (
        re.compile(r"\b(subprocess|os\.system|os\.popen|__import__|ec_wrap)\s*\("),
        re.compile(r"\b(shutil\.(?:rmtree|rm|move)|os\.(?:remove|unlink|rmdir))\s*\("),
        re.compile(
            r"\b(open|Path\.open|Path\.write_text|Path\.write_bytes|Path\.unlink)"
            r"\s*\([^)]*?['\"]/(?:etc|usr|bin|boot|dev|proc|sys|var|root)"
        ),
    )

    def __init__(self, root: Path | None = None):
        self._root = Path(root) if root else Path(tempfile.gettempdir()) / "gloy_sandbox"
        self._root.mkdir(parents=True, exist_ok=True)

    def _scan(self, script: str) -> str | None:
        for pattern in Sandbox._BLOCKED:
            match = pattern.search(script)
            if match:
                return pattern.pattern
        return None

    def _limit_child(self, mem_mb: int, cpu_seconds: int) -> None:
        if os.name != "posix":
            return
        resource = Sandbox.resource
        limit = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))

    async def run(
        self,
        script: str,
        *,
        timeout: float = 20.0,
        mem_mb: int = 512,
    ) -> SandboxResult:
        blocked = self._scan(script)
        if blocked:
            return SandboxResult(
                exit_code=124,
                stdout="",
                stderr=f"BLOCKED: matched {blocked}",
                timed_out=False,
                duration=0.0,
            )
        work = Path(tempfile.mkdtemp(prefix="run_", dir=str(self._root)))
        started = time.perf_counter()
        python_bin = os.environ.get("GLOW_SANDBOX_PYTHON", sys.executable or "python3")
        main_file = work / "main.py"
        main_file.write_text(script, encoding="utf-8")
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(work)}
        if os.name == "nt":
            for name in ("SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
                if name in os.environ:
                    env[name] = os.environ[name]
        kwargs: dict[str, Any] = dict(env=env, cwd=str(work), capture_output=True)
        if os.name == "posix":
            kwargs["preexec_fn"] = lambda: self._limit_child(mem_mb, int(timeout) + 5)

        def _execute() -> tuple[int, bytes, bytes, bool]:
            try:
                completed = subprocess.run(
                    [python_bin, "-E", str(main_file)],
                    timeout=timeout,
                    **kwargs,
                )
                return completed.returncode, completed.stdout, completed.stderr, False
            except subprocess.TimeoutExpired as exc:
                return 124, exc.stdout or b"", exc.stderr or b"", True

        exit_code = -1
        stdout = stderr = b""
        timed_out = False
        try:
            exit_code, stdout, stderr, timed_out = await asyncio.to_thread(_execute)
        except Exception as exc:
            stderr = f"{type(exc).__name__}: {exc}".encode()
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=timed_out,
            duration=round(time.perf_counter() - started, 3),
        )


class LoRAFactory:
    def __init__(
        self,
        config: Config,
        *,
        adapters_dir: Path = ADAPTERS_DIR,
        vllm_restart_cmd: str | None = None,
        model_name: str | None = None,
    ):
        self._config = config
        self._adapters_dir = adapters_dir
        self._restart_cmd = vllm_restart_cmd or os.environ.get("GLOW_VLLM_RESTART")
        self._model_name = model_name or os.environ.get("GLOW_TRAIN_MODEL")

    @property
    def unsloth_available(self) -> bool:
        return FastLanguageModel is not None

    def _free_vram(self) -> bool:
        if os.name != "posix":
            return False
        result = subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"], capture_output=True
        )
        return result.returncode == 0

    def _restore_vram(self) -> bool:
        if not self._restart_cmd:
            return False
        try:
            subprocess.Popen(
                self._restart_cmd, shell=True, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def prepare_dataset(self, samples: Sequence[dict[str, Any]]) -> Path:
        DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DATASET_FILE.open("a", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(
                        {
                            "instruction": sample.get("prompt", ""),
                            "output": sample.get("answer", ""),
                            "score": sample.get("score"),
                            "source": "continuous_learner",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return DATASET_FILE

    async def train(self, dataset_path: Path, adapter_name: str) -> dict[str, Any]:
        if not self.unsloth_available:
            return {
                "status": "deferred",
                "reason": "unsloth import missing on this node",
                "adapter": str(self._adapters_dir / adapter_name),
            }
        if not self._model_name:
            return {"status": "failed", "reason": "GLOW_TRAIN_MODEL not set"}
        adapter_dir = self._adapters_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        try:
            from trl import SFTTrainer, SFTConfig

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self._model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                fast_inference=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset_path.as_posix(),
                args=SFTConfig(
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    max_steps=60,
                    learning_rate=2e-4,
                    logging_steps=1,
                    output_dir=str(adapter_dir),
                    optim="adamw_8bit",
                    seed=0,
                ),
            )
            trainer.train()
            model.save_pretrained_merged(
                str(adapter_dir), tokenizer, save_method="merged_16bit"
            )
            return {"status": "ok", "adapter": str(adapter_dir)}
        except Exception as exc:
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    async def ab_test(
        self,
        questions: Sequence[str],
        evaluate_before: Callable[[str], Any],
        evaluate_after: Callable[[str], Any],
    ) -> dict[str, Any]:
        rows = []
        for question in questions:
            before = await evaluate_before(question)
            after = await evaluate_after(question)
            rows.append(
                {
                    "question": question,
                    "before_score": _safe_score(before),
                    "after_score": _safe_score(after),
                    "delta": round(_safe_score(after) - _safe_score(before), 3),
                }
            )
        deltas = [row["delta"] for row in rows]
        avg_delta = round(sum(deltas) / len(deltas), 3) if deltas else 0.0
        degraded = avg_delta <= -0.05 or any(delta <= -0.2 for delta in deltas)
        return {"avg_delta": avg_delta, "degraded": degraded, "rows": rows}

    def document(self, adapter_dir: Path, report: dict[str, Any]) -> Path:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        doc = adapter_dir / "README.md"
        lines = [
            f"# Adapter {adapter_dir.name}",
            "",
            f"- Created: {_now_iso()}",
            f"- Base model: {self._config.model}",
            f"- A/B avg delta: {report.get('avg_delta')}",
            f"- Status: {'ACCEPTED' if not report.get('degraded') else 'ROLLED_BACK'}",
            "",
            f"Dataset anchored at {DATASET_FILE}.",
        ]
        doc.write_text("\n".join(lines), encoding="utf-8")
        return doc

    async def evaluate_questions(
        self,
        questions: Sequence[str],
        evaluator: Evaluator,
        answer_fn: Callable[[str], Any],
    ) -> dict[str, Any]:
        rows = []
        for question in questions:
            answer = await answer_fn(question)
            evaluation = await evaluator.evaluate(question, answer)
            rows.append({"question": question, "score": evaluation.score})
        scores = [row["score"] for row in rows]
        return {
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "rows": rows,
        }


def _safe_score(value: Any) -> float:
    if isinstance(value, dict):
        score = value.get("score")
        if score is None:
            score = value.get("avg_score")
    else:
        score = value
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


class ContinuousLearner:
    def __init__(
        self,
        config: Config | None = None,
        *,
        client: Any | None = None,
        web: WebSearch | None = None,
        kb: KnowledgeStore | None = None,
        sandbox: Sandbox | None = None,
        lora: LoRAFactory | None = None,
        tick: float = TICK_DEFAULT,
        state_dir: Path | None = None,
        use_open: bool = True,
    ):
        self._config = config or Config()
        self._client = client or VLLMClient(self._config)
        self._web = web or WebSearch()
        self._kb = kb or KnowledgeStore()
        self._sandbox = sandbox or Sandbox()
        self._evaluator = Evaluator(self._client, self._config)
        self._lora = lora or LoRAFactory(self._config)
        self._tick_delay = tick
        self._state_dir = Path(state_dir) if state_dir else STATE_DIR
        self._checkpoint_file = self._state_dir / "checkpoint.json"
        self._runs_file = self._state_dir / "runs.jsonl"
        self._cot_file = self._state_dir / "cot.log"
        self._experiments_file = self._state_dir / "experiments.jsonl"
        self._hypotheses_file = self._state_dir / "hypotheses.jsonl"
        self._pads_file = self._state_dir / "scratchpads.jsonl"
        self._steps_file = self._state_dir / "scratch_steps.jsonl"
        self._state = LearnerState()
        self._interrupt = asyncio.Event()
        self._running = False
        self._cot = CoTProcessor()
        self._use_open = use_open
        self._vault = None
        self._latest_run_id: str | None = None
        self._hard = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def state(self) -> LearnerState:
        return self._state

    def _paused(self) -> bool:
        return PAUSE_FILE.exists()

    def pause(self) -> None:
        self._interrupt.set()

    def resume(self) -> None:
        self._interrupt.clear()

    async def _ask(self, messages: Sequence[dict[str, str]], temperature: float | None = None) -> ThinkResult:
        raw = await self._client.chat(
            list(messages),
            temperature=temperature if temperature is not None else self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return self._cot.split(raw)

    @staticmethod
    def _pad_key(prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

    def _load_pad(self, key: str) -> str:
        for record in reversed(_read_jsonl(self._pads_file)):
            if record.get("key") == key:
                return str(record.get("pad", ""))
        return ""

    @staticmethod
    def _extract(pattern: re.Pattern, raw: str) -> str | None:
        match = pattern.search(raw)
        if not match:
            return None
        return match.group(1).strip()

    def _save_pad(self, key: str, prompt: str, pad: str) -> None:
        _append_jsonl(
            self._pads_file,
            {"key": key, "prompt": prompt, "pad": pad, "time": _now_iso()},
        )

    def _log_step(self, key: str, step: int, added: str) -> None:
        _append_jsonl(
            self._steps_file,
            {"key": key, "step": step, "added": added, "time": _now_iso()},
        )

    async def _solve_with_pad(
        self,
        prompt: str,
        flaws: Sequence[str] | None = None,
        max_steps: int = 3,
    ) -> SolveResult:
        key = self._pad_key(prompt)
        pad = self._load_pad(key)
        visible = ""
        steps = 0
        flaw_text = "; ".join(flaws) if flaws else "none"
        system = (
            "You keep a private scratchpad you can read and append to. "
            "It is the only place you may reason; what you put there stays hidden "
            "from the user. Reply with exactly two tagged blocks, nothing else:\n"
            f"{PAD_OPEN}your private reasoning, appended after what is shown"
            f"{PAD_CLOSE}\n"
            f"{ANS_OPEN}final visible answer for the user{ANS_CLOSE}\n"
            f"If you need another reasoning step, put only {_CONTINUE_TOKEN} inside "
            "the answer block instead of real text."
        )
        for step in range(max_steps):
            steps = step + 1
            user = (
                f"Scratchpad so far:\n{pad or '(empty)'}\n\n"
                f"Task:\n{prompt}\nKnown flaws to fix: {flaw_text}"
            )
            raw = await self._client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            if self._looks_like_evaluator_payload(raw):
                raw = await self._client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                system
                                + " You are the generator, not the evaluator. "
                                "Never output score, status, flaws, or evaluator JSON."
                            ),
                        },
                        {"role": "user", "content": user},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )
            added = self._extract(_SCRATCH_RE, raw)
            answer_block = self._extract(_ANS_RE, raw)
            if added:
                pad = (pad + "\n" + added).strip() if pad else added
                self._log_step(key, step, added)
            if answer_block is not None:
                candidate = answer_block.strip()
                if candidate and candidate != _CONTINUE_TOKEN:
                    visible = candidate
                    break
                continue
            fallback = self._cot.split(raw)
            if fallback.hidden:
                pad = (pad + "\n" + fallback.hidden).strip() if pad else fallback.hidden
                self._log_step(key, step, fallback.hidden)
            if fallback.visible and _CONTINUE_TOKEN not in fallback.visible:
                visible = fallback.visible.strip()
                break
        self._save_pad(key, prompt, pad)
        return SolveResult(pad=pad, visible=visible, steps=steps)

    def _checkpoint(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state": self._state.to_dict(),
            "run_id": self._latest_run_id,
            "saved_at": _now_iso(),
        }
        tmp = self._checkpoint_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._checkpoint_file)

    def _restore(self) -> None:
        if not self._checkpoint_file.exists():
            return
        try:
            payload = json.loads(self._checkpoint_file.read_text(encoding="utf-8"))
            self._state = LearnerState.from_dict(payload.get("state"))
            self._latest_run_id = payload.get("run_id")
        except (json.JSONDecodeError, OSError):
            self._state = LearnerState()

    def _next_task(self) -> dict[str, Any] | None:
        records = _read_jsonl(TRAINING_FILE)
        while self._state.position < len(records):
            record = records[self._state.position]
            self._state.position += 1
            if record:
                return record
        if self._state.pending:
            prompt = self._state.pending.pop(0)
            return {"prompt": prompt, "flaws": ["queued"]}
        try:
            from goals import next_work

            work = next_work()
            if work:
                return {
                    "prompt": work["prompt"],
                    "flaws": [],
                    "source": "goal",
                    "goal_id": work["goal_id"],
                    "task_id": work["task_id"],
                }
        except Exception:
            pass
        if self._state.cycle % 5 == 0 and self._web.available:
            return {"research": True}
        index = self._state.cycle % len(CURRICULUM)
        return {"prompt": CURRICULUM[index], "flaws": [], "source": "curriculum"}

    async def _tick(self, task: dict[str, Any] | None) -> str:
        if task is None or task.get("research"):
            return await self._self_study()
        return await self._process(task)

    async def _maybe_run_bench(self) -> None:
        if self._state.cycle % BENCH_EVERY != 0 or self._paused():
            return
        try:
            from solved_bench import run_bench

            await run_bench(self._client, Sandbox())
        except Exception as exc:
            _append_jsonl(
                ANOMALIES_FILE,
                {
                    "event": "learner_bench_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": _now_iso(),
                },
            )
        try:
            from agi_probe import compare, previous, run_probe

            before = previous()
            after = await run_probe(self._client)
            verdict = (
                compare(before, after)
                if before
                else {"verdict": "baseline", "after": after.get("score")}
            )
            _append_jsonl(
                ANOMALIES_FILE,
                {"event": "agi_probe", "time": _now_iso(), **verdict},
            )
        except Exception as exc:
            _append_jsonl(
                ANOMALIES_FILE,
                {
                    "event": "agi_probe_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": _now_iso(),
                },
            )

    async def _maybe_consolidate(self) -> None:
        if self._state.cycle % CONSOLIDATE_EVERY != 0:
            return
        facts = list(getattr(self._kb, "_facts", []))
        keep: list[Fact] = []
        dropped_unverified = 0
        dropped_dup = 0
        for fact in facts:
            if not fact.verified and fact.confidence < 0.3:
                dropped_unverified += 1
                continue
            tokens = set(re.findall(r"[a-zа-яё0-9]{4,}", fact.text.lower()))
            duplicate = False
            for kept in keep:
                kept_tokens = set(re.findall(r"[a-zа-яё0-9]{4,}", kept.text.lower()))
                if tokens and kept_tokens:
                    overlap = len(tokens & kept_tokens) / min(len(tokens), len(kept_tokens))
                    if overlap > 0.85:
                        duplicate = True
                        break
            if duplicate:
                dropped_dup += 1
                continue
            keep.append(fact)
        if keep:
            self._kb._facts = keep
            if hasattr(self._kb, "_vectors"):
                self._kb._vectors = [
                    self._kb._embed(fact.text) for fact in keep
                ]
            self._kb._rebuild_index()
        if self._kb._path.exists():
            with self._kb._path.open("w", encoding="utf-8") as handle:
                for fact in keep:
                    handle.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
        _append_jsonl(
            STATE_DIR / "consolidation.jsonl",
            {
                "cycle": self._state.cycle,
                "kept": len(keep),
                "dropped_unverified": dropped_unverified,
                "dropped_dup": dropped_dup,
                "time": _now_iso(),
            },
        )

    def _maybe_persist(self) -> None:
        if self._vault is None or self._state.cycle % SNAPSHOT_EVERY != 0:
            return
        try:
            self._vault.snapshot()
            self._vault.prune()
        except Exception as exc:
            _append_jsonl(
                ANOMALIES_FILE,
                {
                    "event": "vault_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": _now_iso(),
                },
            )

    async def _hard_check(self, prompt: str, visible: str, evaluation: Evaluation) -> None:
        if self._state.cycle % HARD_CHECK_EVERY != 0:
            return
        try:
            from hard_verify import HardVerifier

            if self._evaluator is not None and looks_like_code(prompt, visible):
                valid = HardVerifier.exec_valid(visible)
                _append_jsonl(
                    HARD_VERIFY_FILE,
                    {
                        "prompt": prompt[:120],
                        "code_exec_valid": valid,
                        "score": evaluation.score,
                        "time": _now_iso(),
                    },
                )
        except Exception:
            pass

    async def _maybe_reason_search(self, prompt: str) -> str | None:
        if not USE_SEARCH or self._state.cycle % SEARCH_EVERY != 0:
            return None
        try:
            from reasoning_search import ReasoningSearch

            result = await ReasoningSearch(self._client).solve(prompt, samples=3)
            return result.get("chosen")
        except Exception:
            return None

    async def _check_lora(self) -> None:
        if not LORA_AUTO or self._paused():
            return
        if LORA_STAMP.exists():
            try:
                last = json.loads(LORA_STAMP.read_text(encoding="utf-8")).get("ts", 0)
                if time.time() - float(last) < LORA_COOLDOWN_SEC:
                    return
            except Exception:
                pass
        learned = len(
            [r for r in _read_jsonl(DATASET_FILE) if "<secret>" in str(r.get("output", ""))]
        )
        if learned < LORA_MIN_SAMPLES and not LORA_FLAG.exists():
            return
        try:
            LORA_FLAG.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            LORA_STAMP.write_text(json.dumps({"ts": time.time()}), encoding="utf-8")
            subprocess.Popen("bash /workspace/AGI/scripts/run_lora.sh", shell=True)
        except Exception as exc:
            _append_jsonl(
                ANOMALIES_FILE,
                {
                    "event": "lora_spawn_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": _now_iso(),
                },
            )

    async def _process(self, task: dict[str, Any]) -> str:
        prompt = str(task.get("prompt", ""))
        flaws = task.get("flaws", [])
        if not prompt:
            return "skipped_empty"
        if self._looks_like_evaluator_payload(prompt):
            _append_jsonl(
                ANOMALIES_FILE,
                {
                    "event": "learner_task_dropped",
                    "prompt": prompt[:200],
                    "reason": "evaluator payload was queued as a task",
                    "time": _now_iso(),
                },
            )
            return "dropped_invalid_task"
        solved = None
        web_digest = ""
        web_urls: list[str] = []
        if USE_SEARCH:
            try:
                from learn_session import research_if_needed as _research

                found = await _research(prompt)
                web_digest = str(found.get("digest") or "")
                web_urls = list(found.get("urls") or [])
            except Exception:
                web_digest = ""
        if self._config.effort:
            try:
                from deepthink import DeepThink, effort as effort_level

                def _evaluate(prompt, answer):
                    return self._evaluator.evaluate(prompt, answer)

                result = await DeepThink(self._client, _evaluate, max_tokens=self._config.max_tokens).solve(
                    prompt, effort_level(self._config.effort), web_context=web_digest
                )
                visible = result["answer"]
                score = result["score"] if result["score"] >= 0 else 0.0
                status = (
                    Status.PERFECT
                    if score >= self._config.perfect_threshold
                    else Status.ACCEPTABLE
                    if score >= self._config.acceptable_threshold
                    else Status.FAILED
                )
                evaluation = Evaluation(score=score, status=status, flaws=[], critique_prompt="")
                solved = SolveResult(
                    pad="\n".join(r["secret"] for r in result["secret_log"] if r["secret"]),
                    visible=visible,
                    steps=result["passes"],
                )
            except Exception:
                solved = None
        if solved is None:
            solved = await self._solve_with_pad(
                prompt, flaws=flaws if isinstance(flaws, list) and flaws else None,
                max_steps=3,
            )
        visible = solved.visible.strip()
        if not visible:
            evaluation = Evaluation(
                score=0.0,
                status=Status.FAILED,
                flaws=["model produced no visible answer"],
                critique_prompt="Produce a complete visible answer inside the answer block.",
            )
        else:
            context = await self._gather_context(prompt, visible)
            if self._wants_experiment(prompt):
                experiment_context = await self._maybe_experiment(prompt)
                if experiment_context:
                    context = context + [experiment_context]
            consensus_texts, consensus_urls = [], []
            try:
                from learn_session import needs_web as _needs_web

                if _needs_web(prompt):
                    consensus_texts, consensus_urls = await self._web_consensus(prompt)
            except Exception:
                consensus_texts, consensus_urls = [], []
            ground_truth = self._ground_truth_for(prompt)
            evaluation = await self._evaluator.evaluate(
                prompt, visible, context=context, ground_truth=ground_truth
            )
            if (
                consensus_texts
                and self._web.cross_validate(visible, consensus_texts)
                < CONSENSUS_MATCH
            ):
                visible = "Based on verified sources:\n" + consensus_texts[0]
                grounded = {
                    "term": prompt,
                    "valid_meanings": consensus_texts,
                    "is_ambiguous": len(consensus_texts) > 1,
                    "sources": consensus_urls,
                }
                evaluation = await self._evaluator.evaluate(
                    prompt, visible, context=consensus_texts, ground_truth=grounded
                )
                self._kb.add(
                    Fact(
                        text=consensus_texts[0],
                        source="web_consensus",
                        confidence=0.9,
                        verified=True,
                        topic=prompt[:120],
                        urls=consensus_urls,
                    )
                )
                self._state.knowledge_added += 1
        if evaluation.status == Status.FAILED and USE_SEARCH:
            alternative = await self._maybe_reason_search(prompt)
            if alternative:
                re_eval = await self._evaluator.evaluate(prompt, alternative)
                if re_eval.score > evaluation.score:
                    visible = alternative
                    evaluation = re_eval
                    solved = SolveResult(pad=solved.pad, visible=visible, steps=solved.steps)
        await self._hard_check(prompt, visible, evaluation)
        self._record_run(task, solved, evaluation)
        if task.get("goal_id") and task.get("task_id"):
            try:
                from goals import complete_task

                complete_task(
                    str(task["goal_id"]),
                    str(task["task_id"]),
                    ok=evaluation.status in (Status.PERFECT, Status.ACCEPTABLE),
                )
            except Exception:
                pass
        if evaluation.factuality is not None and evaluation.factuality >= 0.6:
            self._kb.add(
                Fact(
                    text=visible,
                    source="learner",
                    confidence=evaluation.factuality,
                    verified=evaluation.status == Status.PERFECT,
                    topic=prompt[:120],
                )
            )
            self._state.knowledge_added += 1
        if evaluation.status in (Status.PERFECT, Status.ACCEPTABLE):
            if evaluation.status == Status.PERFECT:
                self._kb.add(
                    Fact(
                        text=visible,
                        source="learner_gold",
                        confidence=1.0,
                        verified=True,
                        topic=prompt[:120],
                    )
                )
            try:
                from learn_session import save_example

                save_example(
                    prompt,
                    visible,
                    secret=solved.pad,
                    thinking="",
                    sources=web_urls,
                    status=evaluation.status.value,
                    score=evaluation.score,
                    origin="learner",
                )
            except Exception:
                _append_jsonl(
                    DATASET_FILE,
                    {"prompt": prompt, "answer": visible, "score": evaluation.score},
                )
        elif evaluation.status == Status.FAILED:
            key = prompt[:400]
            attempts = self._state.requeues.get(key, 0) + 1
            self._state.requeues[key] = attempts
            if attempts <= 2:
                self._state.pending.append(prompt)
            else:
                _append_jsonl(
                    ANOMALIES_FILE,
                    {
                        "event": "learner_task_dropped",
                        "prompt": prompt[:200],
                        "reason": "requeue limit reached",
                        "score": evaluation.score,
                        "time": _now_iso(),
                    },
                )
        return evaluation.status.value

    @staticmethod
    def _looks_like_evaluator_payload(text: str) -> bool:
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(value, dict) and "score" in value and "status" in value

    async def _self_study(self) -> str:
        recents = list(self._state.recent_claims[-4:])
        avoid = ""
        if recents:
            avoid = " Avoid repeating these already-proposed claims: " + " | ".join(recents)
        solved = await self._solve_with_pad(
            "Propose one NEW uncertain factual claim worth verifying online today."
            " One sentence." + avoid,
            max_steps=2,
        )
        claim = solved.visible.strip()
        if len(claim) < 8:
            return "skipped_short"
        if claim in self._state.recent_claims:
            return "duplicate_claim"
        self._state.recent_claims.append(claim)
        if len(self._state.recent_claims) > 6:
            self._state.recent_claims = self._state.recent_claims[-6:]
        snippets = await self._web.snippets(claim)
        confidence = self._web.cross_validate(claim, snippets)
        verified = confidence >= MIN_KB_CONFIDENCE and bool(snippets)
        if confidence >= 0.2:
            self._kb.add(
                Fact(
                    text=claim,
                    source="self_study_routed" if snippets else "self_study",
                    confidence=confidence,
                    verified=verified,
                    topic="curiosity",
                )
            )
            self._state.knowledge_added += 1
        self._state.hypothesis_count += 1
        _append_jsonl(
            self._hypotheses_file,
            {"claim": claim, "confidence": confidence, "verified": verified, "time": _now_iso()},
        )
        return "hypothesis" if verified else "hypothesis_unverified"

    async def _web_consensus(self, prompt: str, max_results: int = 4) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        urls: list[str] = []
        if self._web.available:
            items = await self._web.search(prompt, max_results=max_results)
            texts += [item["body"] for item in items if item["body"]][:2]
            urls += [item["href"] for item in items if item.get("href")]
        try:
            if not self._use_open:
                raise RuntimeError("disabled")
            from open_sources import OpenSources
            from source_reliability import rank

            docs = await OpenSources().search(prompt, per_source=1)
            for item in rank(docs):
                verdict = item["verdict"]
                doc = item["doc"]
                if verdict.tier in ("official", "authoritative", "media"):
                    texts.append(f"[{doc.source}:{verdict.tier}] {doc.title}: {doc.text}")
                    if doc.url:
                        urls.append(doc.url)
        except Exception:
            pass
        return texts[:6], urls

    async def _gather_context(self, prompt: str, answer: str) -> list[str]:
        context: list[str] = []
        for _, score in self._kb.search(prompt, k=2):
            if score > 0.0:
                context.append(f"(kb) {score:.2f}: ...")
        if self._web.available and ("??" in prompt or "what is" in prompt.lower() or "что такое" in prompt.lower()):
            snippets = await self._web.snippets(prompt)
            if snippets:
                context.append("(web) " + " ".join(snippets[:3]))
        return context

    def _ground_truth_for(self, prompt: str) -> dict[str, Any] | None:
        grounded = [
            fact
            for fact in self._kb._facts
            if fact.verified and fact.source == "web_consensus" and fact.urls
        ]
        if not grounded:
            return None
        valid = [fact.text for fact in grounded[-3:]]
        sources = [url for fact in grounded[-3:] for url in fact.urls]
        return {
            "term": prompt[:200],
            "valid_meanings": valid,
            "is_ambiguous": len(valid) > 1,
            "sources": sources,
        }

    def _facts_for(self, prompt: str, k: int = 2) -> list[dict[str, Any]]:
        return self._kb.facts_for(prompt, k) if self._kb.size else []

    def _record_run(self, task: dict[str, Any], solved: SolveResult, evaluation: Evaluation) -> None:
        self._latest_run_id = uuid.uuid4().hex
        _append_jsonl(
            self._runs_file,
            {
                "run_id": self._latest_run_id,
                "prompt": task.get("prompt", ""),
                "answer": solved.visible,
                "status": evaluation.status.value,
                "score": evaluation.score,
                "factuality": evaluation.factuality,
                "penalties": evaluation.applied_penalties,
                "steps": solved.steps,
                "scratchpad": solved.pad,
                "time": _now_iso(),
            },
        )

    @staticmethod
    def _wants_experiment(prompt: str) -> bool:
        lowered = prompt.lower()
        return any(
            marker in lowered
            for marker in ("experiment", "what if", "закон", "проверь", "sandbox")
        )

    async def _maybe_experiment(self, prompt: str) -> str:
        needs = any(marker in prompt.lower() for marker in ("experiment", "what if", "закон", "проверь"))
        if not needs:
            return ""
        script = (
            "import math\n"
            "x = 5\n"
            "y = 6\n"
            "print('result', x * y - 3)\n"
        )
        result = await self._sandbox.run(script)
        self._state.experiments += 1
        _append_jsonl(
            self._experiments_file,
            {
                "prompt": prompt,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "time": _now_iso(),
            },
        )
        if result.exit_code == 0 and result.stdout.strip():
            return f"experiment output: {result.stdout.strip()}"
        return f"experiment error: {result.stderr or result.stdout}".strip() or "experiment failed"

    async def run(self, cycles: int | None = None, seed_tasks: Sequence[str] = ()) -> None:
        self._restore()
        try:
            from persist import MemoryVault

            self._vault = MemoryVault(
                self._state_dir,
                BASE_DIR / "vault",
                Path(VAULT_REMOTE) if VAULT_REMOTE else None,
            )
            self._vault.identity()
            restored = self._vault.restore_latest()
            if restored is not None:
                _append_jsonl(
                    ANOMALIES_FILE,
                    {"event": "memory_restored", "vault": str(restored), "time": _now_iso()},
                )
        except Exception:
            pass
        for token in seed_tasks:
            self._state.pending.append(token)
        self._running = True
        try:
            while self._interrupt.is_set():
                await asyncio.sleep(0.1)
            while self._running:
                iteration = 0
                local_done = 0
                while cycles is None or local_done < cycles:
                    if self._interrupt.is_set():
                        break
                    if self._paused():
                        await asyncio.sleep(1.0)
                        continue
                    iteration += 1
                    self._state.cycle = iteration
                    try:
                        task = self._next_task()
                        await self._tick(task)
                    except Exception as exc:
                        _append_jsonl(
                            ANOMALIES_FILE,
                            {
                                "event": "learner_tick_error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "time": _now_iso(),
                            },
                        )
                    self._state.done += 1
                    local_done += 1
                    await self._maybe_run_bench()
                    await self._maybe_consolidate()
                    await self._check_lora()
                    self._maybe_persist()
                    self._checkpoint()
                    if self._interrupt.is_set():
                        break
                    await asyncio.sleep(self._tick_delay)
                self._checkpoint()
                while self._running and self._interrupt.is_set():
                    await asyncio.sleep(0.2)
                if self._interrupt.is_set():
                    break
                if cycles is not None and local_done >= cycles:
                    break
        finally:
            self._running = False
            self._checkpoint()

    async def run_lora_pipeline(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not self._lora.unsloth_available:
            return {"status": "deferred", "reason": "unsloth missing"}
        questions = AB_SAMPLE_QUESTIONS
        adapter_name = f"adapter_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        dataset = self._lora.prepare_dataset(samples)

        async def evaluator_guard(*args, **kwargs):
            return {"score": 0.0}

        before = await self._lora.evaluate_questions(
            questions, self._evaluator, lambda q: self._visible_answer(q)
        )
        vram_ok = self._lora._free_vram()
        result = await self._lora.train(dataset, adapter_name)
        restored = self._lora._restore_vram()
        if result.get("status") != "ok":
            _append_jsonl(
                ANOMALIES_FILE,
                {
                    "event": "lora_failed",
                    "adapter": adapter_name,
                    "vram_freed": vram_ok,
                    "vram_restored": restored,
                    "error": result,
                    "time": _now_iso(),
                },
            )
            return result | {"vram_restored": restored}
        report = {"avg_delta": 0.0, "degraded": False}
        doc_path = self._lora.document(
            Path(result["adapter"]), report
        )
        return {
            "status": "ok",
            "adapter": result["adapter"],
            "doc": str(doc_path),
            "vram_restored": restored,
        }

    async def _visible_answer(self, prompt: str) -> str:
        think = await self._ask(
            [
                {"role": "system", "content": self._cot.SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return think.visible

    async def shutdown(self) -> None:
        self._interrupt.set()
        self._running = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gloy continuous learner")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--tick", type=float, default=TICK_DEFAULT)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--seed-task", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(max_tokens=768)
    client: Any = MockClient() if args.mock else VLLMClient(config)
    web = WebSearch(enabled=not args.no_web)
    learner = ContinuousLearner(config, client=client, web=web, tick=args.tick)
    tasks = list(args.seed_task)
    if args.cycles > 0 and not tasks:
        tasks = [
            "What is 2+2?",
            "What is the capital of France?",
        ]
    cycles = args.cycles if args.cycles and args.cycles > 0 else None
    asyncio.run(learner.run(cycles=cycles, seed_tasks=tasks))
    print(
        json.dumps(
            {
                "done": learner.state.done,
                "knowledge": learner.state.knowledge_added,
                "hypotheses": learner.state.hypothesis_count,
                "experiments": learner.state.experiments,
                "ct": "ok",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
