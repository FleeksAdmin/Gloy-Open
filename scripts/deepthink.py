"""DeepThink: private <secret> first, then user-facing thinking, then answer."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable

SECRET_OPEN = "<secret>"
SECRET_CLOSE = "</secret>"
THINK_OPEN = "<thinking>"
THINK_CLOSE = "</thinking>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
_SECRET_RE = re.compile(r"<secret>(.*?)</secret>", re.S)
_NATIVE_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.S)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
_MORE = "[more]"

EFFORT_LEVELS = {"low": 1, "medium": 2, "high": 3, "max": 4}

SECRET_CONTRACT = (
    "Use your native private thinking. Be honest: what you know, what you "
    "don't, contradictions, failed ideas, the plan. Do not perform for the user. "
    "Do not give the final public answer yet — think first."
)

PUBLIC_CONTRACT = (
    "You already thought in a private <secret>. The user must NOT see that raw "
    "scratchpad. Peek at it, then produce the public trace.\n"
    "1) <thinking> — clean, ideal reasoning the user sees: structured, "
    "honest, no inner junk, no 'as an AI', no invented facts. It must follow "
    "from the secret, not replace it.\n"
    "2) <answer> — the final visible answer, built only from that thinking. "
    f"If the secret is still incomplete, put exactly {_MORE} in <answer>.\n"
    "Reply with exactly:\n"
    f"{THINK_OPEN}user-facing reasoning{THINK_CLOSE}\n"
    f"{ANSWER_OPEN}final answer or {_MORE}{ANSWER_CLOSE}\n"
    "Do not repeat <secret>."
)


@dataclass(slots=True)
class Effort:
    name: str
    passes: int


def effort(value: Any = "medium") -> Effort:
    if isinstance(value, Effort):
        return value
    name = str(value or "medium").lower()
    passes = EFFORT_LEVELS.get(name, 2)
    return Effort(name=name, passes=passes)


def _extract(pattern: re.Pattern, raw: str) -> str | None:
    match = pattern.search(raw)
    return match.group(1).strip() if match else None


def _strip(raw: str) -> str:
    cleaned = _ANSWER_RE.sub("", raw)
    cleaned = _SECRET_RE.sub("", cleaned)
    cleaned = _NATIVE_THINK_RE.sub("", cleaned)
    return _THINK_RE.sub("", cleaned).strip()


def _native_secret(raw: str) -> str:
    return (
        _extract(_NATIVE_THINK_RE, raw)
        or _extract(_SECRET_RE, raw)
        or _strip(raw)
        or raw
    )


async def _emit(on_event, kind: str, text: str) -> None:
    if not on_event:
        return
    result = on_event(kind, text)
    if inspect.isawaitable(result):
        await result


def _paragraph(text: str, limit: int = 400) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:limit] or "(no text)"


class DeepThink:
    def __init__(
        self,
        client,
        evaluator: Callable[[str, str], Any] | None = None,
        max_tokens: int = 768,
    ):
        self._client = client
        self._evaluator = evaluator
        self._max_tokens = max_tokens

    async def _chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int | None = None,
        native_think: bool = False,
        on_delta=None,
    ) -> str:
        extra = {
            "enable_thinking": native_think,
            "chat_template_kwargs": {"enable_thinking": native_think},
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        stream = getattr(self._client, "stream_chat", None)
        if on_delta is not None and stream is not None:
            parts: list[str] = []
            try:
                async for token in stream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens or self._max_tokens,
                    extra_body=extra,
                ):
                    if token:
                        parts.append(token)
                        result = on_delta(token)
                        if inspect.isawaitable(result):
                            await result
                return "".join(parts)
            except TypeError:
                pass
        try:
            return await self._client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens or self._max_tokens,
                extra_body=extra,
            )
        except TypeError:
            return await self._client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens or self._max_tokens,
            )

    async def _write_secret(
        self,
        prompt: str,
        secret: str,
        *,
        last: bool,
        web_context: str = "",
        cognition_brief: str = "",
        on_delta=None,
    ) -> str:
        system = SECRET_CONTRACT
        if cognition_brief:
            system += "\n" + cognition_brief
        if last:
            system += "\nLast private pass: settle what you believe and what remains unknown."
        web = f"\n\nOpen-web notes (verify, do not parrot):\n{web_context}" if web_context else ""
        user = (
            f"Task:\n{prompt}{web}\n\n"
            f"Private scratchpad so far (re-read this, then update it):\n"
            f"{secret or '(empty — this is the first look, start from scratch)'}"
        )
        raw = await self._chat(
            system,
            user,
            temperature=0.35,
            max_tokens=max(self._max_tokens, 1024),
            native_think=True,
            on_delta=on_delta,
        )
        scratch = _native_secret(raw)
        return (scratch or "").strip()[:4000]

    async def _write_public(
        self, prompt: str, secret: str, *, last: bool, on_delta=None
    ) -> tuple[str, str]:
        system = PUBLIC_CONTRACT
        if last:
            system += "\nBudget reached: do not use [more]; give the best final answer now."
        user = (
            f"Task:\n{prompt}\n\n"
            f"Peek at your private secret (source of truth, do not dump it):\n"
            f"{secret or '(empty)'}"
        )
        raw = await self._chat(system, user, temperature=0.2, on_delta=on_delta)
        thinking = _extract(_THINK_RE, raw) or ""
        declared = _extract(_ANSWER_RE, raw) or ""
        leftover = _strip(raw)
        if not thinking:
            thinking = leftover or _paragraph(secret)
        if not declared and leftover and leftover != thinking:
            declared = leftover
        return thinking.strip(), declared.strip()

    async def solve(
        self,
        prompt: str,
        effort_level: Any = "medium",
        web_context: str = "",
        on_event=None,
    ) -> dict[str, Any]:
        from cognition import assess, brief, to_dict

        eff = effort(effort_level)
        secret = ""
        steps: list[str] = []
        best_answer = ""
        best_score = -1.0
        terminal_answer = ""
        secret_log: list[dict[str, str]] = []
        cognition_brief = brief(prompt)
        budget = max(1, min(eff.passes, 8))
        for index in range(budget):
            last = index == budget - 1
            if on_event:
                await _emit(on_event, "status", f"secret pass {index + 1}/{budget}")
            delta = await self._write_secret(
                prompt,
                secret,
                last=last,
                web_context=web_context,
                cognition_brief=cognition_brief,
                on_delta=(lambda tok: on_event("secret", tok)) if on_event else None,
            )
            if delta:
                secret = (secret + "\n" + delta).strip()[-8000:]
            meta_mid = assess(secret)
            if meta_mid.need_web and not web_context:
                if on_event:
                    await _emit(on_event, "status", "web lookup")
                try:
                    from learn_session import research

                    found = await research(prompt)
                    web_context = str(found.get("digest") or "")
                except Exception:
                    pass
            if on_event:
                await _emit(on_event, "status", f"public pass {index + 1}/{budget}")
            thinking, declared = await self._write_public(
                prompt,
                secret,
                last=last,
                on_delta=(lambda tok: on_event("public", tok)) if on_event else None,
            )
            steps.append(thinking or _paragraph(delta or secret))
            secret_log.append(
                {
                    "step": index + 1,
                    "secret": delta,
                    "thinking": thinking,
                    "declared": declared,
                }
            )
            finished = bool(declared) and _MORE not in declared.lower()
            draft = declared if finished else ""
            if draft:
                if self._evaluator is not None:
                    evaluation = self._evaluator(prompt, draft)
                    if inspect.isawaitable(evaluation):
                        evaluation = await evaluation
                    score = float(getattr(evaluation, "score", 0.0))
                else:
                    score = -1.0
                if score > best_score:
                    best_answer, best_score = draft, score
                terminal_answer = draft
            if finished:
                break
        answer = terminal_answer or best_answer
        if not answer or re.match(r"^\s*(thinking\b|okay\b|так\b|ну\b)", answer, re.I):
            try:
                raw_final = await self._chat(
                    "Respond ONLY with the final solution inside "
                    f"{ANSWER_OPEN} and {ANSWER_CLOSE}. Build it from the thinking "
                    "implied by the private notes. No commentary.",
                    f"Task:\n{prompt}\n\nPrivate notes:\n{secret or '(empty)'}\n\n"
                    f"User-facing thinking:\n{steps[-1] if steps else '(none)'}",
                    temperature=0.15,
                )
                clean = _extract(_ANSWER_RE, raw_final) or _strip(raw_final)
                if clean and not re.match(r"^\s*(thinking\b|okay\b|так\b|ну\b)", clean, re.I):
                    answer = clean.strip()
            except Exception:
                pass
        meta = assess(secret, answer)
        return {
            "answer": answer,
            "score": best_score,
            "passes": len(secret_log),
            "effort": eff.name,
            "scratchpad": secret,
            "thinking_steps": steps,
            "secret_log": secret_log,
            "metacog": to_dict(meta),
            "cognition_brief": cognition_brief,
        }
