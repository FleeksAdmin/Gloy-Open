"""Reasoning search: sample multiple answers, self-consistency, verify."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Sequence

TEMP_SCHEDULE = (0.2, 0.6, 0.9)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zа-яё0-9]{3,}", text.lower()))


def overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def self_consistent(answers: Sequence[str]) -> tuple[str, float]:
    scores = [sum(overlap(answers[i], answers[j]) for j in range(len(answers)) if i != j) for i in range(len(answers))]
    if not answers:
        return "", 0.0
    best = max(range(len(answers)), key=lambda i: scores[i])
    return answers[best], round(scores[best], 3)


def strip_thinking(raw: str) -> str:
    text = raw.strip()
    if "\nresponse\n" in text:
        text = text.split("\nresponse\n", 1)[1]
    for tag in ("<scratchpad>", "<answer>"):
        text = re.sub(rf"<{tag[1:-1]}.*?</{tag[1:-1]}>", "", text, flags=re.S)
    return text.strip()


class ReasoningSearch:
    def __init__(self, client, max_tokens: int = 512):
        self._client = client
        self._max_tokens = max_tokens

    async def _sample(self, prompt: str, temperature: float) -> str:
        try:
            raw = await self._client.chat(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            return f"[error] {type(exc).__name__}: {exc}"
        return strip_thinking(raw)

    async def solve(
        self,
        prompt: str,
        samples: int = 3,
        verifier=None,
        kind: str = "text",
    ) -> dict[str, Any]:
        temps = TEMP_SCHEDULE[:max(1, samples)]
        answers = [await self._sample(prompt, t) for t in temps]
        chosen, consistency = self_consistent(answers)
        verified = False
        if verifier is not None:
            for candidate in answers:
                try:
                    ok = verifier(candidate, kind=kind)
                except Exception:
                    ok = False
                if ok:
                    chosen, verified = candidate, True
                    break
        return {
            "chosen": chosen,
            "consistency": consistency,
            "samples": answers,
            "verified": verified,
        }