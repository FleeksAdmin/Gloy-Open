"""Agent-layer stand-ins for the AGI checklist. Not model properties."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_PHYS = (
    "gravity", "гравитац", "friction", "трение", "collide", "столкнов",
    "material", "материал", "inertia", "инерц", "torque", "момент силы",
    "liquid", "жидкост", "heat", "температур", "pressure", "давлен",
)
_TOM = (
    "sarcasm", "сарказм", "motive", "мотив", "intent", "намерен",
    "lying", "врёт", "лже", "emotion", "эмоц", "обид", "why did they",
    "почему он", "почему она", "на самом деле",
)
_TRANSFER = (
    "по аналогии", "похожую", "another domain", "как в математике",
    "apply this to", "перенес",
)
_UNKNOWN = re.compile(
    r"(i don'?t know|не знаю|unknown|need (?:to )?(?:search|experiment|check)|"
    r"нужно (?:проверить|поискать|измерить)|cannot verify)",
    re.I,
)
_CONF = re.compile(r"confidence\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)", re.I)


@dataclass(slots=True)
class Profile:
    physics: bool = False
    tom: bool = False
    transfer: bool = False
    vague_goal: bool = False
    experiment: bool = False


@dataclass(slots=True)
class Metacog:
    unknowns: list[str] = field(default_factory=list)
    need_web: bool = False
    need_experiment: bool = False
    confidence: float | None = None
    honest: str = ""


def profile(prompt: str) -> Profile:
    text = (prompt or "").lower()
    words = text.split()
    return Profile(
        physics=any(token in text for token in _PHYS),
        tom=any(token in text for token in _TOM),
        transfer=any(token in text for token in _TRANSFER),
        vague_goal=len(words) <= 12 and any(
            token in text
            for token in ("открой", "построй", "сделай так", "помоги мне", "launch", "build a")
        ),
        experiment=any(
            token in text
            for token in ("experiment", "проверь", "what if", "закон ", "sandbox")
        ),
    )


def brief(prompt: str) -> str:
    flags = profile(prompt)
    lines = [
        "Metacognition: name what you know, what you don't, and whether the "
        "next move is search, a sandbox experiment, or a direct answer. "
        "If you lack a fact, say so — do not invent it.",
        "Reasoning modes: try deduction (must follow), induction (from cases), "
        "abduction (best explanation). Say which one you are using.",
    ]
    if flags.transfer or flags.vague_goal:
        lines.append(
            "Transfer: if a concept from another domain (math, logistics, "
            "biology, code) maps here, name the mapping in one sentence."
        )
    if flags.physics:
        lines.append(
            "Naive physics check: gravity, contact, conservation, materials. "
            "This is text, not a simulator — flag what would need a real experiment."
        )
    if flags.tom:
        lines.append(
            "Theory of mind: separate said words vs likely intent, emotion, "
            "and social rule. Mark sarcasm/hidden motive if present. Guess, don't claim certainty."
        )
    if flags.vague_goal:
        lines.append(
            "Decompose the fuzzy goal into the next 5–12 executable microtasks. "
            "Do not pretend you can hold a months-long plan without a stored goal file."
        )
    if flags.experiment:
        lines.append("Prefer a checkable experiment or calculation over prose.")
    return "\n".join(lines)


def assess(secret: str, answer: str = "") -> Metacog:
    blob = f"{secret or ''}\n{answer or ''}"
    unknowns = []
    for match in _UNKNOWN.finditer(blob):
        start = max(0, match.start() - 40)
        end = min(len(blob), match.end() + 80)
        snippet = re.sub(r"\s+", " ", blob[start:end]).strip()
        if snippet and snippet not in unknowns:
            unknowns.append(snippet[:160])
    need_web = bool(re.search(r"need (?:to )?search|нужно поискать|look up", blob, re.I))
    need_experiment = bool(
        re.search(r"need (?:to )?experiment|нужно (?:проверить|измерить)|sandbox", blob, re.I)
    )
    conf = None
    found = _CONF.search(blob)
    if found:
        try:
            conf = float(found.group(1))
        except ValueError:
            conf = None
    if unknowns and not need_web and not need_experiment:
        need_web = True
    return Metacog(
        unknowns=unknowns[:6],
        need_web=need_web,
        need_experiment=need_experiment,
        confidence=conf,
        honest="unknowns present" if unknowns else "no explicit unknown",
    )


def to_dict(meta: Metacog) -> dict[str, Any]:
    return {
        "unknowns": meta.unknowns,
        "need_web": meta.need_web,
        "need_experiment": meta.need_experiment,
        "confidence": meta.confidence,
        "honest": meta.honest,
    }
