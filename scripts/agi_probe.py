"""Held-out AGI probes. Progress is a number. This is not AGI."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
PROBE_DIR = BASE_DIR / "logs" / "agi_probe"
HISTORY_FILE = PROBE_DIR / "history.jsonl"
LATEST_FILE = PROBE_DIR / "latest.json"

# Never train on these prompts. If a row matches, LoRA must skip it.
PROBES: list[dict[str, Any]] = [
    {
        "id": "novel_count",
        "kind": "exact",
        "prompt": "A gloy-box holds 3 red tokens and 4 blue. You add 2 red and remove 1 blue. How many tokens are left? Reply with one integer.",
        "expect": "8",
    },
    {
        "id": "unknown_mass",
        "kind": "unknown",
        "prompt": "What is the exact rest mass in kilograms of asteroid Gloy-ZX-991-TEST? If you do not have a source, say you do not know.",
    },
    {
        "id": "transfer_budget",
        "kind": "contains",
        "prompt": "Map conservation of energy to a monthly household budget in 4 sentences. Use the words 'conserved', 'leak', and 'account'.",
        "need": ["conserved", "leak", "account"],
    },
    {
        "id": "plan_bakery",
        "kind": "steps",
        "prompt": "You have 7 days to open a tiny bakery. List exactly 7 numbered steps. One action per step.",
        "min_steps": 7,
    },
    {
        "id": "abduce",
        "kind": "contains",
        "prompt": "The kitchen floor is wet, the window is open, a cup is on its side. Give the single best explanation in one sentence. Use the word 'because'.",
        "need": ["because"],
    },
    {
        "id": "physics_drop",
        "kind": "contains",
        "prompt": "You drop a steel ball and a same-size wooden ball in vacuum. Which hits first and why? Mention vacuum and gravity.",
        "need": ["vacuum", "gravity"],
    },
    {
        "id": "tom_sarcasm",
        "kind": "contains",
        "prompt": "A friend looks at a ruined cake and says 'perfect, just what I wanted'. What do they likely mean? Use the word 'sarcasm' or 'irony'.",
        "need_any": ["sarcasm", "irony", "сарказм", "ирони"],
    },
    {
        "id": "code_unique",
        "kind": "code",
        "prompt": "def unique_sorted(xs):",
        "tests": [
            "assert unique_sorted([3,1,2,1])==[1,2,3]",
            "assert unique_sorted([])==[]",
        ],
    },
]


def holdout_prompts() -> set[str]:
    return {str(item["prompt"]).strip().lower() for item in PROBES}


def is_holdout(prompt: str) -> bool:
    text = (prompt or "").strip().lower()
    if not text:
        return False
    if text in holdout_prompts():
        return True
    return any(text in item or item in text for item in holdout_prompts() if len(item) > 20)


def _steps(text: str) -> list[str]:
    return re.findall(r"^\s*(?:\d+[\).:-])\s+\S+", text or "", flags=re.M)


def score_answer(probe: dict[str, Any], answer: str) -> dict[str, Any]:
    text = (answer or "").strip()
    kind = probe["kind"]
    ok = False
    detail = ""
    if kind == "exact":
        nums = re.findall(r"-?\d+", text)
        ok = bool(nums) and nums[-1] == str(probe["expect"])
        detail = f"last_number={nums[-1] if nums else None}"
    elif kind == "unknown":
        refuse = bool(
            re.search(
                r"don'?t know|do not know|не знаю|unknown|no (?:reliable )?source|нет (?:данных|источника)",
                text,
                re.I,
            )
        )
        invented = bool(re.search(r"\d+\s*(kg|кг|kilogram)", text, re.I))
        ok = refuse and not invented
        detail = f"refuse={refuse} invented={invented}"
    elif kind == "contains":
        low = text.lower()
        need = [str(x).lower() for x in probe.get("need") or []]
        need_any = [str(x).lower() for x in probe.get("need_any") or []]
        ok_all = all(token in low for token in need) if need else True
        ok_any = any(token in low for token in need_any) if need_any else True
        ok = bool(text) and ok_all and ok_any
        detail = f"all={ok_all} any={ok_any}"
    elif kind == "steps":
        found = _steps(text)
        need = int(probe.get("min_steps") or 5)
        ok = len(found) >= need
        detail = f"steps={len(found)}"
    elif kind == "code":
        ok = False
        detail = "needs_runtime"
    return {"id": probe["id"], "kind": kind, "ok": ok, "detail": detail, "answer": text[:400]}


async def _score_code(probe: dict[str, Any], answer: str) -> dict[str, Any]:
    from hard_verify import _extract_python
    from continuous_learner import Sandbox

    code = _extract_python(answer)
    script = "\n".join(["import math", code, *probe["tests"], "print('SOLVED')"])
    result = await Sandbox().run(script, timeout=20, mem_mb=256)
    ok = result.exit_code == 0 and "SOLVED" in (result.stdout or "")
    return {
        "id": probe["id"],
        "kind": "code",
        "ok": ok,
        "detail": f"exit={result.exit_code}",
        "answer": (answer or "")[:400],
    }


async def _ask(client, prompt: str) -> str:
    raw = await client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
        extra_body={"enable_thinking": False},
    )
    return (raw or "").strip()


async def run_probe(client) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for probe in PROBES:
        try:
            raw = await _ask(client, probe["prompt"])
        except Exception as exc:
            results.append(
                {
                    "id": probe["id"],
                    "kind": probe["kind"],
                    "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "answer": "",
                }
            )
            continue
        if probe["kind"] == "code":
            scored = await _score_code(probe, raw)
        else:
            scored = score_answer(probe, raw)
        results.append(scored)
    solved = sum(1 for item in results if item["ok"])
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "solved": solved,
        "total": len(results),
        "score": round(solved / max(1, len(results)), 3),
        "results": results,
    }
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LATEST_FILE.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    b = float(before.get("score") or 0)
    a = float(after.get("score") or 0)
    lost = [
        item["id"]
        for item in after.get("results") or []
        if not item.get("ok")
        and any(
            old.get("id") == item["id"] and old.get("ok")
            for old in before.get("results") or []
        )
    ]
    gained = [
        item["id"]
        for item in after.get("results") or []
        if item.get("ok")
        and any(
            old.get("id") == item["id"] and not old.get("ok")
            for old in before.get("results") or []
        )
    ]
    if lost:
        verdict = "forgot"
    elif a > b + 1e-9:
        verdict = "grew"
    elif a + 1e-9 < b:
        verdict = "dropped"
    else:
        verdict = "flat"
    return {
        "verdict": verdict,
        "before": b,
        "after": a,
        "gained": gained,
        "lost": lost,
    }


def previous() -> dict[str, Any] | None:
    if not LATEST_FILE.exists():
        return None
    try:
        return json.loads(LATEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


async def _amain() -> dict[str, Any]:
    from orchestrator import Config, VLLMClient

    return await run_probe(VLLMClient(Config()))


def main() -> int:
    argparse.ArgumentParser(description="held-out AGI probe").parse_args()
    record = asyncio.run(_amain())
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
