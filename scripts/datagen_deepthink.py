"""Build a LoRA dataset that bakes the <secret> multi-pass reasoning pattern into weights."""

from __future__ import annotations

import json
from pathlib import Path

from deepthink import EFFORT_LEVELS

BASE = Path(__file__).resolve().parent.parent
RUNS_FILE = BASE / "logs" / "continuous" / "runs.jsonl"
OUT = BASE / "logs" / "deepthink_dataset.jsonl"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build() -> int:
    runs = _load(RUNS_FILE)
    examples: dict[str, dict] = {}
    for run in runs:
        if run.get("status") not in ("perfect", "acceptable"):
            continue
        prompt = str(run.get("prompt", ""))
        answer = str(run.get("answer", ""))
        secret = str(run.get("scratchpad", "") or "")
        if len(prompt) < 4 or len(answer) < 2 or len(secret) < 4:
            continue
        secret_summary = secret[:1200]
        thinking = str(run.get("thinking") or run.get("scratchpad") or "")[:800]
        output = (
            f"<secret>{secret_summary}</secret>\n"
            f"<thinking>{thinking}</thinking>\n"
            f"<answer>{answer[:1500]}</answer>"
        )
        examples[prompt] = {
            "instruction": f"Task: {prompt}",
            "output": output,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for example in examples.values():
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def main() -> int:
    count = build()
    print(json.dumps({
        "dataset": str(OUT),
        "examples": count,
        "effort_levels": EFFORT_LEVELS,
        "note": ("feed to Unsloth LoRA via train_lora/replay to bake the "
                 "<secret> multi-pass reasoning into the model weights"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())