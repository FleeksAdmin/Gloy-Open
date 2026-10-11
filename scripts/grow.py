"""The only honest bet: probe must rise and must not forget.

Not AGI. A closed loop you can run when the GPU is up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agi_probe import compare, previous, run_probe
from orchestrator import Config, VLLMClient

BASE_DIR = Path(__file__).resolve().parent.parent
GROW_FILE = BASE_DIR / "logs" / "agi_probe" / "grow.jsonl"


def _append(record: dict[str, Any]) -> None:
    GROW_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GROW_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def one_round(*, learn: bool) -> dict[str, Any]:
    client = VLLMClient(Config())
    before = previous() or await run_probe(client)
    if learn:
        try:
            from continuous_learner import ContinuousLearner

            learner = ContinuousLearner(client=client)
            await learner.run(cycles=1)
        except Exception as exc:
            learn = False
            learn_error = f"{type(exc).__name__}: {exc}"
        else:
            learn_error = ""
    else:
        learn_error = ""
    after = await run_probe(client)
    verdict = compare(before, after)
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "learn": learn,
        "learn_error": learn_error,
        **verdict,
    }
    _append(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="probe → optional learn → probe")
    parser.add_argument("--learn", action="store_true", help="run one learner cycle between probes")
    args = parser.parse_args()
    record = asyncio.run(one_round(learn=args.learn))
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record.get("verdict") != "forgot" else 2


if __name__ == "__main__":
    raise SystemExit(main())
