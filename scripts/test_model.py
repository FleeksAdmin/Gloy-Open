import json
import sys
from datetime import datetime
from pathlib import Path

from orchestrator import ask_model, load_model

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_FILE = BASE_DIR / "reports" / "2026-09-10.md"
TRAINING_LOG = BASE_DIR / "logs" / "training.log"

PROMPT = "What is 2+2?"


def _append_report(lines):
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = REPORT_FILE.open("a", encoding="utf-8")
    handle.write("\n".join(lines) + "\n")
    handle.close()


def _append_log(message):
    TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.utcnow().isoformat()} {message}\n")


def main():
    try:
        session = load_model()
        answer = ask_model(PROMPT, session=session)
        payload = {
            "endpoint_ok": True,
            "prompt": PROMPT,
            "answer": answer,
            "checked_at": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        payload = {
            "endpoint_ok": False,
            "prompt": PROMPT,
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at": datetime.utcnow().isoformat(),
        }
    _append_log(json.dumps(payload))
    _append_report(
        [
            "## First model test",
            "",
            f"- Prompt: `{payload['prompt']}`",
            f"- Endpoint ok: {payload['endpoint_ok']}",
        ]
    )
    if payload.get("endpoint_ok"):
        _append_report([f"- Answer: {payload['answer']!r}"])
    else:
        _append_report([f"- Error: {payload['error']}"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["endpoint_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())