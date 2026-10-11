"""Persistent goals + microtasks. This is a file, not months of inner life."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
GOALS_FILE = BASE_DIR / "logs" / "continuous" / "goals.json"
_STEP_RE = re.compile(r"^\s*(?:\d+[\).:-]|[-*])\s+(.+)$", re.M)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict[str, Any]:
    if not GOALS_FILE.exists():
        return {"goals": []}
    try:
        return json.loads(GOALS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"goals": []}


def _save(data: dict[str, Any]) -> None:
    GOALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOALS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_steps(text: str) -> list[str]:
    steps = [match.group(1).strip() for match in _STEP_RE.finditer(text or "")]
    clean = [item for item in steps if 3 <= len(item) <= 240]
    return clean[:16]


def seed_plan(text: str) -> list[str]:
    found = extract_steps(text)
    if found:
        return found
    goal = (text or "").strip() or "unspecified goal"
    return [
        f"Restate constraints for: {goal[:160]}",
        "List unknowns that block the next action",
        "Gather the cheapest missing fact (search or calculate)",
        "Do the first executable slice",
        "Write how we know that slice worked",
        "Queue the next slice from what we learned",
    ]


def add_goal(text: str, *, horizon: str = "open", source: str = "user") -> dict[str, Any]:
    text = (text or "").strip()
    if len(text) < 3:
        return {}
    data = _load()
    for goal in data["goals"]:
        if goal.get("status") == "active" and goal.get("text") == text:
            return goal
    goal = {
        "id": uuid.uuid4().hex[:12],
        "text": text,
        "horizon": horizon,
        "source": source,
        "status": "active",
        "created": _now(),
        "tasks": [
            {"id": uuid.uuid4().hex[:8], "text": step, "status": "todo"}
            for step in seed_plan(text)
        ],
    }
    data["goals"].append(goal)
    _save(data)
    return goal


def attach_plan(goal_id: str, steps: list[str]) -> dict[str, Any] | None:
    data = _load()
    for goal in data["goals"]:
        if goal.get("id") != goal_id:
            continue
        existing = {task["text"] for task in goal.get("tasks", [])}
        for step in steps:
            if step and step not in existing:
                goal.setdefault("tasks", []).append(
                    {"id": uuid.uuid4().hex[:8], "text": step, "status": "todo"}
                )
        _save(data)
        return goal
    return None


def list_active() -> list[dict[str, Any]]:
    return [goal for goal in _load().get("goals", []) if goal.get("status") == "active"]


def next_work() -> dict[str, Any] | None:
    for goal in list_active():
        for task in goal.get("tasks", []):
            if task.get("status") in ("todo", "doing"):
                task["status"] = "doing"
                data = _load()
                for stored in data["goals"]:
                    if stored.get("id") == goal["id"]:
                        stored["tasks"] = goal["tasks"]
                _save(data)
                return {
                    "prompt": (
                        f"Long-term goal ({goal.get('horizon')}): {goal['text']}\n"
                        f"Do only this microtask, then say what is next:\n{task['text']}"
                    ),
                    "goal_id": goal["id"],
                    "task_id": task["id"],
                    "goal": goal["text"],
                    "task": task["text"],
                }
    return None


def complete_task(goal_id: str, task_id: str, *, ok: bool = True) -> None:
    data = _load()
    for goal in data["goals"]:
        if goal.get("id") != goal_id:
            continue
        for task in goal.get("tasks", []):
            if task.get("id") == task_id:
                task["status"] = "done" if ok else "todo"
        if goal.get("tasks") and all(t.get("status") == "done" for t in goal["tasks"]):
            goal["status"] = "done"
            goal["finished"] = _now()
        _save(data)
        return


def ingest_from_text(text: str, goal_id: str | None = None) -> list[str]:
    steps = extract_steps(text)
    if goal_id and steps:
        attach_plan(goal_id, steps)
    return steps
