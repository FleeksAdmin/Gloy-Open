"""Solved-benchmark: measure how many tasks the model solves autonomously."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from orchestrator import Config, VLLMClient
from continuous_learner import PAUSE_FILE, Sandbox

BASE_DIR = Path(__file__).resolve().parent.parent
BENCH_DIR = BASE_DIR / "logs" / "bench"
HISTORY_FILE = BENCH_DIR / "solved_history.jsonl"
PROGRESS_FILE = BENCH_DIR / "progress.jsonl"

CODING_TASKS: list[dict[str, Any]] = [
    {"id": "ds_reverse", "prompt": "def reverse_string(s):", "tests": ["assert reverse_string('abc')=='cba'", "assert reverse_string('')==''"]},
    {"id": "ds_fib", "prompt": "def fib(n):", "tests": ["assert fib(0)==0", "assert fib(10)==55"]},
    {"id": "ds_anagram", "prompt": "def is_anagram(a,b):", "tests": ["assert is_anagram('listen','silent')", "assert not is_anagram('a','b')"]},
    {"id": "ds_two_sum", "prompt": "def two_sum(nums,target):", "tests": ["assert two_sum([2,7,11,15],9)==[0,1]", "assert two_sum([3,3],6)==[0,1]"]},
    {"id": "ds_max_sub", "prompt": "def max_subarray(nums):", "tests": ["assert max_subarray([-2,1,-3,4,-1,2,1,-5,4])==6"]},
    {"id": "ds_prime", "prompt": "def is_prime(n):", "tests": ["assert is_prime(17)", "assert not is_prime(18)"]},
    {"id": "ds_gcd", "prompt": "def gcd(a,b):", "tests": ["assert gcd(48,18)==6"]},
    {"id": "ds_merge", "prompt": "def merge_sorted(a,b):", "tests": ["assert merge_sorted([1,3],[2,4])==[1,2,3,4]"]},
    {"id": "ds_palindrome", "prompt": "def is_palindrome(s):", "tests": ["assert is_palindrome('racecar')", "assert not is_palindrome('hello')"]},
    {"id": "ds_bubble", "prompt": "def bubble_sort(arr):", "tests": ["assert bubble_sort([3,1,2])==[1,2,3]"]},
    {"id": "ds_bin", "prompt": "def to_binary(n):", "tests": ["assert to_binary(10)=='1010'"]},
    {"id": "ds_dupe", "prompt": "def contains_duplicate(nums):", "tests": ["assert contains_duplicate([1,2,1])", "assert not contains_duplicate([1,2,3])"]},
    {"id": "ds_factorial", "prompt": "def factorial(n):", "tests": ["assert factorial(5)==120"]},
    {"id": "ds_reverse_list", "prompt": "def reverse_list(xs):", "tests": ["assert reverse_list([1,2,3])==[3,2,1]"]},
    {"id": "ds_swap", "prompt": "def swap_first_last(xs):", "tests": ["assert swap_first_last([1,2,3,4])==[4,2,3,1]"]},
    {"id": "mbpp_sum", "prompt": "def sum_list(nums):", "tests": ["assert sum_list([1,2,3])==6"]},
    {"id": "mbpp_max", "prompt": "def max_of(a,b):", "tests": ["assert max_of(3,7)==7"]},
    {"id": "mbpp_even", "prompt": "def is_even(n):", "tests": ["assert is_even(10)", "assert not is_even(3)"]},
    {"id": "mbpp_vowels", "prompt": "def remove_vowels(s):", "tests": ["assert remove_vowels('hello')=='hll'"]},
    {"id": "mbpp_words", "prompt": "def count_words(s):", "tests": ["assert count_words('a b c')==3"]},
    {"id": "mbpp_longest", "prompt": "def longest(words):", "tests": ["assert longest(['a','bb','ccc'])=='ccc'"]},
    {"id": "mbpp_char", "prompt": "def count_char(s,c):", "tests": ["assert count_char('banana','a')==3"]},
    {"id": "mbpp_absdiff", "prompt": "def absolute_difference(a,b):", "tests": ["assert absolute_difference(3,10)==7"]},
]


def _wrap_run(tests: Sequence[str]) -> str:
    payload = "import math\n"
    runner = "\n".join(tests)
    return payload + "\n" + runner + "\nprint('SOLVED')"


def _extract_function(text: str) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(r"^\s*def\s+\w+", line):
            start = index
            break
    if start is None:
        return text.strip()
    out = [lines[start]]
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped or line.startswith(" ") or line.startswith("\t"):
            out.append(line)
            continue
        if re.match(r"^(def|class)\s+\w+", line):
            out.append(line)
            continue
        break
    return "\n".join(out).strip()


def _clean_code(raw: str) -> str:
    text = raw.strip()
    if "\nresponse\n" in text:
        text = text.split("\nresponse\n", 1)[1]
    text = re.sub(r"^```(?:python)?\s*", "", text).strip()
    if text.startswith("```"):
        text = text.split("```", 1)[1]
    text = re.sub(r"```\s*$", "", text).strip()
    return _extract_function(text)


async def _solve_and_check(client: VLLMClient, task: dict[str, Any], sandbox: Sandbox) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = (
        "Write only Python code (no markdown fences) that defines the function "
        f"{task['prompt']}. Do not include tests or prints.",
    )
    try:
        raw = await client.chat(
            [{"role": "user", "content": "Write only Python code (no markdown fences) "
                                        f"that defines the function {task['prompt']} "
                                        "Do not include tests or prints."}],
            temperature=0.2,
            max_tokens=768,
            extra_body={"enable_thinking": False},
        )
    except Exception as exc:
        return {"id": task["id"], "ok": False, "error": f"{type(exc).__name__}: {exc}", "ms": 0}
    code = _clean_code(raw)
    script = "\n".join(["import math", code, *task["tests"], "print('SOLVED')"])
    result = await sandbox.run(script, timeout=25, mem_mb=512)
    ok = result.exit_code == 0 and "SOLVED" in result.stdout
    return {
        "id": task["id"],
        "ok": ok,
        "exit": result.exit_code,
        "stderr": result.stderr[:200],
        "ms": round((time.perf_counter() - started) * 1000),
    }


async def run_bench(client: VLLMClient, sandbox: Sandbox) -> dict[str, Any]:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    wall_started = time.perf_counter()
    outcomes = await asyncio.gather(
        *[_solve_and_check(client, task, sandbox) for task in CODING_TASKS]
    )
    results = []
    for task, outcome in zip(CODING_TASKS, outcomes):
        result = {"id": task["id"], **outcome}
        results.append(result)
        with PROGRESS_FILE.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"ran_at": datetime.now(timezone.utc).isoformat(), **result},
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(f"[bench] {result['id']} ok={result['ok']} exit={result.get('exit')} ms={result['ms']}")
    wall_ms = round((time.perf_counter() - wall_started) * 1000)
    solved = sum(1 for r in results if r["ok"])
    record = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "solved": solved,
        "total": len(results),
        "wall_ms": wall_ms,
        "solved_ids": [r["id"] for r in results if r["ok"]],
        "results": results,
    }
    with HISTORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    args = argparse.ArgumentParser(description="run solved benchmark").parse_args(argv)
    config = Config()
    client = VLLMClient(config)
    sandbox = Sandbox()
    try:
        record = asyncio.run(run_bench(client, sandbox))
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    finally:
        try:
            PAUSE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())