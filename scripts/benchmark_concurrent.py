import concurrent.futures
import json
import os
import time

import requests

ENDPOINT = os.environ.get(
    "GLOW_ENDPOINT", "http://localhost:8000/v1/chat/completions"
)
MODEL = os.environ.get("GLOW_MODEL", "google/gemma-4-12B-it-qat-w4a16-ct")
PROMPT = "Write a detailed explanation of how a transformer neural network works."
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
CONC = int(os.environ.get("CONC", "4"))


def one():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
    }
    start = time.perf_counter()
    response = requests.post(ENDPOINT, json=body, timeout=900)
    response.raise_for_status()
    elapsed = time.perf_counter() - start
    return elapsed, response.json()["usage"]["completion_tokens"]


def main():
    one()
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as pool:
        rows = list(pool.map(lambda _: one(), range(CONC)))
    wall = time.perf_counter() - start
    total = sum(tokens for _, tokens in rows)
    per = [tokens / elapsed for elapsed, tokens in rows]
    result = {
        "model": MODEL,
        "concurrency": CONC,
        "max_tokens": MAX_TOKENS,
        "wall_s": round(wall, 3),
        "total_tokens": total,
        "aggregate_tok_per_s": round(total / wall, 1),
        "per_request_tok_per_s": [round(value, 1) for value in per],
    }
    print(json.dumps(result, indent=2))
    with open("/workspace/AGI/logs/benchmark.log", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
