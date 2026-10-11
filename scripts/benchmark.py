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
RUNS = int(os.environ.get("RUNS", "3"))


def payload(stream=False, max_tokens=None):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "max_tokens": max_tokens or MAX_TOKENS,
        "ignore_eos": True,
        "stream": stream,
    }


def run_once(max_tokens=None):
    start = time.perf_counter()
    response = requests.post(ENDPOINT, json=payload(False, max_tokens), timeout=900)
    response.raise_for_status()
    elapsed = time.perf_counter() - start
    usage = response.json()["usage"]
    return elapsed, usage["completion_tokens"], usage["prompt_tokens"]


def run_stream(max_tokens):
    start = time.perf_counter()
    ttft = None
    tokens = 0
    with requests.post(
        ENDPOINT, json=payload(True, max_tokens), stream=True, timeout=900
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode() if isinstance(line, bytes) else line
            if not text.startswith("data:"):
                continue
            body = text[5:].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                tokens += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    return ttft, total, tokens


def main():
    print("warmup")
    run_once()
    results = {"model": MODEL, "max_tokens": MAX_TOKENS, "runs": []}
    for index in range(RUNS):
        elapsed, completion, prompt = run_once()
        tps = completion / elapsed
        results["runs"].append(
            {
                "elapsed_s": round(elapsed, 3),
                "completion_tokens": completion,
                "prompt_tokens": prompt,
                "tok_per_s": round(tps, 1),
            }
        )
        print(
            f"run {index + 1}: {completion} tok / {elapsed:.2f}s = {tps:.1f} tok/s"
        )
    avg = sum(r["tok_per_s"] for r in results["runs"]) / len(results["runs"])
    results["avg_tok_per_s"] = round(avg, 1)

    ttft, total, tokens = run_stream(MAX_TOKENS)
    tpot = (total - ttft) / max(1, tokens - 1)
    results["stream"] = {
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "tokens": tokens,
        "tok_per_s": round(tokens / total, 1),
        "tpot_ms": round(tpot * 1000, 1),
    }
    print(
        f"stream: TTFT {ttft*1000:.0f} ms, {tokens} tok / {total:.2f}s = "
        f"{tokens/total:.1f} tok/s, TPOT {tpot*1000:.1f} ms"
    )
    print(json.dumps(results, indent=2))
    with open("/workspace/AGI/logs/benchmark.log", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
