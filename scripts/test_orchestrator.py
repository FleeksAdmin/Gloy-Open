import asyncio
import json
import tempfile
from pathlib import Path

from orchestrator import (
    Config,
    Evaluator,
    MockClient,
    Orchestrator,
    Status,
)


class ScriptedEvaluatorClient:
    def __init__(self, payload):
        self._payload = payload

    async def chat(self, messages, *, temperature, max_tokens, response_format=None):
        return self._payload

    async def stream_chat(self, messages, *, temperature, max_tokens):
        if False:
            yield ""


async def run_mock(answers, scores, prompt="What is 2+2?", retries=2):
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        config = Config(max_retries=retries, repeat_line_limit=4)
        client = MockClient(answers=answers, judge_scores=scores)
        result = await Orchestrator(client, config, work_dir=work_dir).run(prompt)
        buffer_lines = (work_dir / "buffer.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        report_files = list((work_dir / "reports").glob("*.md"))
        return {
            "result": result,
            "buffer": buffer_lines,
            "success": (work_dir / "success.jsonl").exists(),
            "training": (work_dir / "training_queue.jsonl").exists(),
            "report": report_files[0].read_text(encoding="utf-8") if report_files else "",
        }


def test_perfect():
    data = asyncio.run(run_mock(["2 + 2 = 4"], [0.95]))
    result = data["result"]
    assert result.delivered_status == Status.PERFECT
    assert data["success"] and not data["training"]
    assert len(data["buffer"]) == 1
    assert "Delivered answer" in data["report"]


def test_acceptable():
    data = asyncio.run(run_mock(["roughly four"], [0.8]))
    result = data["result"]
    assert result.delivered_status == Status.ACCEPTABLE
    assert data["training"] and not data["success"]


def test_retry_then_perfect():
    data = asyncio.run(run_mock(["wrong", "correct"], [0.2, 0.95]))
    result = data["result"]
    assert result.delivered_status == Status.PERFECT
    assert len(result.attempts) == 2
    assert len(data["buffer"]) == 2
    assert data["success"]


def test_failed_best_effort():
    data = asyncio.run(
        run_mock(["a", "b", "c"], [0.1, 0.3, 0.2], retries=2)
    )
    result = data["result"]
    assert result.delivered_status == Status.FAILED
    assert abs(result.delivered_score - 0.3) < 1e-9
    assert len(result.attempts) == 3
    assert "best-effort" in result.note


def test_anomaly_aborts_stream():
    data = asyncio.run(
        run_mock(
            ["loop\nloop\nloop\nloop\nloop\n", "clean and correct"],
            [0.95],
        )
    )
    result = data["result"]
    first = result.attempts[0]
    assert first.generation.anomaly is not None
    assert first.generation.anomaly.kind == "repeated_line"
    assert result.delivered_status == Status.PERFECT


def test_v11_factuality_gate_caps_score():
    payload = json.dumps(
        {
            "claims": [{"claim": "A", "factuality": 0.5}],
            "factuality": 0.5,
            "rubric_marks": {
                "ambiguity_flag": True,
                "unwarned_hallucination": True,
                "purely_cosmetic_analogy": False,
                "technical_accuracy": True,
            },
            "flaws": ["hallucinated an entity"],
            "critique": "remove the invented entity",
            "critique_prompt": "remove the invented entity",
            "score": 1.0,
            "status": "perfect",
        }
    )
    config = Config()
    evaluation = asyncio.run(
        Evaluator(ScriptedEvaluatorClient(payload), config).evaluate(
            "q", "a", ground_truth={"is_ambiguous": True}
        )
    )
    assert evaluation.factuality == 0.5
    assert "unwarned_hallucination" in evaluation.applied_penalties
    assert evaluation.score <= config.factuality_cap
    assert evaluation.status == Status.FAILED


def test_v11_clean_run_keeps_base():
    payload = json.dumps(
        {
            "factuality": 1.0,
            "rubric_marks": {
                "ambiguity_flag": True,
                "unwarned_hallucination": False,
                "purely_cosmetic_analogy": False,
                "technical_accuracy": True,
            },
            "flaws": [],
            "critique": "",
            "critique_prompt": "",
            "score": 0.82,
            "status": "acceptable",
        }
    )
    config = Config()
    evaluation = asyncio.run(
        Evaluator(ScriptedEvaluatorClient(payload), config).evaluate("q", "a")
    )
    assert abs(evaluation.score - 0.82) < 1e-9
    assert evaluation.status == Status.ACCEPTABLE
    assert evaluation.applied_penalties == []


def main():
    tests = [
        test_perfect,
        test_acceptable,
        test_retry_then_perfect,
        test_failed_best_effort,
        test_anomaly_aborts_stream,
        test_v11_factuality_gate_caps_score,
        test_v11_clean_run_keeps_base,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    print("all tests passed")


if __name__ == "__main__":
    main()
