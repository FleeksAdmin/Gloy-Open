import asyncio
from pathlib import Path

from hard_verify import HardVerifier
from reasoning_search import ReasoningSearch, overlap, self_consistent


class ScriptedClient:
    def __init__(self, answers):
        self._answers = list(answers)

    async def chat(self, messages, *, temperature, max_tokens, response_format=None, extra_body=None):
        return self._answers.pop(0) if self._answers else ""


def test_exec_valid():
    v = HardVerifier()
    assert v.exec_valid("```python\ndef f(x):\n    return x\n```")
    assert not v.exec_valid("def f(x):\nreturn x")


def test_overlap_and_consistency():
    a = "the capital of france is paris"
    b = "paris is the capital of france"
    assert overlap(a, b) > 0.5
    best, score = self_consistent([a, b, "totally unrelated words here"])
    assert best in (a, b)
    assert score > 0.0


def test_reasoning_search_self_consistency():
    client = ScriptedClient([
        "def reverse_string(s):\n    return s[::-1]",
        "def reverse_string(s):\n    return s[::-1]",
        "banana bread recipe",
    ])
    result = asyncio.run(ReasoningSearch(client).solve("reverse a string", samples=3))
    assert "reverse_string" in result["chosen"]
    assert result["consistency"] > 0.0


def main():
    tests = [test_exec_valid, test_overlap_and_consistency, test_reasoning_search_self_consistency]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    print("all advanced-module tests passed")


if __name__ == "__main__":
    main()