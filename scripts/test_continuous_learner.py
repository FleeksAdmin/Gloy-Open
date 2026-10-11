import asyncio
import tempfile
from pathlib import Path

from continuous_learner import (
    CoTProcessor,
    PAUSE_FILE,
    Config,
    ContinuousLearner,
    Fact,
    KnowledgeStore,
    MockClient,
    Sandbox,
)


def test_cot_split():
    result = CoTProcessor.split(
        "hello <thinking>plan carefully</thinking> visible answer"
    )
    assert result.hidden == "plan carefully"
    assert "visible answer" in result.visible
    assert "<thinking>" not in result.visible


def test_cot_unclosed():
    result = CoTProcessor.split("lead<thinking>in progress")
    assert result.hidden == "in progress"
    assert result.visible == "lead"


def test_sandbox_ok():
    result = asyncio.run(Sandbox().run("print('hi')"))
    assert result.exit_code == 0
    assert result.stdout.strip() == "hi"


def test_sandbox_blocks_system_deletion():
    result = asyncio.run(Sandbox().run("import os\nos.remove('/etc/passwd')"))
    assert result.exit_code == 124
    assert "BLOCKED" in result.stderr


def test_sandbox_timeout():
    script = "import time\nwhile True:\n    time.sleep(1)"
    result = asyncio.run(Sandbox().run(script, timeout=1))
    assert result.timed_out


def test_knowledge_store_faiss_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        store = KnowledgeStore(Path(tmp) / "knowledge.jsonl")
        store.add(Fact("the sky is blue", "t", 0.9, True))
        store.add(Fact("gpu memory is dithered", "t", 0.8, True))
        hits = store.search("sky color")
        assert hits
        assert hits[0][1] > hits[1][1]


async def _scenario():
    try:
        PAUSE_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    tmp = Path(tempfile.mkdtemp())
    config = Config()
    client = MockClient()
    store = KnowledgeStore(tmp / "knowledge.jsonl")
    learner = ContinuousLearner(
        config,
        client=client,
        kb=store,
        tick=0.01,
        state_dir=tmp / "state",
        use_open=False,
    )
    task = asyncio.create_task(
        learner.run(cycles=50, seed_tasks=["question one", "question two"])
    )
    await asyncio.sleep(0.05)
    learner.pause()
    await asyncio.sleep(0.15)
    paused_done = learner.state.done
    checkpoint = tmp / "state" / "checkpoint.json"
    assert checkpoint.exists(), "checkpoint must be serialized on pause"
    assert paused_done >= 1
    learner.resume()
    await asyncio.sleep(0.1)
    resumed_done = learner.state.done
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return paused_done, resumed_done, checkpoint


def test_interrupt_checkpoint_and_resume():
    paused_done, resumed_done, checkpoint = asyncio.run(_scenario())
    assert resumed_done >= paused_done
    assert checkpoint.stat().st_size > 0


class ScriptedPadClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def chat(self, messages, *, temperature, max_tokens, response_format=None):
        return self._responses.pop(0) if self._responses else "<answer>done</answer>"

    async def stream_chat(self, messages, *, temperature, max_tokens):
        if False:
            yield ""


def test_scratchpad_accumulates_and_returns_visible():
    client = ScriptedPadClient(
        [
            "<scratchpad>first private thought</scratchpad><answer><short/></answer>",
            "<scratchpad>second private thought</scratchpad><answer>Final visible.</answer>",
        ]
    )
    learner = ContinuousLearner(
        Config(), client=client, state_dir=Path(tempfile.mkdtemp())
    )
    result = asyncio.run(learner._solve_with_pad("test task"))
    assert result.steps == 2
    assert "first private thought" in result.pad
    assert "second private thought" in result.pad
    assert result.visible == "Final visible."


def main():
    tests = [
        test_cot_split,
        test_cot_unclosed,
        test_sandbox_ok,
        test_sandbox_blocks_system_deletion,
        test_sandbox_timeout,
        test_knowledge_store_faiss_fallback,
        test_interrupt_checkpoint_and_resume,
        test_scratchpad_accumulates_and_returns_visible,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    print("all continuous_learner tests passed")


if __name__ == "__main__":
    main()