import tempfile
from pathlib import Path

import cognition
import goals


def test_profile_physics_and_tom():
    phys = cognition.profile("What happens to the ball after collision if gravity doubles?")
    assert phys.physics
    tom = cognition.profile("Why did she say that — sarcasm or real motive?")
    assert tom.tom
    vague = cognition.profile("помоги мне открыть ресторан")
    assert vague.vague_goal


def test_assess_unknown():
    meta = cognition.assess("I don't know the melting point, need to search", "maybe 100")
    assert meta.unknowns
    assert meta.need_web


def test_goals_roundtrip(tmp_path: Path):
    goals.GOALS_FILE = tmp_path / "goals.json"
    created = goals.add_goal("open a small restaurant", horizon="months")
    assert created["id"]
    assert len(created["tasks"]) >= 5
    work = goals.next_work()
    assert work and "restaurant" in work["prompt"].lower()
    goals.complete_task(work["goal_id"], work["task_id"], ok=True)
    leftover = [t for t in goals.list_active()[0]["tasks"] if t["status"] != "done"]
    assert leftover


def test_extract_steps():
    text = "1. Rent a kitchen\n2. Get a license\n- Hire a cook"
    assert goals.extract_steps(text) == ["Rent a kitchen", "Get a license", "Hire a cook"]


def main():
    tests = [
        test_profile_physics_and_tom,
        test_assess_unknown,
        test_extract_steps,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    with tempfile.TemporaryDirectory() as folder:
        test_goals_roundtrip(Path(folder))
    print("test_goals_roundtrip: ok")
    print("cognition tests passed")


if __name__ == "__main__":
    main()
