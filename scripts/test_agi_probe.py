from agi_probe import compare, is_holdout, score_answer, PROBES


def test_exact_and_unknown():
    count = next(item for item in PROBES if item["id"] == "novel_count")
    unknown = next(item for item in PROBES if item["id"] == "unknown_mass")
    assert score_answer(count, "there are 8 left")["ok"]
    assert not score_answer(count, "7")["ok"]
    assert score_answer(unknown, "I don't know; no source")["ok"]
    assert not score_answer(unknown, "The mass is 12 kg")["ok"]


def test_holdout_never_trains():
    assert is_holdout(PROBES[0]["prompt"])
    assert not is_holdout("Write a Python function that reverses a string")


def test_compare_forgot():
    before = {
        "score": 0.5,
        "results": [{"id": "a", "ok": True}, {"id": "b", "ok": False}],
    }
    after = {
        "score": 0.5,
        "results": [{"id": "a", "ok": False}, {"id": "b", "ok": True}],
    }
    verdict = compare(before, after)
    assert verdict["verdict"] == "forgot"
    assert "a" in verdict["lost"]


def main():
    for test in (test_exact_and_unknown, test_holdout_never_trains, test_compare_forgot):
        test()
        print(f"{test.__name__}: ok")
    print("agi_probe tests passed")


if __name__ == "__main__":
    main()
