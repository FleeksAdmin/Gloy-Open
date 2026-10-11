"""HardVerifier: executable code check + evidence-based fact verification."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Sequence

from reasoning_search import strip_thinking


def _extract_python(text: str) -> str:
    code = strip_thinking(text)
    code = re.sub(r"^```(?:python)?\s*", "", code).strip()
    if code.startswith("```"):
        code = code.split("```", 1)[1]
    code = re.sub(r"```\s*$", "", code).strip()
    lines = code.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:async\s+)?def\s+\w+", line):
            start = index
            break
    if start is None:
        return code
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
    return "\n".join(out)


class HardVerifier:
    @staticmethod
    def exec_valid(answer: str) -> bool:
        code = _extract_python(answer)
        if not code:
            return False
        try:
            compile(code, "<generated>", "exec")
            return True
        except SyntaxError:
            return False

    @staticmethod
    def consensus_overlap(answer: str, evidence: Sequence[str]) -> float:
        from reasoning_search import overlap

        if not evidence:
            return 0.0
        return max(overlap(answer, item) for item in evidence if item)

    async def verify_fact(
        self,
        claim: str,
        query: str,
        use_open: bool = True,
    ) -> dict[str, Any]:
        evidence: list[str] = []
        urls: list[str] = []
        if use_open:
            try:
                from open_sources import OpenSources

                docs = await OpenSources().search(query, per_source=1)
                evidence = [f"{d.title}: {d.text}" for d in docs if d.text]
                urls = [d.url for d in docs if d.url]
            except Exception:
                pass
        score = self.consensus_overlap(claim, evidence)
        return {
            "verified": score >= 0.3,
            "score": round(score, 3),
            "evidence_checked": len(evidence),
            "urls": urls,
        }

    def make(self) -> Callable[[str, str], bool]:
        return lambda answer, kind: self.exec_valid(answer) if kind == "code" else True


if __name__ == "__main__":
    v = HardVerifier()
    ok = v.exec_valid("```python\ndef f(x):\n    return x + 1\n```")
    bad = v.exec_valid("def f(x):\nreturn x")
    print("exec_valid ok:", ok, "| bad:", bad)
    loop = asyncio.new_event_loop()
    res = loop.run_until_complete(v.verify_fact("black hole forms from collapsed star", "black hole formation", use_open=False))
    print("fact verify (no net):", res)