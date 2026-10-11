"""Memory Vault: persistent identity + snapshots so the agent survives restarts."""

from __future__ import annotations

import json
import shutil
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class MemoryVault:
    def __init__(
        self,
        state_dir: Path,
        vault_dir: Path,
        remote_dir: Path | None = None,
    ):
        self._state_dir = Path(state_dir)
        self._vault_dir = Path(vault_dir)
        self._remote_dir = Path(remote_dir) if remote_dir else None
        self._identity_file = self._vault_dir / "identity.json"
        self._vault_dir.mkdir(parents=True, exist_ok=True)

    def identity(self) -> dict[str, Any]:
        if self._identity_file.exists():
            try:
                return json.loads(self._identity_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        record = {
            "agent_id": uuid.uuid4().hex,
            "name": "gloy-open",
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "note": "self-improving agent; memory survives resets",
        }
        self._identity_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return record

    def snapshot(self, label: str = "auto") -> Path:
        self.identity()
        self._vault_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self._vault_dir / f"memory_{label}_{ts}.tar.gz"
        if self._state_dir.exists():
            with tarfile.open(path, "w:gz") as tar:
                tar.add(self._state_dir, arcname="continuous")
        if self._remote_dir is not None:
            try:
                self._remote_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, self._remote_dir / path.name)
            except OSError:
                pass
        return path

    def list_snapshots(self) -> list[Path]:
        return sorted(self._vault_dir.glob("memory_*.tar.gz"), reverse=True)

    def restore_latest(self) -> Path | None:
        latest = self.list_snapshots()
        if not latest:
            return None
        path = latest[0]
        self._state_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                if member.name.startswith("continuous/"):
                    member.name = member.name[len("continuous/"):]
                    tar.extract(member, self._state_dir)
        return path

    def prune(self, keep: int = 10) -> int:
        removed = 0
        for path in self.list_snapshots()[keep:]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="memory vault")
    parser.add_argument("--state", default=str(LOGS_DIR / "continuous"))
    parser.add_argument("--vault", default=str(LOGS_DIR / "continuous" / "vault"))
    parser.add_argument("--remote")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args(argv)
    vault = MemoryVault(
        Path(args.state),
        Path(args.vault),
        Path(args.remote) if args.remote else None,
    )
    vault.identity()
    if args.restore:
        restored = vault.restore_latest()
        print(json.dumps({"restored": str(restored) if restored else None}))
        return 0
    if args.snapshot:
        path = vault.snapshot()
        print(json.dumps({"snapshot": str(path), "pruned": vault.prune()}))
        return 0
    print(json.dumps({"identity": vault.identity(), "snapshots": [str(p) for p in vault.list_snapshots()]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())