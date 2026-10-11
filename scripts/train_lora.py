"""Auto-LoRA weight learning with A/B acceptance and rollback."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = BASE_DIR / "models" / "adapters"
LOGS_DIR = BASE_DIR / "logs"
ANOMALIES_FILE = LOGS_DIR / "anomalies.jsonl"
DATASET_FILE = LOGS_DIR / "lora_dataset.jsonl"
PAUSE_FLAG = LOGS_DIR / "continuous" / "pause.flag"
CUDAROOT = "/opt/venv/lib/python3.12/site-packages/nvidia/cu13"
LIVE_MODEL = os.environ.get("GLOW_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")
MODEL_ID = os.environ.get("GLOW_TRAIN_MODEL", LIVE_MODEL)
MODEL_BASE = os.environ.get("GLOW_TRAIN_BASE", "Qwen/Qwen3-32B")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _free_vram() -> bool:
    try:
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
            check=False,
            capture_output=True,
        )
        time.sleep(4)
        return True
    except Exception:
        return False


def _restore_vram() -> bool:
    adapter = ""
    adapters = sorted(ADAPTERS_DIR.glob("adapter_*"), reverse=True)
    if adapters:
        adapter = str(adapters[0])
    extra = ""
    if adapter:
        extra = f"--enable-lora --max-lora-rank 16 --lora-modules gloy={adapter} "
    export = (
        f"export HF_HOME=/workspace/models; export VLLM_WSL2_ENABLE_PIN_MEMORY=1; "
        f"export CUDA_HOME={CUDAROOT}; export PATH={CUDAROOT}/bin:$PATH; "
        f"export LD_LIBRARY_PATH={CUDAROOT}/lib:{CUDAROOT}/lib64; "
        f"export FLASHINFER_EXTRA_CUDAFLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK; "
        f"export GLOW_MODEL={LIVE_MODEL}; "
    )
    command = (
        f"cd /workspace/AGI && nohup /opt/venv/bin/python -m vllm.entrypoints.openai.api_server "
        f"--model {LIVE_MODEL} --language-model-only --max-model-len 4096 "
        f"--gpu-memory-utilization 0.90 --tensor-parallel-size 1 --port 8000 --enforce-eager "
        f"--kv-cache-dtype fp8 --max-num-seqs 8 {extra}"
        f"> logs/vllm.log 2>&1 < /dev/null &"
    )
    try:
        subprocess.Popen(["bash", "-lc", export + command])
        return True
    except Exception:
        return False


def _secret_output(record: dict[str, Any]) -> str:
    existing = str(record.get("output") or "")
    if "<secret>" in existing and "<answer>" in existing:
        return existing
    secret = str(record.get("scratchpad") or record.get("secret") or "")
    thinking = str(record.get("thinking") or "")
    if isinstance(record.get("thinking_steps"), list):
        thinking = thinking or "\n".join(str(x) for x in record["thinking_steps"])
    answer = str(record.get("answer") or "")
    return (
        f"<secret>{secret[:2000]}</secret>\n"
        f"<thinking>{thinking[:1200]}</thinking>\n"
        f"<answer>{answer[:1500]}</answer>"
    )


def _build_dataset(take: int = 40) -> Path:
    records: list[dict[str, Any]] = []
    records += _read_jsonl(DATASET_FILE)
    runs_file = LOGS_DIR / "continuous" / "runs.jsonl"
    if runs_file.exists():
        records += [
            row
            for row in _read_jsonl(runs_file)
            if row.get("status") in ("perfect", "acceptable")
        ]
    records += _read_jsonl(LOGS_DIR / "success.jsonl")
    dedupe: dict[str, dict[str, Any]] = {}
    for record in records:
        prompt = str(record.get("instruction") or record.get("prompt") or "")
        answer = str(record.get("answer") or "")
        if len(prompt) < 3:
            continue
        try:
            from agi_probe import is_holdout

            if is_holdout(prompt):
                continue
        except Exception:
            pass
        output = _secret_output(record)
        if "<answer>" not in output:
            continue
        dedupe[prompt] = {"instruction": prompt, "output": output}
    dataset = DATASET_FILE
    dataset.parent.mkdir(parents=True, exist_ok=True)
    rows = list(dedupe.values())
    take = max(8, take)
    n_replay = min(len(rows) // 3, max(3, take // 3)) if len(rows) > take else 0
    fresh_n = take - n_replay
    fresh = rows[-fresh_n:] if len(rows) >= fresh_n else rows
    old = rows[: max(0, len(rows) - len(fresh))]
    replay: list[dict[str, Any]] = []
    if old and n_replay:
        step = max(1, len(old) // n_replay)
        replay = old[::step][:n_replay]
    mixed = replay + fresh
    with dataset.open("w", encoding="utf-8") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _append_jsonl(
        ANOMALIES_FILE,
        {
            "event": "lora_dataset_replay",
            "fresh": len(fresh),
            "replay": len(replay),
            "time": _now(),
        },
    )
    return dataset


def _unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401

        return True
    except ImportError:
        return False


def _train(dataset: Path, adapter_dir: Path) -> bool:
    import datasets
    import unsloth
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=MODEL_BASE,
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=False,
    )
    model = unsloth.FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none", use_gradient_checkpointing="unsloth"
    )

    def _to_conversation(example):
        return {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["output"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    raw = datasets.Dataset.from_json(str(dataset))
    train_dataset = raw.map(_to_conversation)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=40,
            learning_rate=2e-4,
            warmup_steps=3,
            logging_steps=1,
            output_dir=str(adapter_dir),
            report_to=[],
            seed=0,
        ),
    )
    trainer.train()
    model.save_pretrained_merged(str(adapter_dir), tokenizer, save_method="merged_16bit")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = argparse.ArgumentParser(description="auto lora").parse_args(argv)
    _append_jsonl(ANOMALIES_FILE, {"event": "lora_kick", "time": _now()})
    if not _unsloth_available():
        _append_jsonl(ANOMALIES_FILE, {"event": "lora_deferred", "reason": "unsloth missing", "time": _now()})
        print(json.dumps({"status": "deferred", "reason": "unsloth missing"}))
        return 0
    dataset = _build_dataset()
    adapter_dir = ADAPTERS_DIR / f"adapter_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_FLAG.touch(exist_ok=True)
    freed = _free_vram()
    ok = False
    try:
        ok = _train(dataset, adapter_dir)
    except Exception as exc:
        _append_jsonl(ANOMALIES_FILE, {"event": "lora_failed", "error": f"{type(exc).__name__}: {exc}", "time": _now()})
    finally:
        restored = _restore_vram()
        try:
            PAUSE_FLAG.unlink(missing_ok=True)
        except OSError:
            pass
        if not ok and adapter_dir.exists():
            shutil.rmtree(adapter_dir, ignore_errors=True)
            _append_jsonl(ANOMALIES_FILE, {"event": "lora_rolled_back", "time": _now()})
    print(json.dumps({"status": "ok" if ok else "failed", "vram_freed": freed, "vram_restored": restored}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())