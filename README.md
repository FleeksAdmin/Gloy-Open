# Gloy Open 🍇

> **Autonomous Self-Improving AI Agent Framework with DeepThink Reasoning and 24/7 Continuous Learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Engine: vLLM](https://img.shields.io/badge/Engine-vLLM-green.svg)](https://github.com/vllm-project/vllm)
[![Telegram: @gloy_dev](https://img.shields.io/badge/Telegram-@gloy__dev-2CA5E0?logo=telegram&logoColor=white)](https://t.me/gloy_dev)
[![Rent GPU: @bi_stan](https://img.shields.io/badge/Rent%20GPU-@bi__stan-76B900?logo=nvidia&logoColor=white)](https://t.me/bi_stan)

> 📢 **Stay updated & support the project!**  
> • Join our official Telegram channel **[@gloy_dev](https://t.me/gloy_dev)** for updates, beta tests, and the upcoming Gloy Code release.  
> • If you find this project exciting, please give us a **⭐ Star** on GitHub!

**Gloy Open** is an open-source agentic architecture built on top of high-performance open-weight LLMs (defaulting to `nvidia/Qwen3.6-35B-A3B-NVFP4` or `Qwen/Qwen3-32B` via vLLM). It wraps foundational models in a closed-loop self-refinement cycle: deep multi-step thinking, strict programmatic judging, anomaly detection, open-source research consensus, sandboxed code execution, and 24/7 background self-training via Auto-LoRA.

---

## 🔬 How It Works & Core Principles

Most systems treat LLMs as static black boxes. **Gloy Open** treats the base model as an active cognitive kernel operating inside a closed feedback and self-evolution loop:

1. **Dual-Phase Cognitive Separation (DeepThink)**:
   - When given a complex prompt, the agent does not immediately generate the public answer.
   - It first triggers a private, unconstrained scratchpad pass (`<secret>` / native `<think>`), generating hypotheses, exploring contradictions, and testing logical forks.
   - Only after this internal draft is complete does the agent synthesize a clean, structured step-by-step reasoning trace (`<thinking>`) and the definitive solution (`<answer>`).

2. **Programmatic Self-Critique & Verification**:
   - Every solution is audited at zero temperature ($T=0$) by an objective judge.
   - The judge decomposes the output into verifiable claims, scores factual accuracy, and applies strict penalties for ungrounded assertions or vague hand-waving.
   - If the score falls below threshold ($<0.75$), the agent automatically enters a critique-and-retry loop (up to 2 iterations).

3. **Continuous Background Self-Study (24/7 Learner)**:
   - When no user requests are incoming, the agent does not sit idle.
   - It continuously drains a task queue, rotates through a multi-domain curriculum (algorithms, math, system architecture, physics), formulates hypotheses, executes sandboxed code experiments, and searches open technical archives (Wikipedia, arXiv, GitHub, Stack Exchange).
   - High-quality, verified facts and problem-solution pairs are consolidated into FAISS vector memory.

4. **Autonomous LoRA Factory with A/B Verification & Rollback**:
   - **The Challenge**: Autonomous weight adaptation risks catastrophic forgetting or model degradation.
   - **Our Solution**:
     - Once at least 10 high-scoring problem solutions are collected, the agent triggers an automated fine-tuning cycle (via Unsloth / PEFT).
     - **Pre-Training Baseline**: Before applying new adapter weights, baseline accuracy is measured across a suite of calibration problems and held-out probes (`scripts/agi_probe.py`).
     - **Post-Training A/B Verification**: The candidate LoRA adapter is tested against the baseline.
     - **Automatic Rollback**: If performance drops (average delta $\le -0.05$ or any single task degradation $> 0.20$), the adapter is **instantly rejected and rolled back**, and the incident is logged to anomaly records (`anomalies.jsonl`). Only adapters that improve or preserve capabilities are committed.
   - *(Continuous enhancement)*: We are actively expanding this safety gate to include multi-turn conversational regression tests and wider domain hold-outs.

---

## 💡 Why Is This Useful?

- **Zero Human-in-the-Loop Maintenance**: The system autonomously refines its reasoning capabilities overnight without needing data scientists to curate datasets or trigger manual training pipelines.
- **Auditable & Safe Reasoning**: Full separation of thoughts, transparent judge logs, and held-out probe verification prevent the agent from silently degrading or drifting into hallucinations.
- **Enterprise & Local-First**: Runs entirely on self-hosted infrastructure via vLLM with no dependency on proprietary external LLM providers.

---

## ✨ Key Features

- 🧠 **Dual-Phase DeepThink**:
  Separates internal raw scratchpad reasoning (`<secret>` / native `<think>`) from structured user-facing step-by-step thinking (`<thinking>`) and final answers (`<answer>`). Configurable effort levels (low to max).
- 🔄 **24/7 Continuous Learner**:
  Autonomous background self-study loop when user traffic is idle. Decomposes tasks, explores curriculum problems, verifies facts, and commits verified discoveries to memory.
- 🎯 **Fail-Gated Programmatic Evaluator (T=0)**:
  Answers are broken down into atomic claims and judged strictly against factuality thresholds. Penalizes hallucinations, missing ambiguity warnings, and empty analogies. Retries with critique prompts if scores fall below threshold.
- 🌐 **Multi-Source Autonomous Research & RAG**:
  Queries Wikipedia, arXiv, GitHub, Stack Exchange, and web search engines with automated reliability classification (Official > Authoritative > Community > Forum) and consensus verification before inserting into FAISS/numpy memory.
- 🧪 **Sandboxed Code Execution**:
  Validates code solutions by running them in isolated sub-processes with CPU/RAM/process limits and execution timeouts.
- 🧬 **Autonomous LoRA Factory with Rollback**:
  Gathers high-quality successful traces, frees VRAM during idle cycles, trains LoRA adapters (via Unsloth/PEFT), runs A/B benchmark evaluation, and commits or rolls back weights automatically.
- 📊 **Held-Out AGI Probe Benchmarks**:
  Zero-leakage test suite (`scripts/agi_probe.py`) evaluating non-memorized reasoning, abduction, intuitive physics, Theory of Mind, and logic with a strict `grew / flat / forgot` verdict to safeguard against catastrophic forgetting.
- 🖥️ **Command Center Dashboard**:
  FastAPI + modern responsive Web UI featuring real-time SSE streaming, latency & token stats, score history charts, and live task queues.

---

## 🏗️ Architecture Overview

```text
[ User / Queue ]
       │
       ▼
[ Orchestrator & DeepThink ] ───► [ Secret Scratchpad (<think>) ]
       │                                     │
       ▼                                     ▼
[ Anomaly Detector ]              [ Public Reasoning (<thinking>) ]
       │                                     │
       ▼                                     ▼
[ Evaluator / Judge (T=0) ] ◄──── [ Sandboxed Verification / Web RAG ]
       │
       ├─ Score >= 0.90 ──────────► [ Output to User & Memory ]
       ├─ 0.75 <= Score < 0.90 ───► [ Queued for Training ]
       └─ Score < 0.75 ───────────► [ Critique & Auto-Retry Loop ]
                                             │
                                    (Background 24/7)
                                             ▼
                                  [ Auto-LoRA Fine-Tuning ]
```

---

## 🚀 Quick Start

### 1. Requirements

- Linux or Windows (WSL2 recommended for vLLM)
- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (for local vLLM 32B/35B quant models) or access to an OpenAI-compatible endpoint.

### 2. Installation

```bash
git clone https://github.com/FleeksAdmin/Gloy-Open.git
cd Gloy-Open
pip install -r requirements.txt
```

### 3. Launching vLLM Engine

Start an OpenAI-compatible vLLM inference server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --language-model-only \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

### 4. Running the Agent Pipeline

Test a single prompt through the self-checking orchestrator:

```bash
cd scripts
python orchestrator.py "Explain the difference between a semaphore and a mutex."
```

Offline/Mock mode (without GPU):
```bash
python orchestrator.py --mock "Test reasoning"
```

### 5. Launch the Command Center Dashboard

```bash
python dashboard.py --port 8010
```
Open `http://localhost:8010` in your browser.

### 6. Start the 24/7 Continuous Learner

```bash
python continuous_learner.py --cycles 0 --tick 2.5
```

---

## 📁 Repository Structure

```text
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── vault/
│   └── identity.json           # Agent persistent identity & metadata
├── logs/                       # Training logs, trajectories, and datasets
└── scripts/
    ├── orchestrator.py         # Main execution pipeline, judge & retry engine
    ├── continuous_learner.py   # 24/7 autonomous learning daemon
    ├── deepthink.py            # Dual-phase secret & public chain-of-thought
    ├── cognition.py            # Cognitive evaluation checklists (physics, ToM, etc.)
    ├── open_sources.py         # Multi-platform research retriever
    ├── source_reliability.py   # Source credibility classifier
    ├── research_agent.py       # Consensus-driven research agent
    ├── hard_verify.py          # Sandboxed code execution validator
    ├── train_lora.py           # Auto-LoRA fine-tuning and A/B verification
    ├── agi_probe.py            # Held-out progress probe suite
    ├── grow.py                 # Growth vs Forget evaluation pipeline
    ├── solved_bench.py         # Coding benchmark suite
    ├── dashboard.py            # FastAPI dashboard server
    └── dashboard.html          # Interactive control center UI
```

---

## 🔒 Safety & Responsible Use

Gloy Open executes self-generated code and conducts automated web lookups. In production environments, sandbox limits (`resource.RLIMIT_AS`, `RLIMIT_CPU`, timeout restrictions) should be enforced strictly.

---

## ⚡ Hardware & GPU Compute

Looking for reliable, cost-effective GPU compute to run **Gloy Open**, large **vLLM** workloads, or high-throughput **Unsloth/LoRA** fine-tuning?

We offer dedicated, high-performance GPU instances at **significantly below-market pricing** compared to mainstream cloud platforms (Vast.ai, RunPod), backed by flexible conditions and personal support.

### Available Accelerators:
- 🟢 **NVIDIA GeForce RTX 5090** (32 GB) & **RTX 5090 Laptop** (24 GB)
- 🟢 **NVIDIA GeForce RTX 4090** (24 GB)
- 🟢 **NVIDIA GeForce RTX 3090** (24 GB)
- 🟢 **NVIDIA RTX A5000** (24 GB)
- 🟢 **NVIDIA GeForce RTX 5070**
- *...and other tailored multi-GPU configurations upon request.*

💬 **To check current node availability, get customized pricing, or reserve compute:**  
Contact directly on Telegram: **[@bi_stan](https://t.me/bi_stan)**

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
