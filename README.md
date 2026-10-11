# Gloy Open 🍇

> **Autonomous Self-Improving AI Agent Framework with DeepThink Reasoning and 24/7 Continuous Learning.**

[**Join our Telegram Channel**](https://t.me/gloy_dev) • [**Star on GitHub**](https://github.com/FleeksAdmin/Gloy-Open) • [**License: MIT**](LICENSE)

---

> 📢 **Join our Community!**
> All latest updates, closed beta tests of new models, architecture discussions, and the upcoming **Gloy Code** release are announced first in our Telegram channel: **[t.me/gloy_dev](https://t.me/gloy_dev)**.
>
> 🌟 **If you find this project interesting, please consider giving it a Star on GitHub!** It really helps open-source development.

---

## ⚡ What is Gloy Open?

Most modern "AI agents" are thin wrappers around proprietary commercial APIs. They do not learn from their own mistakes, cannot independently verify facts, and discard all accumulated experience once the prompt session ends.

**Gloy Open** takes a radically different approach. It is a full closed-loop, **autonomous agentic framework** built on top of high-performance open-weight LLMs (defaulting to `nvidia/Qwen3.6-35B-A3B-NVFP4` or `Qwen/Qwen3-32B` powered by **vLLM**).

The agent pairs deep test-time reasoning (*DeepThink*), strict programmatic fact-checking with zero-temperature judging (T=0), sandboxed code execution, and **24/7 background self-training via autonomous LoRA weight adaptation (Auto-LoRA)** without human intervention.

---

## 🎯 Why Gloy Open?

| Feature | Standard AI Agents / GPT Wrappers | Gloy Open 🍇 |
|---|---|---|
| **Reasoning Architecture** | Single-pass prompt or shallow ReAct | **Dual-Phase DeepThink**: private scratchpad (`<secret>`) ➔ structured public reasoning (`<thinking>`) |
| **Learning & Adaptation** | Static weights, prompt context only | **Autonomous 24/7 LoRA Loop**: collects verified successes and bakes patterns into weights |
| **Quality Control (Judge)** | None or uncalibrated self-critique | **Programmatic Judge (T=0)**: claim extraction, penalty gating for hallucinations, auto-retry loop |
| **Research & Fact-Checking**| Naive search engine queries | Multi-source retrieval (Wikipedia, arXiv, GitHub, DDG) with **source credibility scoring** |
| **Code Execution** | Unchecked or unconstrained | **Guarded Sandbox**: strict CPU/RAM/process limits, timeouts, assert testing |
| **Knowledge Preservation** | Uncontrolled drift | **Held-Out AGI Probes**: 8 non-leaked tasks ensuring zero catastrophic forgetting |

---

## 🚀 Performance & Speed

- **High-Throughput vLLM Engine**: Leveraging MoE architectures (`Qwen3.6-35B-A3B-NVFP4` activates ~3B parameters per token) with FP8 KV cache delivers Time-To-First-Token (TTFT) around **80–150 ms** and sustained generation speeds of **40–80+ tokens/sec** on a single GPU (RTX 3090 / 4090 / 5090).
- **Instant Preemption for User Requests**: When an interactive user query arrives, the background self-learning daemon pauses immediately (`pause.flag`), releasing 100% of compute and VRAM to the user.
- **Asynchronous Token Streaming**: Full Server-Sent Events (SSE) support with stage decomposition (`temp_start` ➔ `tokens` ➔ `generated` ➔ `done`) in the web dashboard.

---

## 🧠 Architecture Lifecycle

```text
[ User Prompt / Curriculum Task ]
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  1. DeepThink Reasoning Engine                              │
│     ├─ Private native scratchpad: <secret> / <think>        │
│     └─ Clean user-facing trace: <thinking>                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Autonomous Research & Sandbox (if required)             │
│     ├─ Open Sources (arXiv, GitHub, Wikipedia, DDG)         │
│     ├─ Source Reliability Classifier (authority weighting)  │
│     └─ Sandboxed Code Execution (isolated test runner)      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Programmatic Evaluator (Judge, T=0)                     │
│     ├─ Atomic claim extraction                              │
│     ├─ Factuality verification against ground-truth/context │
│     └─ Penalties: unwarned hallucination, empty analogies   │
└─────────────┬───────────────────────────────┬───────────────┘
              │                               │
      Score >= 0.90                   Score < 0.75
              │                               │
              ▼                               ▼
      [ Deliver Output ]              [ Critique Prompt ]
      [ Store to FAISS KB ]           [ Auto-Retry Attempt ]
              │
              ▼ (when 10+ traces collected)
┌─────────────────────────────────────────────────────────────┐
│  4. Background Auto-LoRA Factory (24/7 Learner)             │
│     ├─ Suspend vLLM & free VRAM                             │
│     ├─ Train LoRA adapter via Unsloth / PEFT                │
│     ├─ A/B benchmark evaluation against held-out prompts    │
│     └─ Commit adapter or trigger automatic rollback         │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Quick Start & Setup

### 1. Clone Repository

```bash
git clone https://github.com/FleeksAdmin/Gloy-Open.git
cd Gloy-Open
pip install -r requirements.txt
```

### 2. Launch Local vLLM Inference Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --language-model-only \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

*(You can point to any existing OpenAI-compatible endpoint using `export GLOW_ENDPOINT="http://localhost:8000/v1/chat/completions"`).*

### 3. Launch Command Center Dashboard

```bash
cd scripts
python dashboard.py --port 8010
```
Open **`http://localhost:8010`** in your browser to access the real-time telemetry, queue inspector, and inference workspace.

### 4. Single-Prompt Orchestrator Test

```bash
cd scripts
python orchestrator.py "Explain the difference between a semaphore and a mutex."
```

*Offline mode (no GPU required):*
```bash
python orchestrator.py --mock "Offline pipeline verification"
```

### 5. Start Autonomous 24/7 Learning Daemon

```bash
cd scripts
python continuous_learner.py --cycles 0 --tick 2.5
```

---

## 📂 Repository Structure

```text
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignore rules for logs and model checkpoints
├── vault/
│   └── identity.json            # Agent persistent identity & metadata
├── logs/                        # Training queues and runtime buffers
└── scripts/
    ├── orchestrator.py          # Core pipeline, stream anomaly detection & judge
    ├── continuous_learner.py    # 24/7 autonomous self-study daemon
    ├── deepthink.py             # Dual-phase secret scratchpad & public trace
    ├── cognition.py             # Cognitive checklists (physics, ToM, planning)
    ├── open_sources.py          # Multi-platform research retriever
    ├── source_reliability.py    # Source credibility classifier
    ├── research_agent.py        # Consensus-driven research agent
    ├── hard_verify.py           # Sandboxed code execution validator
    ├── train_lora.py            # Auto-LoRA fine-tuning and A/B verification
    ├── agi_probe.py             # 8 held-out progress probe benchmarks
    ├── grow.py                  # Growth vs forgetting evaluation
    ├── solved_bench.py          # Coding benchmark suite (23 tasks)
    ├── dashboard.py             # FastAPI dashboard backend
    └── dashboard.html           # Command center Web UI
```

---

## 🗺️ Roadmap

- [x] Autonomous agentic orchestration & dual-phase DeepThink
- [x] Strict programmatic evaluator (Judge T=0) with critique retries
- [x] vLLM integration and 24/7 background learning loop
- [x] Multi-source research & RAG with credibility weighting
- [ ] **Gloy Code**: Dedicated agentic CLI environment for software engineering
- [ ] Hosted API access with expanded foundation models
- [ ] Multimodal reasoning support (diagrams and system architecture analysis)

---

## 💜 Community & Support

- 💬 **Telegram:** [t.me/gloy_dev](https://t.me/gloy_dev)
- 🐛 **Issues & Feedback:** Submit an [Issue](https://github.com/FleeksAdmin/Gloy-Open/issues) or Pull Request!
- ⭐ **Don't forget to star the repository if you enjoy our work!**
