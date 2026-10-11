#!/bin/bash
export GLOW_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
export GLOW_ENDPOINT=http://localhost:8000/v1/chat/completions
export GLOW_SEARCH=1
export GLOW_EFFORT=medium
export GLOW_LORA_AUTO=1
mkdir -p /workspace/AGI/logs/continuous
python3 - <<'PY'
import json, time
from pathlib import Path
Path("/workspace/AGI/logs/continuous/lora_last.json").write_text(json.dumps({"ts": time.time()}), encoding="utf-8")
PY
pkill -f 'gradio_ap[p]' 2>/dev/null || true
pkill -f 'continuous_learne[r]' 2>/dev/null || true
pkill -f 'dashboard.p[y]' 2>/dev/null || true
sleep 2
cd /workspace/AGI/scripts
nohup /opt/venv/bin/python dashboard.py --host 0.0.0.0 --port 8010 > /workspace/AGI/logs/dashboard.log 2>&1 &
echo "dashboard pid=$!"
nohup /opt/venv/bin/python continuous_learner.py --cycles 0 --tick 2.5 > /workspace/AGI/logs/learner.log 2>&1 &
echo "learner pid=$!"
setsid /opt/venv/bin/python gradio_app.py > /workspace/AGI/logs/gradio.log 2>&1 < /dev/null &
echo "gradio pid=$!"
sleep 8
curl -s -o /dev/null -w 'gradio=%{http_code}\n' --max-time 5 http://localhost:7860/
curl -s -o /dev/null -w 'dash=%{http_code}\n' --max-time 5 http://localhost:8010/
curl -s -o /dev/null -w 'vllm=%{http_code}\n' --max-time 5 http://localhost:8000/v1/models
echo DONE
