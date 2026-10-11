pkill -f 'continuous_learne[r]' 2>/dev/null || true
sleep 2
export GLOW_MODEL=${GLOW_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
export GLOW_SEARCH=1
export GLOW_EFFORT=medium
export GLOW_LORA_AUTO=1
export HF_HOME=/workspace/models
cd /workspace/AGI/scripts
setsid /opt/venv/bin/python continuous_learner.py --cycles 0 --tick 2.5 > /workspace/AGI/logs/learner.log 2>&1 < /dev/null &
echo "learner restarted pid=$!"