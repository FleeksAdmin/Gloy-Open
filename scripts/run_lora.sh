export GLOW_TRAIN_BASE=${GLOW_TRAIN_BASE:-Qwen/Qwen3-32B}
export GLOW_MODEL=${GLOW_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
export HF_HOME=/workspace/models
# Train on the 4-bit *base* (fits 24GB after vLLM is stopped). Dataset is <secret>/<thinking>/<answer>.
# Serving after train is always the live Qwen3.6; adapter is attached if vLLM accepts it.
touch /workspace/AGI/logs/continuous/pause.flag
cd /workspace/AGI/scripts
if [ -x /workspace/venv_lora/bin/python ]; then
  PY=/workspace/venv_lora/bin/python
else
  PY=/opt/venv/bin/python
fi
setsid $PY train_lora.py > /workspace/AGI/logs/lora_run.log 2>&1 < /dev/null &
echo "lora spawned pid=$!"
