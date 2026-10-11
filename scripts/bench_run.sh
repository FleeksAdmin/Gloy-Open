export GLOW_MODEL=${GLOW_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
export HF_HOME=/workspace/models
touch /workspace/AGI/logs/continuous/pause.flag
cd /workspace/AGI/scripts
setsid /opt/venv/bin/python -u solved_bench.py > /workspace/AGI/logs/bench_run.log 2>&1 < /dev/null &
echo "bench pid=$!"