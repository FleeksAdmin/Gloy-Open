pkill -f 'gradio_ap[p]' 2>/dev/null || true
sleep 2
export GLOW_ENDPOINT=http://localhost:8000/v1/chat/completions
export GLOW_MODEL=${GLOW_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
cd /workspace/AGI/scripts
setsid /opt/venv/bin/python gradio_app.py > /workspace/AGI/logs/gradio.log 2>&1 < /dev/null &
echo "gradio restarted pid=$!"
sleep 8
curl -s -o /dev/null -w 'local=%{http_code}\n' --max-time 5 http://localhost:7860/
