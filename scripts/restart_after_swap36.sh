export GLOW_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
export GLOW_ENDPOINT=http://localhost:8000/v1/chat/completions
mkdir -p /workspace/AGI/logs

# smoke
echo '===SMOKE==='
curl -s --max-time 90 http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.6-35B-A3B-NVFP4","messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":32,"temperature":0}' \
  | head -c 800
echo

# restart apps on the new model name
pkill -f 'gradio_ap[p]' 2>/dev/null || true
pkill -f 'dashboar[d]' 2>/dev/null || true
pkill -f 'continuous_learne[r]' 2>/dev/null || true
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

# tunnels: keep healthy gradio tunnel, recreate dash + vllm
if [ ! -x /usr/local/bin/cloudflared ]; then
  curl -sL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
# drop only dash/vllm tunnels; leave 7860 if already up
pgrep -af 'cloudflared tunnel --url http://localhost:8010' | awk '{print $1}' | xargs -r kill 2>/dev/null || true
pgrep -af 'cloudflared tunnel --url http://localhost:8000' | awk '{print $1}' | xargs -r kill 2>/dev/null || true
sleep 2
rm -f /workspace/AGI/logs/dash_tunnel.log /workspace/AGI/logs/vllm_tunnel.log
nohup /usr/local/bin/cloudflared tunnel --url http://localhost:8010 --protocol http2 --no-autoupdate \
  > /workspace/AGI/logs/dash_tunnel.log 2>&1 &
echo "dash tunnel pid=$!"
nohup /usr/local/bin/cloudflared tunnel --url http://localhost:8000 --protocol http2 --no-autoupdate \
  > /workspace/AGI/logs/vllm_tunnel.log 2>&1 &
echo "vllm tunnel pid=$!"
# ensure gradio tunnel
if ! pgrep -af 'cloudflared tunnel --url http://localhost:7860' >/dev/null; then
  rm -f /workspace/AGI/logs/gradio_tunnel.log
  nohup /usr/local/bin/cloudflared tunnel --url http://localhost:7860 --protocol http2 --no-autoupdate \
    > /workspace/AGI/logs/gradio_tunnel.log 2>&1 &
  echo "gradio tunnel pid=$!"
fi
sleep 18
echo "GRADIO_URL=$(grep -oaE 'https://[a-z0-9-]+\.trycloudflare\.com' /workspace/AGI/logs/gradio_tunnel.log 2>/dev/null | grep -v api | head -1)"
echo "DASH_URL=$(grep -oaE 'https://[a-z0-9-]+\.trycloudflare\.com' /workspace/AGI/logs/dash_tunnel.log 2>/dev/null | grep -v api | head -1)"
echo "VLLM_URL=$(grep -oaE 'https://[a-z0-9-]+\.trycloudflare\.com' /workspace/AGI/logs/vllm_tunnel.log 2>/dev/null | grep -v api | head -1)"
echo '===GPU==='
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo DONE
