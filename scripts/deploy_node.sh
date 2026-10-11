#!/bin/bash
set -e
DLOG=/tmp/deploy.log
mkdir -p /workspace/AGI/logs/continuous
export HF_HOME=/workspace/models
exec >"$DLOG" 2>&1
echo "=== deploy start $(date -u) ==="
command -v aria2c >/dev/null 2>&1 || { apt-get update -qq; apt-get install -y -qq aria2; }

mkdir -p /tmp/w1
/opt/venv/bin/pip install --dry-run --ignore-installed --report /tmp/r1.json \
  vllm transformers huggingface_hub accelerate hf-transfer bitsandbytes 2>/dev/null | tail -1 || true
python3 - <<'PY'
import json
j = json.load(open('/tmp/r1.json'))
urls = [i['download_info']['url'] for i in j.get('install', []) if i.get('download_info')]
open('/tmp/u1.txt', 'w').write('\n'.join(urls))
print('core wheels:', len(urls))
PY
cd /tmp/w1
aria2c --input-file=/tmp/u1.txt -j 6 -x 16 -s 16 -c --console-log-level=error --dir=/tmp/w1 >/dev/null 2>&1 || true
/opt/venv/bin/pip install --no-index --find-links /tmp/w1 \
  vllm transformers huggingface_hub accelerate hf-transfer bitsandbytes 2>&1 | tail -1

mkdir -p /tmp/w2
/opt/venv/bin/pip install --dry-run --ignore-installed --report /tmp/r2.json \
  ddgs faiss-cpu 2>/dev/null | tail -1 || true
python3 - <<'PY'
import json
j = json.load(open('/tmp/r2.json'))
urls = [i['download_info']['url'] for i in j.get('install', []) if i.get('download_info')]
open('/tmp/u2.txt', 'w').write('\n'.join(urls))
print('extras wheels:', len(urls))
PY
cd /tmp/w2
aria2c --input-file=/tmp/u2.txt -j 4 -x 12 -s 12 -c --console-log-level=error --dir=/tmp/w2 >/dev/null 2>&1 || true
/opt/venv/bin/pip install --no-index --find-links /tmp/w2 ddgs faiss-cpu 2>&1 | tail -1 || true

C=/opt/venv/lib/python3.12/site-packages/nvidia/cu13
mkdir -p "$C/lib64/stubs"
[ -e "$C/lib64/libcudart.so" ] || ln -sf "$C/lib/libcudart.so.13" "$C/lib64/libcudart.so"
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -e "$C/lib64/stubs/libcuda.so" ]; then
  ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$C/lib64/stubs/libcuda.so"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/models
export GLOW_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
/opt/venv/bin/hf download nvidia/Qwen3.6-35B-A3B-NVFP4 2>&1 | tail -1

export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME="$C"
export PATH="$C/bin:$PATH"
export LD_LIBRARY_PATH="$C/lib:$C/lib64"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
cd /workspace/AGI
rm -f logs/vllm.log
nohup /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --language-model-only \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 --port 8000 --enforce-eager \
  --kv-cache-dtype fp8 --max-num-seqs 8 > logs/vllm.log 2>&1 &
echo "vllm pid=$!"
for i in $(seq 1 60); do
  if curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null; then
    echo "vllm ready after ~${i}0s"; break
  fi
  sleep 10
done

cd /workspace/AGI/scripts
rm -f /workspace/AGI/logs/dashboard.log
nohup /opt/venv/bin/python dashboard.py --host 0.0.0.0 --port 8010 > /workspace/AGI/logs/dashboard.log 2>&1 &
echo "dashboard pid=$!"
nohup /opt/venv/bin/python continuous_learner.py --cycles 0 --tick 2.5 \
  --seed-task "What is 2+2?" \
  --seed-task "What is the capital of France?" \
  > /workspace/AGI/logs/learner.log 2>&1 &
echo "learner pid=$!"

if [ ! -x /usr/local/bin/cloudflared ]; then
  curl -sL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi
nohup /usr/local/bin/cloudflared tunnel --url http://localhost:8010 --protocol http2 --no-autoupdate \
  > /workspace/AGI/logs/dash_tunnel.log 2>&1 &
echo "dash tunnel pid=$!"
nohup /usr/local/bin/cloudflared tunnel --url http://localhost:8000 --protocol http2 --no-autoupdate \
  > /workspace/AGI/logs/vllm_tunnel.log 2>&1 &
echo "vllm tunnel pid=$!"
sleep 25
echo "DASH_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /workspace/AGI/logs/dash_tunnel.log | head -1)"
echo "VLLM_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /workspace/AGI/logs/vllm_tunnel.log | head -1)"
echo "=== deploy done $(date -u) ==="
