pkill -f 'vllm.entr[y]points' 2>/dev/null || true
sleep 4
mkdir -p /workspace/AGI/logs/continuous
touch /workspace/AGI/logs/continuous/pause.flag
export HF_HOME=/workspace/models
export HF_HUB_ENABLE_HF_TRANSFER=1
export GLOW_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
C=/opt/venv/lib/python3.12/site-packages/nvidia/cu13
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME=$C
export PATH=$C/bin:$PATH
export LD_LIBRARY_PATH=$C/lib:$C/lib64
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# Model is already in HF cache (~22G). 24G card: keep context short so KV fits.
rm -f /workspace/AGI/logs/vllm.log
nohup /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --language-model-only \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 --port 8000 --enforce-eager \
  --kv-cache-dtype fp8 --max-num-seqs 8 \
  > /workspace/AGI/logs/vllm.log 2>&1 &
echo "vllm pid=$!"
for i in $(seq 1 90); do
  if curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null; then
    echo "vllm ready after ~${i}0s"
    curl -s --max-time 6 http://localhost:8000/v1/models | head -c 300
    echo
    rm -f /workspace/AGI/logs/continuous/pause.flag
    echo DONE
    exit 0
  fi
  if ! pgrep -f 'vllm.entr[y]points' >/dev/null; then
    echo "vllm died; last log:"
    tail -n 20 /workspace/AGI/logs/vllm.log | tr -d '\r'
    exit 1
  fi
  sleep 10
done
echo "vllm still starting; last log:"
tail -n 12 /workspace/AGI/logs/vllm.log | tr -d '\r'
exit 2
