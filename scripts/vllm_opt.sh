pkill -f 'vllm.entr[y]points' 2>/dev/null || true
sleep 5
touch /workspace/AGI/logs/continuous/pause.flag
C=/opt/venv/lib/python3.12/site-packages/nvidia/cu13
export HF_HOME=/workspace/models
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME=$C
export PATH=$C/bin:$PATH
export LD_LIBRARY_PATH=$C/lib:$C/lib64
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
rm -f /workspace/AGI/logs/vllm.log
nohup /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --language-model-only \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 --port 8000 --enforce-eager \
  --kv-cache-dtype fp8 --max-num-seqs 8 \
  > /workspace/AGI/logs/vllm.log 2>&1 &
echo "vllm pid=$!"
for i in $(seq 1 60); do
  if curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null; then echo "vllm ready after ~${i}0s"; break; fi
  sleep 10
done
rm -f /workspace/AGI/logs/continuous/pause.flag
echo DONE