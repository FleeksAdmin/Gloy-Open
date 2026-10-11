#!/bin/bash
# Serve base model WITH a trained LoRA adapter applied at inference (vLLM).
# Prior step: bash run_lora.sh (trains adapter into models/adapters/<ts>).
pkill -f 'vllm.entr[y]points' 2>/dev/null || true
sleep 5
ADAPTER=$(ls -1dt /workspace/AGI/models/adapters/adapter_* 2>/dev/null | head -1)
if [ -z "$ADAPTER" ]; then echo "no adapter found; serving base only"; ADAPTER=""; fi
C=/opt/venv/lib/python3.12/site-packages/nvidia/cu13
export HF_HOME=/workspace/models
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME=$C
export PATH=$C/bin:$PATH
export LD_LIBRARY_PATH=$C/lib:$C/lib64
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
rm -f /workspace/AGI/logs/vllm.log
EXTRA=("--enable-lora" "--max-lora-rank" "16" "--lora-modules" "gloy=$ADAPTER")
nohup /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 --language-model-only \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 --port 8000 --enforce-eager \
  --kv-cache-dtype fp8 --max-num-seqs 8 \
  "${EXTRA[@]}" > /workspace/AGI/logs/vllm.log 2>&1 &
echo "vllm pid=$! adapter=$ADAPTER"
for i in $(seq 1 60); do
  if curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null; then echo "vllm ready"; break; fi
  sleep 10
done
echo "apply adapter at inference via model=gloy in /api/generate (GLOW_MODEL=gloy)"