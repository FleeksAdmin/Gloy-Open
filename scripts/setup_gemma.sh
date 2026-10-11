#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-12B-it-qat-w4a16-ct}"
FULL_MODEL="${FULL_MODEL:-google/gemma-4-12B-it}"
HF_HOME="${HF_HOME:-/workspace/models}"
PORT="${PORT:-8000}"
CUDAROOT=/opt/venv/lib/python3.12/site-packages/nvidia/cu13

git config --global http.postBuffer 104857600 || true

/opt/venv/bin/pip install -U pip
/opt/venv/bin/pip install vllm transformers torch accelerate huggingface_hub hf-transfer

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME="$CUDAROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

mkdir -p "$CUDAROOT/lib64/stubs"
if [ ! -e "$CUDAROOT/lib64/libcudart.so" ]; then
  ln -sf "$CUDAROOT/lib/libcudart.so.13" "$CUDAROOT/lib64/libcudart.so"
fi
if [ ! -e "$CUDAROOT/lib64/stubs/libcuda.so" ]; then
  ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDAROOT/lib64/stubs/libcuda.so"
fi

/opt/venv/bin/huggingface-cli download "$MODEL"

if command -v docker >/dev/null 2>&1; then
  docker run --gpus all -p "$PORT:8000" \
    vllm/vllm-openai:gemma4-unified "$MODEL" \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4
else
  /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization compressed-tensors \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --enforce-eager
fi