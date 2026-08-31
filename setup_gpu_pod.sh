#!/usr/bin/env bash
# Bootstrap a fresh GPU pod (RunPod / Vast.ai) for Qwen anchoring replication.
# Usage (on the GPU machine):
#   cd fluid_dynamics && bash setup_gpu_pod.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== GPU pod setup: fluid_dynamics / Qwen replication ==="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script must run on a GPU machine."
  echo "  Provision a pod with >=24GB VRAM (e.g. RTX 4090, A5000, A10)."
  exit 1
fi

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -d .venv ]]; then
  echo "--- Creating venv ---"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip wheel

echo "--- PyTorch (CUDA) ---"
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))"
else
  echo "Installing CUDA-enabled torch..."
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA still unavailable after torch install'"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))"
fi

echo "--- Project deps ---"
pip install -r requirements-qwen.txt

echo "--- Optional: HuggingFace token (higher rate limits) ---"
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "HF_TOKEN not set. Model download may be slow; export HF_TOKEN if you have one."
else
  echo "HF token present."
fi

echo "--- Phase 1 on GPU (Qwen3.5-4B) ---"
python qwen_anchoring_replication.py all

echo ""
echo "=== Setup complete ==="
echo "Next: python qwen_anchoring_replication.py smoke-test"
echo "Phase 2: python qwen_anchoring_replication.py behavior   # (once implemented)"
