#!/usr/bin/env bash
# Sync, run Phase 2 behavior, push outputs. For RunPod at /fluid_dynamics.
#
# Usage:
#   cd /fluid_dynamics
#   source .venv/bin/activate
#   export GITHUB_PAT=ghp_xxxx   # optional; else enter PAT when prompted
#   bash run_behavior_and_push.sh
#
#   bash run_behavior_and_push.sh "Custom commit message."
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

bash pod_sync.sh

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv missing. Run: bash setup_gpu_pod.sh"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python qwen_anchoring_replication.py behavior
bash push_qwen_results.sh "${1:-Qwen Phase 2 behavior (answer_pos scoring, GPU).}"
