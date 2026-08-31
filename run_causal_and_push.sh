#!/usr/bin/env bash
# Sync, run Phase 3 causal, push outputs. For RunPod at /fluid_dynamics.
#
# Usage:
#   cd /fluid_dynamics && source .venv/bin/activate
#   export GITHUB_PAT=ghp_xxxx
#   bash run_causal_and_push.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

bash pod_sync.sh
source .venv/bin/activate

python qwen_anchoring_replication.py causal
bash push_qwen_results.sh "${1:-Qwen Phase 3 causal (answer_pos scoring, GPU).}"
