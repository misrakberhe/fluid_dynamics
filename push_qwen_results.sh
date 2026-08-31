#!/usr/bin/env bash
# Commit and push qwen_replication_outputs/ to GitHub after a pod run.
#
# Usage:
#   cd fluid_dynamics
#   bash push_qwen_results.sh
#   bash push_qwen_results.sh "Re-run Phase 2 with answer_pos scoring."
#
# Auth (private repo): use a GitHub PAT as the password when prompted,
# or: export GITHUB_PAT=ghp_xxxx && bash push_qwen_results.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MSG="${1:-Update Qwen replication outputs from GPU run.}"
OUT="qwen_replication_outputs"

if [[ ! -d "$OUT" ]]; then
  echo "ERROR: $OUT/ not found. Run experiments first."
  exit 1
fi

if [[ -z "$(ls -A "$OUT" 2>/dev/null)" ]]; then
  echo "ERROR: $OUT/ is empty."
  exit 1
fi

echo "=== Outputs to push ==="
ls -la "$OUT"

# identity for this repo only (does not touch global git config)
git config user.email "misraketshai@gmail.com"
git config user.name "Misrak Berhe"

git pull --rebase origin main

git add "$OUT/"
git status

if git diff --cached --quiet; then
  echo "Nothing new to commit."
  exit 0
fi

git commit -m "$MSG"

if [[ -n "${GITHUB_PAT:-}" ]]; then
  git push "https://misrakberhe:${GITHUB_PAT}@github.com/misrakberhe/fluid_dynamics.git" main
else
  git push origin main
fi

echo "=== Pushed to origin/main ==="
