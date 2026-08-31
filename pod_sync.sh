#!/usr/bin/env bash
# Sync pod repo to latest origin/main (handles divergent pod commits).
# Preserves qwen_replication_outputs/ across reset.
#
# Usage: cd /fluid_dynamics && bash pod_sync.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OUT="qwen_replication_outputs"
BACKUP=""

cleanup() {
  [[ -n "$BACKUP" && -d "$BACKUP" ]] && rm -rf "$BACKUP"
}
trap cleanup EXIT

echo "=== Syncing to origin/main ==="
git fetch origin main

if [[ -d "$OUT" && -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
  BACKUP="$(mktemp -d)"
  cp -a "$OUT/." "$BACKUP/"
  echo "Backed up $OUT/ to $BACKUP"
fi

git reset --hard origin/main

if [[ -n "$BACKUP" ]]; then
  mkdir -p "$OUT"
  cp -a "$BACKUP/." "$OUT/"
  echo "Restored $OUT/ (re-run behavior if you needed code fixes)"
fi

echo "At commit: $(git log -1 --oneline)"

if grep -q "score_W_minus_C_at_tstar" qwen_anchoring_replication.py; then
  echo "answer_pos scoring: present"
else
  echo "WARNING: old qwen_anchoring_replication.py — git pull may have failed earlier"
fi
