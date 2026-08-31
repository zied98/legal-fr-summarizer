#!/usr/bin/env bash
# Surveille le Job d'entraînement et résume son état.
set -euo pipefail
cd /data/projets/legal-fr-summarizer
set -a; . ./.env; set +a
JOB_ID="${1:-6a955b4a21c5aa7c83649545}"

STATUT=$(hf jobs ps -a 2>/dev/null | awk -v id="$JOB_ID" '$1==id {print $(NF-1)}')
echo "=== Job $JOB_ID : ${STATUT:-inconnu} ==="

timeout 25 hf jobs logs "$JOB_ID" 2>/dev/null \
  | grep -E "===|BASELINE|FINE-TUNÉ|COMPARAISON|généré [0-9]+/|'loss'|Publié|Error|Traceback" \
  | tail -25 || true
