#!/usr/bin/env bash
# Smoke test : valide le pipeline complet sur un mini-échantillon.
# Objectif : détecter les erreurs de code en ~6 minutes plutôt qu'après
# 1h08 d'entraînement. Aucune valeur scientifique — seulement "ça tourne".
set -euo pipefail
cd /data/projets/legal-fr-summarizer
set -a; . ./.env; set +a

hf jobs uv run --flavor l40sx1 --timeout 30m \
  -s HF_TOKEN="$HF_TOKEN" \
  -e DATASET_TRAVAIL=ZiedBz/legal-fr-cassation-sft \
  -e DEPOT_SORTIE=ZiedBz/legal-fr-cassation-smoketest \
  -e N_EVAL=4 \
  -e N_TRAIN_MAX=24 \
  -e EPOQUES=1 \
  -e LORA_R=8 \
  -e LORA_ALPHA=16 \
  scripts/train_job.py
