#!/usr/bin/env python
"""Publie les cartes du modèle et du dataset sur le Hub."""

from pathlib import Path

from huggingface_hub import HfApi

RACINE = Path(__file__).resolve().parents[1]
api = HfApi()

api.upload_file(
    path_or_fileobj=str(RACINE / "MODEL_CARD.md"),
    path_in_repo="README.md",
    repo_id="ZiedBz/legal-fr-cassation-qwen3-4b-lora",
    repo_type="model",
    commit_message="docs: model card complète avec résultats et limites",
)
print("Model card publiée")

api.upload_file(
    path_or_fileobj=str(RACINE / "DATASET_CARD.md"),
    path_in_repo="README.md",
    repo_id="ZiedBz/legal-fr-cassation-sft",
    repo_type="dataset",
    commit_message="docs: dataset card avec traitements et biais documentés",
)
print("Dataset card publiée")

api.update_repo_settings("ZiedBz/legal-fr-cassation-sft", private=False, repo_type="dataset")
print("Dataset rendu public")
