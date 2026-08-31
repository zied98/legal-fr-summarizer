#!/usr/bin/env python
"""Pousse les jeux préparés vers un dataset privé du Hub.

Le Job d'entraînement tourne sur l'infrastructure Hugging Face : il ne voit
pas le disque local. Les données doivent donc transiter par le Hub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RACINE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="ex. ZiedBz/legal-fr-cassation-sft")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    from datasets import load_dataset

    fichiers = {
        "train": str(RACINE / "data" / "train.jsonl"),
        "validation": str(RACINE / "data" / "validation.jsonl"),
        "test": str(RACINE / "data" / "test.jsonl"),
    }
    ds = load_dataset("json", data_files=fichiers)
    for nom, split in ds.items():
        print(f"  {nom:11s} {len(split):>5} exemples")

    ds.push_to_hub(args.repo, private=args.private)
    print(f"\nPoussé : https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
