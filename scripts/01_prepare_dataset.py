#!/usr/bin/env python
"""Prépare les jeux train / validation / test à partir du dataset source.

Le jeu de test est mis de côté AVANT tout entraînement : c'est lui qui
mesurera l'apport réel du fine-tuning. Sans cette séparation, aucune
comparaison n'est honnête.

Usage:
    python scripts/01_prepare_dataset.py --n-train 3000 --n-eval 150
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import ecrire_jsonl, iterer_exemples  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
GRAINE = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-test", type=int, default=150)
    args = parser.parse_args()

    total = args.n_train + args.n_eval + args.n_test
    print(f"Collecte de {total} exemples depuis le dataset source…")

    exemples = []
    for i, ex in enumerate(iterer_exemples(limite=total), start=1):
        exemples.append(ex)
        if i % 500 == 0:
            print(f"  {i}/{total}")

    print(f"Collectés : {len(exemples)}")
    if len(exemples) < total:
        print("⚠️  moins d'exemples que demandé — le filtre est peut-être trop strict")

    random.Random(GRAINE).shuffle(exemples)

    n_test, n_eval = args.n_test, args.n_eval
    test = exemples[:n_test]
    validation = exemples[n_test : n_test + n_eval]
    train = exemples[n_test + n_eval :]

    for nom, lot in (("train", train), ("validation", validation), ("test", test)):
        chemin = RACINE / "data" / f"{nom}.jsonl"
        ecrire_jsonl(chemin, lot)
        print(f"  {nom:11s} {len(lot):>5} exemples → {chemin.relative_to(RACINE)}")

    print("\nTerminé. Le jeu de test ne doit JAMAIS servir à l'entraînement.")


if __name__ == "__main__":
    main()
