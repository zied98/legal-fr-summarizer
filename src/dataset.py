"""Construction du jeu de données d'entraînement à partir du dataset source.

Source : artefactory/Argimi-Legal-French-Jurisprudence, configuration `juri`
(Cour de cassation, 147 674 arrêts, CC-BY-SA-4.0).

Le dataset fournit le sommaire officiel rédigé par la Cour (`summary`), qui
sert de vérité terrain : on n'a donc pas besoin de générer un corpus
synthétique.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from src.clean import (
    nettoyer_arret,
    normaliser_articles,
    normaliser_formation,
    normaliser_solution,
)
from src.prompts import construire_prompt

DATASET_ID = "artefactory/Argimi-Legal-French-Jurisprudence"
CONFIG = "juri"

#: Longueurs retenues. Trop court = arrêt sans substance ;
#: trop long = coût d'entraînement inutile et troncature destructrice.
MIN_ARRET_CHARS = 1500
MAX_ARRET_CHARS = 12000
MIN_RESUME_CHARS = 200
MAX_RESUME_CHARS = 3000


def exemple_utilisable(ligne: dict) -> bool:
    """Filtre les lignes exploitables pour l'entraînement.

    On exige les quatre champs de la fiche. Un exemple incomplet apprendrait
    au modèle à inventer, ce qui est exactement ce qu'on veut éviter dans un
    contexte juridique.
    """
    resume = (ligne.get("summary") or "").strip()
    if not (MIN_RESUME_CHARS <= len(resume) <= MAX_RESUME_CHARS):
        return False
    if not normaliser_solution(ligne.get("solution") or ""):
        return False
    if not (ligne.get("formation") or "").strip():
        return False
    if not (ligne.get("applied_laws") or "").strip():
        return False
    return bool((ligne.get("content") or "").strip())


def construire_exemple(ligne: dict) -> dict | None:
    """Transforme une ligne brute en paire (prompt, fiche JSON attendue)."""
    arret = nettoyer_arret(ligne["content"])
    if not (MIN_ARRET_CHARS <= len(arret)):
        return None

    fiche = {
        "formation": normaliser_formation(ligne["formation"]),
        "solution": normaliser_solution(ligne["solution"]),
        "articles_vises": normaliser_articles(ligne["applied_laws"]),
        "resume": " ".join(ligne["summary"].split()),
    }

    return {
        "id": ligne.get("id", ""),
        "prompt": construire_prompt(arret, max_chars=MAX_ARRET_CHARS),
        "completion": json.dumps(fiche, ensure_ascii=False),
        "reference": fiche,
    }


def iterer_exemples(limite: int | None = None) -> Iterator[dict]:
    """Parcourt le dataset en streaming et produit les exemples retenus.

    Le streaming évite de télécharger 448 Mo pour n'en garder qu'une fraction.
    """
    from datasets import load_dataset

    flux = load_dataset(DATASET_ID, CONFIG, split="train", streaming=True)
    produits = 0
    for ligne in flux:
        if not exemple_utilisable(ligne):
            continue
        exemple = construire_exemple(ligne)
        if exemple is None:
            continue
        yield exemple
        produits += 1
        if limite is not None and produits >= limite:
            return


def ecrire_jsonl(chemin: Path, exemples: list[dict]) -> None:
    """Écrit une liste d'exemples au format JSONL."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as f:
        for ex in exemples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def lire_jsonl(chemin: Path) -> list[dict]:
    """Lit un fichier JSONL."""
    with chemin.open(encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]
