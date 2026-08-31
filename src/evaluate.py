"""Évaluation des fiches produites par le modèle.

Trois niveaux de mesure, du plus strict au plus souple :

1. **Validité JSON** — le modèle produit-il un objet exploitable ? C'est la
   condition d'usage réel : une fiche non parsable est inutilisable par un
   logiciel métier, quelle que soit la qualité du texte.
2. **Exactitude des champs catégoriels** (`solution`, `formation`) — mesure
   objective, sans ambiguïté.
3. **Qualité du résumé** — ROUGE-L, qui mesure le recouvrement de la plus
   longue sous-séquence commune avec le sommaire officiel.

ROUGE-L est imparfait (il ignore les reformulations correctes) mais il est
reproductible et sans coût. Il est complété par une inspection manuelle d'un
échantillon, documentée dans la model card.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from src.schema import extraire_json, valider_fiche

_MOT = re.compile(r"\w+", re.UNICODE)


def _normaliser(texte: str) -> str:
    """Minuscules, sans accents, espaces réduits — pour comparer des libellés."""
    texte = unicodedata.normalize("NFD", texte or "")
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return " ".join(texte.lower().split())


def _tokens(texte: str) -> list[str]:
    return _MOT.findall(_normaliser(texte))


def _lcs(a: list[str], b: list[str]) -> int:
    """Longueur de la plus longue sous-séquence commune (programmation dynamique)."""
    if not a or not b:
        return 0
    precedent = [0] * (len(b) + 1)
    for x in a:
        courant = [0]
        for j, y in enumerate(b):
            courant.append(precedent[j] + 1 if x == y else max(courant[j], precedent[j + 1]))
        precedent = courant
    return precedent[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """F-mesure ROUGE-L entre une prédiction et une référence."""
    p, r = _tokens(prediction), _tokens(reference)
    if not p or not r:
        return 0.0
    commun = _lcs(p, r)
    if commun == 0:
        return 0.0
    precision, rappel = commun / len(p), commun / len(r)
    return 2 * precision * rappel / (precision + rappel)


def jaccard(predits: list[str], references: list[str]) -> float:
    """Recouvrement entre deux listes d'articles visés."""
    a = {_normaliser(x) for x in predits if x}
    b = {_normaliser(x) for x in references if x}
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluer_exemple(sortie_brute: str, reference: dict) -> dict:
    """Évalue une sortie de modèle contre sa fiche de référence."""
    fiche = extraire_json(sortie_brute)
    if fiche is None:
        return {"json_valide": 0.0, "schema_valide": 0.0}

    schema_ok, _ = valider_fiche(fiche)
    resultat = {"json_valide": 1.0, "schema_valide": 1.0 if schema_ok else 0.0}

    if not schema_ok:
        return resultat

    resultat["solution_exacte"] = float(
        _normaliser(fiche["solution"]) == _normaliser(reference["solution"])
    )
    resultat["formation_exacte"] = float(
        _normaliser(fiche["formation"]) == _normaliser(reference["formation"])
    )
    resultat["articles_jaccard"] = jaccard(fiche["articles_vises"], reference["articles_vises"])
    resultat["resume_rouge_l"] = rouge_l(fiche["resume"], reference["resume"])
    return resultat


def agreger(resultats: list[dict]) -> dict:
    """Moyenne les métriques sur l'ensemble du jeu de test.

    Les métriques de qualité ne sont moyennées que sur les sorties au schéma
    valide : moyenner un ROUGE sur des fiches non parsables n'aurait pas de
    sens. Le taux de validité est reporté séparément.
    """
    if not resultats:
        return {}

    total = len(resultats)
    sommes: Counter[str] = Counter()
    effectifs: Counter[str] = Counter()
    for r in resultats:
        for cle, valeur in r.items():
            sommes[cle] += valeur
            effectifs[cle] += 1

    agrege = {
        "n": total,
        "json_valide": sommes["json_valide"] / total,
        "schema_valide": sommes["schema_valide"] / total,
    }
    for cle in ("solution_exacte", "formation_exacte", "articles_jaccard", "resume_rouge_l"):
        if effectifs[cle]:
            agrege[cle] = sommes[cle] / effectifs[cle]
    return agrege
