"""Schéma de la fiche structurée produite par le modèle.

Un seul endroit définit le format de sortie : prompts, parsing et évaluation
importent tous d'ici. Si le format change, il ne change qu'ici.
"""

from __future__ import annotations

import json
import re

#: Clés attendues dans la sortie du modèle, dans l'ordre.
FICHE_KEYS = ("formation", "solution", "articles_vises", "resume")


def valider_fiche(obj: object) -> tuple[bool, str]:
    """Valide qu'un objet respecte le schéma de fiche.

    Returns:
        (True, "") si valide, sinon (False, motif du rejet).
    """
    if not isinstance(obj, dict):
        return False, "la sortie n'est pas un objet JSON"

    manquantes = [k for k in FICHE_KEYS if k not in obj]
    if manquantes:
        return False, f"clés manquantes : {', '.join(manquantes)}"

    for cle in ("formation", "solution", "resume"):
        if not isinstance(obj[cle], str):
            return False, f"'{cle}' doit être une chaîne"

    if not isinstance(obj["articles_vises"], list):
        return False, "'articles_vises' doit être une liste"
    if not all(isinstance(a, str) for a in obj["articles_vises"]):
        return False, "'articles_vises' doit ne contenir que des chaînes"

    return True, ""


def extraire_json(texte: str) -> dict | None:
    """Extrait le premier objet JSON valide d'une sortie de modèle.

    Les LLM entourent souvent leur JSON de texte ou de balises markdown.
    On tente d'abord un parsing direct, puis on cherche un bloc ```json,
    puis le premier objet {...} équilibré.
    """
    texte = texte.strip()

    try:
        return json.loads(texte)
    except json.JSONDecodeError:
        pass

    bloc = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texte, re.DOTALL)
    if bloc:
        try:
            return json.loads(bloc.group(1))
        except json.JSONDecodeError:
            pass

    debut = texte.find("{")
    if debut == -1:
        return None
    profondeur = 0
    for i, c in enumerate(texte[debut:], start=debut):
        if c == "{":
            profondeur += 1
        elif c == "}":
            profondeur -= 1
            if profondeur == 0:
                try:
                    return json.loads(texte[debut : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
