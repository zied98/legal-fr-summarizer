"""Nettoyage du texte brut des arrêts.

Le champ `content` du dataset source contient du HTML (<br>, entités) hérité
de Légifrance. On le normalise avant tout usage.
"""

from __future__ import annotations

import html
import re

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_ESPACES = re.compile(r"[ \t]+")
_SAUTS = re.compile(r"\n{3,}")


def nettoyer_arret(texte: str) -> str:
    """Convertit le HTML de Légifrance en texte lisible."""
    if not texte:
        return ""
    texte = _BR.sub("\n", texte)
    texte = _TAGS.sub(" ", texte)
    texte = html.unescape(texte)
    texte = _ESPACES.sub(" ", texte)
    texte = _SAUTS.sub("\n\n", texte)
    return "\n".join(ligne.strip() for ligne in texte.split("\n")).strip()


def normaliser_formation(valeur: str) -> str:
    """'CHAMBRE_COMMERCIALE' -> 'Chambre commerciale'."""
    if not valeur:
        return ""
    return valeur.replace("_", " ").strip().capitalize()


#: Solutions canoniques de la Cour de cassation. Le dataset source contient
#: des variantes de casse ("Rejet" / "REJET") et des valeurs composites
#: issues d'arrêts multi-pourvois mal aplatis.
SOLUTIONS_CANONIQUES = (
    "Cassation partielle sans renvoi",
    "Cassation partielle",
    "Cassation sans renvoi",
    "Cassation",
    "Rejet",
    "Irrecevabilité",
    "Annulation",
    "Non-lieu à statuer",
    "Déchéance",
)


def normaliser_solution(valeur: str) -> str:
    """Ramène le sens de la décision à une valeur canonique.

    Retourne "" si la valeur est composite (arrêt multi-pourvois) ou
    inconnue : ces exemples sont écartés plutôt qu'appris de travers.
    """
    if not valeur:
        return ""
    brut = " ".join(valeur.split())
    bas = brut.lower()

    correspondances = [c for c in SOLUTIONS_CANONIQUES if c.lower() in bas]
    if not correspondances:
        return ""

    # Une valeur composite comme "Cassation partielle REJET Cassation" cite
    # plusieurs solutions distinctes : on l'écarte.
    racines = {c.split()[0].lower() for c in correspondances}
    if len(racines) > 1:
        return ""

    return max(correspondances, key=len)


def normaliser_articles(valeur: str) -> list[str]:
    """Découpe le champ applied_laws en liste de textes visés."""
    if not valeur:
        return []
    morceaux = re.split(r"\s*;\s*|\n+", valeur)
    return [m.strip().rstrip(".").strip() for m in morceaux if m.strip()]
