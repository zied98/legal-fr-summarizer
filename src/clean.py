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


def normaliser_articles(valeur: str) -> list[str]:
    """Découpe le champ applied_laws en liste de textes visés."""
    if not valeur:
        return []
    morceaux = re.split(r"\s*;\s*|\n+", valeur)
    return [m.strip().rstrip(".").strip() for m in morceaux if m.strip()]
