"""Construction des prompts d'entrée du modèle."""

from __future__ import annotations

SYSTEM = (
    "Tu es un assistant juridique spécialisé dans la jurisprudence française. "
    "Tu analyses un arrêt et produis une fiche structurée au format JSON."
)

INSTRUCTION = """Analyse l'arrêt suivant et produis une fiche structurée.

Réponds UNIQUEMENT par un objet JSON valide avec exactement ces clés :
- "formation" : la formation ayant rendu la décision
- "solution" : le sens de la décision (ex : Cassation, Rejet, Cassation partielle)
- "articles_vises" : liste des textes visés
- "resume" : le sommaire de l'arrêt, en un paragraphe

ARRÊT :
{arret}"""


def construire_prompt(arret: str, max_chars: int = 12000) -> str:
    """Construit l'instruction utilisateur à partir du texte d'un arrêt.

    Args:
        arret: texte brut de l'arrêt, déjà nettoyé.
        max_chars: troncature de sécurité. Les arrêts très longs dépassent
            la fenêtre de contexte et coûtent cher à entraîner ; on garde
            le début, qui contient l'essentiel (formation, visas, moyens).
    """
    if len(arret) > max_chars:
        arret = arret[:max_chars].rsplit(" ", 1)[0] + " […]"
    return INSTRUCTION.format(arret=arret)
