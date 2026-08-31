# legal-fr-summarizer

Fine-tuning LoRA d'un petit modèle de langage pour la **synthèse structurée
d'arrêts de la Cour de cassation**.

## Problème

Les cabinets d'avocats et directions juridiques françaises traitent des
volumes importants de jurisprudence. Les solutions actuelles reposent sur des
APIs propriétaires américaines — incompatibles avec les contraintes RGPD et
le secret professionnel, et coûteuses à l'échelle.

## Approche

Un modèle de petite taille (~4B paramètres), spécialisé par fine-tuning LoRA,
capable de tourner **on-premise sur un GPU unique**, qui transforme un arrêt
brut en fiche structurée JSON :

```json
{
  "formation": "Chambre commerciale",
  "solution": "Cassation partielle",
  "articles_vises": ["Article 1186, alinéas 2 et 3, du code civil"],
  "resume": "..."
}
```

## Données

[`artefactory/Argimi-Legal-French-Jurisprudence`](https://huggingface.co/datasets/artefactory/Argimi-Legal-French-Jurisprudence)
— configuration `juri` (Cour de cassation), 147 674 arrêts, licence CC-BY-SA-4.0.

Le dataset contient les sommaires officiels rédigés par la Cour, qui servent
de vérité terrain.

## Résultats

⏳ En cours — baseline et modèle fine-tuné à venir.

| Métrique | Modèle de base | Fine-tuné | Δ |
|---|---|---|---|
| JSON valide (%) | – | – | – |
| Exactitude `solution` | – | – | – |
| Exactitude `formation` | – | – | – |
| ROUGE-L (résumé) | – | – | – |

## Structure

```
src/          code réutilisable (dataset, prompts, évaluation)
scripts/      scripts exécutables (préparation, baseline, entraînement)
tests/        tests unitaires
eval/         jeux d'évaluation et résultats
```

## Reproduire

```bash
pip install -e ".[dev]"
python scripts/01_prepare_dataset.py
python scripts/02_baseline_eval.py
```

## Licence

MIT (code) · CC-BY-SA-4.0 (données dérivées du dataset source)
