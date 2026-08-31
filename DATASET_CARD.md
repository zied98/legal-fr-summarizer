---
license: cc-by-sa-4.0
language:
  - fr
task_categories:
  - text-generation
  - summarization
tags:
  - legal
  - french
  - jurisprudence
  - structured-output
size_categories:
  - 1K<n<10K
---

# legal-fr-cassation-sft

Jeu de données d'instruction pour la **synthèse structurée d'arrêts de la Cour
de cassation française**, dérivé de
[`artefactory/Argimi-Legal-French-Jurisprudence`](https://huggingface.co/datasets/artefactory/Argimi-Legal-French-Jurisprudence)
(configuration `juri`).

## Contenu

| Split | Exemples |
|---|---|
| `train` | 3 000 |
| `validation` | 150 |
| `test` | 150 |

Le split `test` a été mis de côté **avant tout entraînement** et n'a jamais
été vu par le modèle.

## Structure

| Champ | Description |
|---|---|
| `id` | Identifiant Légifrance de l'arrêt (`JURITEXT…`) |
| `prompt` | Instruction complète, incluant le texte nettoyé de l'arrêt |
| `completion` | Fiche JSON attendue, sérialisée |
| `reference` | La même fiche sous forme d'objet, pour l'évaluation |

La fiche cible :

```json
{
  "formation": "Chambre commerciale",
  "solution": "Cassation partielle",
  "articles_vises": ["Article 1186, alinéas 2 et 3, du code civil"],
  "resume": "Selon l'article 1186..."
}
```

## Vérité terrain

Le corpus source contient les **sommaires officiels rédigés par la Cour de
cassation** (champ `summary`). Ils servent directement de cible : **aucune
donnée n'a été générée synthétiquement**.

Les autres champs de la fiche proviennent également du corpus :
`formation`, `solution` et `applied_laws`.

## Traitements appliqués

1. **Nettoyage HTML** — le champ `content` de Légifrance contient des balises
   `<br>` et des entités ; le texte est normalisé.
2. **Normalisation de la formation** — `CHAMBRE_COMMERCIALE` → `Chambre commerciale`.
3. **Découpage des visas** — `applied_laws` converti en liste de chaînes.
4. **Normalisation des solutions** — voir ci-dessous.
5. **Filtrage** — les quatre champs de la fiche doivent être présents ;
   longueur du résumé entre 200 et 3 000 caractères ; arrêt d'au moins
   1 500 caractères, tronqué à 12 000.

### Normalisation des solutions

Le champ `solution` du corpus source contenait **61 valeurs distinctes** pour
seulement **8 sens de décision réels** :

- des variantes de casse comptées séparément (`Rejet` : 967, `REJET` : 492) ;
- des valeurs composites issues d'arrêts multi-pourvois mal aplatis, du type
  `"Cassation partielle REJET Cassation"`.

Après normalisation :

| Solution | Exemples |
|---|---|
| Rejet | 1 555 |
| Cassation | 673 |
| Cassation partielle | 577 |
| Cassation sans renvoi | 77 |
| Cassation partielle sans renvoi | 72 |
| Annulation | 22 |
| Irrecevabilité | 22 |
| Déchéance | 2 |

Les valeurs composites sont **écartées** plutôt qu'apprises de travers.

## Biais et limites

- **Déséquilibre des chambres** : la Chambre commerciale représente ~65 % du
  corpus. Les autres formations sont sous-représentées.
- **Ancienneté** : le corpus source couvre une longue période ; certains arrêts
  font référence à des montants en francs ou à des textes abrogés.
- **Hétérogénéité des visas** : le format du champ `applied_laws` varie
  fortement d'un arrêt à l'autre, ce qui rend la tâche d'extraction difficile.
- **Troncature** : les arrêts sont coupés à 12 000 caractères. Le début d'un
  arrêt contient l'essentiel (formation, visas, moyens), mais une partie du
  raisonnement peut être perdue sur les décisions longues.

## Modèle entraîné

[`ZiedBz/legal-fr-cassation-qwen3-4b-lora`](https://huggingface.co/ZiedBz/legal-fr-cassation-qwen3-4b-lora)
— adaptateur LoRA sur Qwen3-4B-Instruct.

Résultats sur le split `test` (jamais vu) :
formation **0 % → 100 %**, ROUGE-L du résumé **22,5 % → 55,0 %**,
Jaccard des articles **6,7 % → 32,4 %**.

## Licence

CC-BY-SA-4.0, héritée du corpus source.

## Auteur

**Zied Bouzekri** — [LinkedIn](https://www.linkedin.com/in/zied-bouzekri-28a4861a1/) · [GitHub](https://github.com/zied98)
