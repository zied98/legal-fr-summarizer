---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
datasets:
  - ZiedBz/legal-fr-cassation-sft
  - artefactory/Argimi-Legal-French-Jurisprudence
language:
  - fr
library_name: peft
pipeline_tag: text-generation
tags:
  - legal
  - french
  - lora
  - qlora
  - structured-output
  - summarization
---

# legal-fr-cassation-qwen3-4b-lora

Adaptateur LoRA spécialisant **Qwen3-4B-Instruct** dans la **synthèse structurée
d'arrêts de la Cour de cassation française**.

Le modèle prend un arrêt brut en entrée et produit une fiche JSON exploitable :

```json
{
  "formation": "Chambre commerciale",
  "solution": "Cassation partielle",
  "articles_vises": ["Article 1186, alinéas 2 et 3, du code civil"],
  "resume": "Selon l'article 1186..."
}
```

## Pourquoi ce modèle

Les cabinets d'avocats et directions juridiques françaises traitent des volumes
importants de jurisprudence. Les solutions reposant sur des APIs propriétaires
posent deux problèmes : la confidentialité des dossiers (RGPD, secret
professionnel) et le coût à l'échelle.

Un modèle de 4 milliards de paramètres quantifié en 4 bits tourne sur un GPU
unique, **on-premise**, sans que les données ne quittent l'infrastructure du
cabinet.

## Résultats

Évaluation sur des arrêts **jamais vus pendant l'entraînement**
(150 pour la baseline, 100 pour le modèle entraîné).

| Métrique | Modèle de base | Fine-tuné (A) | Fine-tuné (B) |
|---|---|---|---|
| **Formation exacte** | 0,0 % | **100 %** | **100 %** |
| **Résumé (ROUGE-L)** | 22,5 % | **55,0 %** | 52,4 % |
| **Articles visés (Jaccard)** | 6,7 % | **32,4 %** | 25,2 % |
| Solution exacte | 93,3 % | 96,8 % | **97,0 %** |
| JSON valide | 100 % | 95,0 % | **100 %** |

**(A)** `max_new_tokens=700`, sans pénalité de répétition.
**(B)** `max_new_tokens=900`, `repetition_penalty=1.1`.

Même adaptateur, même jeu de test de 100 arrêts : seuls les paramètres de
génération changent.

### Comparaison avec des modèles de base plus gros

Même jeu de test (100 arrêts jamais vus), même prompt, mêmes paramètres de
génération, quantification 4-bit pour tous.

| Modèle | Taille | JSON valide | Solution | **Formation** | Articles | Résumé |
|---|---|---|---|---|---|---|
| Qwen3-4B base | 4B | 100 % | 93,3 % | 0 % | 6,7 % | 22,5 % |
| Qwen3-14B | 14B | 95 % | 81,1 % | 0 % | 8,0 % | 22,7 % |
| Mistral-Small-24B | 24B | 100 % | 95,0 % | 16 % | 11,5 % | 25,7 % |
| Qwen3-32B | 32B | 86 % | 94,2 % | 0 % | 9,9 % | 22,4 % |
| **Ce modèle (LoRA)** | **4B** | **100 %** | **97,0 %** | **100 %** | **25,2 %** | **52,4 %** |

**Ce qu'on observe :**

1. **La taille ne résout pas le problème du format.** De 4B à 32B, la
   formation reste à 0 %. Aucun modèle généraliste ne connaît les conventions
   de nommage de la Cour de cassation — ce n'est pas une question de capacité
   de raisonnement, mais de convention métier.

2. **Plus gros n'est pas toujours mieux.** Le 14B fait *pire* que le 4B sur la
   solution (81,1 % contre 93,3 %), et le 32B produit le *moins* de JSON valide
   (86 %) : les grands modèles respectent moins volontiers un format imposé.

3. **Tous comprennent le droit** (solution exacte entre 81 % et 95 %) mais
   aucun ne sait produire la fiche attendue.

4. **Mistral-Small-24B est le seul modèle de base** à deviner quelques
   formations (16 %) — réputation méritée sur le français, mais loin des 100 %.

> ⚠️ **Limite de ce benchmark.** Il mesure l'**adéquation à un format métier**,
> pas la capacité de raisonnement juridique générale. Les modèles de base
> n'ont pas été prompt-engineerés spécifiquement : un prompt few-shot avec des
> exemples de fiches améliorerait probablement leurs scores. La comparaison est
> honnête sur son périmètre, pas au-delà.

Détail complet : `benchmark.json` dans ce dépôt.

### Un arbitrage, pas une amélioration

La configuration **(B)** corrige intégralement la régression de validité JSON
(95 % → 100 %), ce qui confirme que le problème venait de la génération et non
de l'entraînement. Mais elle dégrade la fidélité : **−7,2 points** sur les
articles visés et −2,6 sur le résumé.

L'explication est mécanique : `repetition_penalty` pénalise la réutilisation de
tokens déjà produits. Or un sommaire juridique **répète nécessairement** les
termes de l'arrêt, et une liste de visas contient plusieurs fois « article…
du code… ». La pénalité pousse le modèle à varier là où il devrait répéter.

**Quelle configuration choisir :**

| Usage | Recommandation |
|---|---|
| Pipeline automatisé (toute sortie doit être parsable) | **(B)** |
| Fiches relues par un juriste (fidélité prioritaire) | **(A)** |

**Piste non testée, probablement meilleure** : `max_new_tokens=900` **sans**
`repetition_penalty`. Cela devrait résoudre la troncature — cause majoritaire
des échecs — sans payer le coût de la pénalité.

### Lecture des résultats

**Le fine-tuning n'a pas appris le droit — il a appris le format.**

Le modèle de base identifiait déjà correctement le sens de la décision dans
93 % des cas : il *comprend* le raisonnement juridique. En revanche il ignorait
totalement les conventions de nommage de la Cour (formation à 0 %) et le format
canonique des visas (articles à 6,7 %).

C'est précisément ce que le fine-tuning corrige, et c'est pourquoi le gain est
spectaculaire là où le modèle ignorait une convention, et marginal là où il
raisonnait déjà bien.

### ⚠️ Limites connues

**Régression de validité JSON — diagnostiquée et corrigée.** L'inspection
manuelle des 5 sorties non parsables de la configuration (A) a montré qu'elles
font toutes 2 100 à 2 600 caractères : elles atteignent la limite de 700 tokens
et le JSON est tronqué avant fermeture. Une sortie présente en outre une
**dégénérescence par répétition** (la même proposition répétée en boucle
jusqu'à épuisement du budget).

La configuration (B) corrige entièrement le problème, au prix d'une perte de
fidélité documentée ci-dessus.

**Erreurs sur les articles visés.** Le Jaccard de 0,32 traduit une réalité
observable : sur un arrêt fiscal, le modèle cite « article 691 du CGI » là où
la référence attend « article 1594-0 G, A ». Le raisonnement est correct mais
le visa est faux. **En usage professionnel, c'est la limite la plus critique** :
un mauvais numéro d'article est plus grave qu'un résumé approximatif. Toute
sortie doit être vérifiée par un juriste.

**Biais du corpus.** Le jeu d'entraînement est dominé par la Chambre commerciale
(~65 %). Les performances sur les autres chambres sont probablement inférieures.
Le corpus source contient également des arrêts anciens (références en francs).

**Métrique imparfaite.** ROUGE-L mesure un recouvrement lexical et ne valorise
pas les reformulations correctes. Il est ici complété par une inspection
manuelle, mais ne constitue pas une évaluation juridique.

## Utilisation

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(BASE)
modele = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
modele = PeftModel.from_pretrained(modele, "ZiedBz/legal-fr-cassation-qwen3-4b-lora")

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

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": INSTRUCTION.format(arret=mon_arret)},
]
entrees = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to(modele.device)

sortie = modele.generate(entrees, max_new_tokens=900, repetition_penalty=1.1)
print(tokenizer.decode(sortie[0][entrees.shape[1]:], skip_special_tokens=True))
```

> **Note.** `max_new_tokens=900` et `repetition_penalty=1.1` sont recommandés :
> les valeurs utilisées à l'évaluation (700, sans pénalité) sont à l'origine de
> la régression de validité JSON documentée ci-dessus.

## Entraînement

| Paramètre | Valeur |
|---|---|
| Modèle de base | `Qwen/Qwen3-4B-Instruct-2507` |
| Méthode | QLoRA (4-bit NF4) |
| Rang LoRA (`r`) | 16 |
| `lora_alpha` | 32 |
| Modules ciblés | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Exemples d'entraînement | 3 000 |
| Époques | 1 |
| Learning rate | 2e-4 (cosine) |
| Longueur max | 4 096 tokens |
| Matériel | 1× NVIDIA L40S 48 Go |
| Durée | 1 h 08 |
| Coût GPU | ≈ 2,10 USD |
| Loss finale | 0,926 |

## Données

Dérivé de
[`artefactory/Argimi-Legal-French-Jurisprudence`](https://huggingface.co/datasets/artefactory/Argimi-Legal-French-Jurisprudence)
(configuration `juri`, Cour de cassation, CC-BY-SA-4.0).

Le dataset fournit les **sommaires officiels rédigés par la Cour**, utilisés
comme vérité terrain — aucune donnée synthétique n'a été générée.

Jeu de travail préparé et publié :
[`ZiedBz/legal-fr-cassation-sft`](https://huggingface.co/datasets/ZiedBz/legal-fr-cassation-sft).

**Nettoyage notable** : le champ `solution` du corpus source contenait
**61 valeurs distinctes** pour seulement **8 sens de décision réels**
(`Rejet` et `REJET` comptés séparément, valeurs composites du type
`"Cassation partielle REJET Cassation"` issues d'arrêts multi-pourvois).
Une normalisation ramène le champ à 8 classes propres ; les valeurs composites
sont écartées plutôt qu'apprises de travers.

## Reproductibilité

Code complet, tests et pipeline d'évaluation :
le dépôt contient les scripts de préparation, d'entraînement (Hugging Face Jobs)
et d'évaluation, avec 46 tests unitaires et une CI.

Fichiers publiés ici :

| Fichier | Contenu |
|---|---|
| `resultats.json` | Métriques baseline vs configuration (A) |
| `resultats_generation_v2.json` | Métriques de la configuration (B) |
| `baseline.json` | Modèle de base, avant fine-tuning |
| `sorties_finetune.jsonl` | Les 100 sorties brutes, configuration (A) |
| `sorties_v2.jsonl` | Les 100 sorties brutes, configuration (B) |

L'ensemble permet de reproduire les tableaux ci-dessus et d'inspecter les
sorties réelles, y compris les échecs.

## Licence

Apache 2.0 pour l'adaptateur. Les données dérivées restent soumises à la
licence CC-BY-SA-4.0 du corpus source.

## Auteur

**Zied Bouzekri** — [LinkedIn](https://www.linkedin.com/in/zied-bouzekri-28a4861a1/) · [GitHub](https://github.com/zied98)
