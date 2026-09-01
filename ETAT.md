# ETAT — legal-fr-summarizer

> Dernière mise à jour : 01/09/2026 — **PROJET LIVRÉ**

## 1. Où en est-on

**Terminé et publié.** Modèle LoRA de synthèse d'arrêts de la Cour de cassation,
avec dataset, benchmark à 5 modèles, model card documentée et code testé.

## 2. Décisions prises

| Décision | Raison |
|---|---|
| Sous-ensemble `juri` (Cour de cassation) | 147 674 arrêts avec sommaires officiels = vérité terrain gratuite |
| Conseil d'État écarté | Seulement ~122 exemples exploitables sur 613 488 (vérifié) |
| Sortie JSON structurée | Mesurable objectivement, exploitable par un logiciel métier |
| Publier l'adaptateur AVANT l'évaluation | Un run perdu après 1h08 de GPU l'a imposé |
| Versions épinglées | transformers 5.x a supprimé `warmup_ratio` → job planté après 19 min |

## 3. Prochaine action

Aucune côté technique. **Reste l'article LinkedIn** — matière prête dans
`/data/apprentissage/notes-article-linkedin.md`.

Extension possible : JurisBench-FR (voir roadmap).

## 4. Blocages

Aucun.

## Résultats

| Métrique | Base 4B | **LoRA 4B** | Qwen3-32B |
|---|---|---|---|
| Formation exacte | 0 % | **100 %** | 0 % |
| Résumé ROUGE-L | 22,5 % | **52,4 %** | 22,4 % |
| Articles Jaccard | 6,7 % | **25,2 %** | 9,9 % |
| Solution exacte | 93,3 % | **97,0 %** | 94,2 % |

**Le LoRA 4B bat Qwen3-14B, Mistral-24B et Qwen3-32B sur les 5 métriques.**

## Liens

- Repo : https://github.com/zied98/legal-fr-summarizer
- Modèle : https://huggingface.co/ZiedBz/legal-fr-cassation-qwen3-4b-lora
- Dataset : https://huggingface.co/datasets/ZiedBz/legal-fr-cassation-sft
