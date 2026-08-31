#!/usr/bin/env python
"""Inspection manuelle des sorties du modèle fine-tuné.

Les métriques automatiques ne disent pas si une fiche est utilisable par un
juriste. On regarde les sorties réelles, y compris les échecs.
"""

import json
import sys

sys.path.insert(0, "/data/projets/legal-fr-summarizer")

from huggingface_hub import hf_hub_download

from src.schema import extraire_json, valider_fiche

REPO = "ZiedBz/legal-fr-cassation-qwen3-4b-lora"

chemin = hf_hub_download(REPO, "sorties_finetune.jsonl", repo_type="model")
sorties = [json.loads(x)["sortie"] for x in open(chemin, encoding="utf-8") if x.strip()]

test = [json.loads(x) for x in open("/data/projets/legal-fr-summarizer/data/test.jsonl")]
refs = [t["reference"] for t in test[: len(sorties)]]

print(f"{len(sorties)} sorties analysees\n")

echecs = []
for i, (s, r) in enumerate(zip(sorties, refs)):
    f = extraire_json(s)
    if f is None or not valider_fiche(f)[0]:
        echecs.append((i, s))

print(f"=== {len(echecs)} sorties non parsables ===")
for i, s in echecs[:3]:
    print(f"\n--- echec #{i} ({len(s)} chars) ---")
    print("DEBUT:", s[:150].replace("\n", " "))
    print("FIN  :", s[-200:].replace("\n", " "))

print("\n\n=== 2 EXEMPLES REUSSIS (modele vs reference) ===")
montres = 0
for s, r in zip(sorties, refs):
    f = extraire_json(s)
    if f is None or not valider_fiche(f)[0]:
        continue
    print(f"\n{'=' * 70}")
    print("MODELE   formation :", f["formation"])
    print("REFERENCE          :", r["formation"])
    print("MODELE   solution  :", f["solution"])
    print("REFERENCE          :", r["solution"])
    print("MODELE   articles  :", f["articles_vises"])
    print("REFERENCE          :", r["articles_vises"])
    print("\nMODELE resume :", f["resume"][:400])
    print("\nREFERENCE     :", r["resume"][:400])
    montres += 1
    if montres >= 2:
        break
