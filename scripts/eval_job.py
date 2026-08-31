"""Réévalue un adaptateur LoRA déjà entraîné, sans réentraînement.

Objectif : vérifier si la régression de validité JSON observée sur la v1
(95 % au lieu de 100 %) vient bien des paramètres de génération, et non de
l'entraînement lui-même.

Diagnostic posé par inspection manuelle des 5 sorties non parsables :
  - toutes font 2100-2600 caractères et sont tronquées à 700 tokens ;
  - l'une d'elles dégénère en répétant la même proposition en boucle.

Hypothèse testée ici : `repetition_penalty` + un budget de tokens plus large
suffisent à corriger, sans toucher au modèle.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers==4.57.6",
#   "peft==0.17.1",
#   "datasets>=3.0,<5",
#   "accelerate>=1.0",
#   "bitsandbytes>=0.44",
#   "huggingface_hub>=0.26",
# ]
# ///

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELE_BASE = os.environ.get("MODELE_BASE", "Qwen/Qwen3-4B-Instruct-2507")
#: Adaptateur LoRA optionnel. Vide = on évalue le modèle de base seul, ce qui
#: permet d'utiliser ce script pour benchmarker des modèles concurrents.
ADAPTATEUR = os.environ.get("ADAPTATEUR", "").strip()
DATASET_TRAVAIL = os.environ["DATASET_TRAVAIL"]
DEPOT_SORTIE = os.environ.get("DEPOT_SORTIE", "")

N_EVAL = int(os.environ.get("N_EVAL", "100"))
MAX_NOUVEAUX_TOKENS = int(os.environ.get("MAX_NOUVEAUX_TOKENS", "900"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.1"))
LONGUEUR_MAX = 4096

SORTIE = Path("/tmp/sortie")
SORTIE.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "Tu es un assistant juridique spécialisé dans la jurisprudence française. "
    "Tu analyses un arrêt et produis une fiche structurée au format JSON."
)

_MOT = re.compile(r"\w+", re.UNICODE)
FICHE_KEYS = ("formation", "solution", "articles_vises", "resume")


def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFD", texte or "")
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return " ".join(texte.lower().split())


def _tokens(t: str) -> list[str]:
    return _MOT.findall(_normaliser(t))


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prec = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prec[j] + 1 if x == y else max(cur[j], prec[j + 1]))
        prec = cur
    return prec[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    c = _lcs(p, r)
    if c == 0:
        return 0.0
    prec, rap = c / len(p), c / len(r)
    return 2 * prec * rap / (prec + rap)


def jaccard(a_list: list[str], b_list: list[str]) -> float:
    a = {_normaliser(x) for x in a_list if x}
    b = {_normaliser(x) for x in b_list if x}
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extraire_json(texte: str) -> dict | None:
    texte = (texte or "").strip()
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
    prof = 0
    for i, c in enumerate(texte[debut:], start=debut):
        if c == "{":
            prof += 1
        elif c == "}":
            prof -= 1
            if prof == 0:
                try:
                    return json.loads(texte[debut : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def valider_fiche(obj: object) -> bool:
    if not isinstance(obj, dict) or any(k not in obj for k in FICHE_KEYS):
        return False
    if not all(isinstance(obj[k], str) for k in ("formation", "solution", "resume")):
        return False
    return isinstance(obj["articles_vises"], list)


def evaluer(sorties: list[str], references: list[dict]) -> dict:
    n = len(sorties)
    n_json = n_schema = 0
    qual: dict[str, list[float]] = {
        "solution_exacte": [],
        "formation_exacte": [],
        "articles_jaccard": [],
        "resume_rouge_l": [],
    }
    for brut, ref in zip(sorties, references, strict=True):
        fiche = extraire_json(brut)
        if fiche is None:
            continue
        n_json += 1
        if not valider_fiche(fiche):
            continue
        n_schema += 1
        qual["solution_exacte"].append(
            float(_normaliser(fiche["solution"]) == _normaliser(ref["solution"]))
        )
        qual["formation_exacte"].append(
            float(_normaliser(fiche["formation"]) == _normaliser(ref["formation"]))
        )
        qual["articles_jaccard"].append(jaccard(fiche["articles_vises"], ref["articles_vises"]))
        qual["resume_rouge_l"].append(rouge_l(fiche["resume"], ref["resume"]))

    res = {"n": n, "json_valide": n_json / n, "schema_valide": n_schema / n}
    for cle, vals in qual.items():
        res[cle] = sum(vals) / len(vals) if vals else 0.0
    return res


def main() -> None:
    print(f"Modèle              : {MODELE_BASE}")
    print(f"Adaptateur          : {ADAPTATEUR or '(aucun — modèle de base)'}")
    print(f"max_new_tokens      : {MAX_NOUVEAUX_TOKENS}")
    print(f"repetition_penalty  : {REPETITION_PENALTY}", flush=True)

    donnees = load_dataset(DATASET_TRAVAIL)
    test = donnees["test"].select(range(min(N_EVAL, len(donnees["test"]))))
    prompts = list(test["prompt"])
    refs = [json.loads(c) for c in test["completion"]]

    tokenizer = AutoTokenizer.from_pretrained(MODELE_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    modele = AutoModelForCausalLM.from_pretrained(
        MODELE_BASE, quantization_config=quant, dtype=torch.bfloat16, device_map="auto"
    )
    if ADAPTATEUR:
        modele = PeftModel.from_pretrained(modele, ADAPTATEUR)
        print("Adaptateur chargé.", flush=True)
    else:
        print("Aucun adaptateur — évaluation du modèle de base.", flush=True)
    modele.eval()

    sorties: list[str] = []
    lot = 4
    for debut in range(0, len(prompts), lot):
        groupe = prompts[debut : debut + lot]
        textes = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in groupe
        ]
        entrees = tokenizer(
            textes, return_tensors="pt", padding=True, truncation=True, max_length=LONGUEUR_MAX
        ).to(modele.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen = modele.generate(
                **entrees,
                max_new_tokens=MAX_NOUVEAUX_TOKENS,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
            )
        for i in range(len(groupe)):
            sorties.append(
                tokenizer.decode(gen[i][entrees["input_ids"].shape[1] :], skip_special_tokens=True)
            )
        print(f"  généré {min(debut + lot, len(prompts))}/{len(prompts)}", flush=True)

    resultat = evaluer(sorties, refs)
    resultat["modele"] = MODELE_BASE
    resultat["adaptateur"] = ADAPTATEUR or None
    resultat["max_new_tokens"] = MAX_NOUVEAUX_TOKENS
    resultat["repetition_penalty"] = REPETITION_PENALTY
    print("\nRESULTAT_V2 :", json.dumps(resultat, indent=2), flush=True)

    # Suffixe dérivé du modèle : les jobs tournent en parallèle et publient
    # dans le même dépôt, ils ne doivent pas s'écraser mutuellement.
    slug = os.environ.get("SLUG", MODELE_BASE.split("/")[-1].lower())
    f_res = f"bench_{slug}.json"
    f_out = f"bench_{slug}_sorties.jsonl"

    (SORTIE / f_res).write_text(json.dumps(resultat, indent=2))
    (SORTIE / f_out).write_text(
        "\n".join(json.dumps({"sortie": s}, ensure_ascii=False) for s in sorties)
    )

    if DEPOT_SORTIE:
        api = HfApi()
        for f in (f_res, f_out):
            api.upload_file(
                path_or_fileobj=str(SORTIE / f),
                path_in_repo=f,
                repo_id=DEPOT_SORTIE,
                repo_type="model",
            )
        print(f"Publié : https://huggingface.co/{DEPOT_SORTIE}", flush=True)


if __name__ == "__main__":
    main()
