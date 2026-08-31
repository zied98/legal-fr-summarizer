"""Fine-tuning LoRA + évaluation avant/après, exécuté comme Job Hugging Face.

Script UV autonome : les dépendances sont déclarées dans l'en-tête ci-dessous,
UV les installe avant l'exécution. Aucune image Docker à construire.

Déroulé en un seul job, pour ne payer le GPU qu'une fois :
  1. Évaluation du modèle de base   -> baseline
  2. Fine-tuning LoRA (QLoRA 4-bit)
  3. Évaluation du modèle entraîné  -> comparaison
  4. Publication de l'adaptateur et des résultats sur le Hub

Le jeu de test n'est jamais vu pendant l'entraînement.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers==4.57.6",
#   "peft==0.17.1",
#   "trl==0.25.0",
#   "datasets>=3.0,<5",
#   "accelerate>=1.0",
#   "bitsandbytes>=0.44",
#   "huggingface_hub>=0.26",
# ]
# ///

# NOTE: versions figées volontairement. transformers 5.x a supprimé
# `warmup_ratio` de TrainingArguments, ce qui casse SFTConfig. Un job GPU
# qui plante après 19 minutes coûte de l'argent : on ne laisse pas les
# dépendances flotter.

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import unicodedata
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# --------------------------------------------------------------------------
# Configuration (surchargeable par variables d'environnement)
# --------------------------------------------------------------------------
MODELE_BASE = os.environ.get("MODELE_BASE", "Qwen/Qwen3-4B-Instruct-2507")
DATASET_TRAVAIL = os.environ["DATASET_TRAVAIL"]  # dataset préparé, poussé sur le Hub
DEPOT_SORTIE = os.environ["DEPOT_SORTIE"]  # où publier l'adaptateur

N_EVAL = int(os.environ.get("N_EVAL", "150"))
#: Baseline déjà mesurée lors d'un run précédent : évite de repayer 19 min de
#: GPU pour un résultat connu. Vide = on la recalcule.
BASELINE_CONNUE = os.environ.get("BASELINE_CONNUE", "")
#: Limite le jeu d'entraînement. Sert au smoke test : valider le pipeline
#: complet en quelques minutes avant d'engager une heure de GPU.
N_TRAIN_MAX = int(os.environ.get("N_TRAIN_MAX", "0")) or None
EPOQUES = float(os.environ.get("EPOQUES", "2"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LONGUEUR_MAX = int(os.environ.get("LONGUEUR_MAX", "4096"))
MAX_NOUVEAUX_TOKENS = int(os.environ.get("MAX_NOUVEAUX_TOKENS", "700"))

SORTIE = Path("/tmp/sortie")
SORTIE.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "Tu es un assistant juridique spécialisé dans la jurisprudence française. "
    "Tu analyses un arrêt et produis une fiche structurée au format JSON."
)

# --------------------------------------------------------------------------
# Métriques (dupliquées ici pour que le script reste autonome)
# --------------------------------------------------------------------------
_MOT = re.compile(r"\w+", re.UNICODE)
FICHE_KEYS = ("formation", "solution", "articles_vises", "resume")


def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFD", texte or "")
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return " ".join(texte.lower().split())


def _tokens(texte: str) -> list[str]:
    return _MOT.findall(_normaliser(texte))


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
    stats = {"n": n, "json_valide": 0, "schema_valide": 0}
    qualite = {
        "solution_exacte": [],
        "formation_exacte": [],
        "articles_jaccard": [],
        "resume_rouge_l": [],
    }
    for brut, ref in zip(sorties, references, strict=True):
        fiche = extraire_json(brut)
        if fiche is None:
            continue
        stats["json_valide"] += 1
        if not valider_fiche(fiche):
            continue
        stats["schema_valide"] += 1
        qualite["solution_exacte"].append(
            float(_normaliser(fiche["solution"]) == _normaliser(ref["solution"]))
        )
        qualite["formation_exacte"].append(
            float(_normaliser(fiche["formation"]) == _normaliser(ref["formation"]))
        )
        qualite["articles_jaccard"].append(jaccard(fiche["articles_vises"], ref["articles_vises"]))
        qualite["resume_rouge_l"].append(rouge_l(fiche["resume"], ref["resume"]))

    resultat = {
        "n": n,
        "json_valide": stats["json_valide"] / n,
        "schema_valide": stats["schema_valide"] / n,
    }
    for cle, valeurs in qualite.items():
        resultat[cle] = sum(valeurs) / len(valeurs) if valeurs else 0.0
    return resultat


# --------------------------------------------------------------------------
# Génération
# --------------------------------------------------------------------------
def generer(modele, tokenizer, prompts: list[str], lot: int = 4) -> list[str]:
    """Génère les fiches pour une liste de prompts, par lots."""
    modele.eval()
    sorties: list[str] = []
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
        with torch.no_grad():
            gen = modele.generate(
                **entrees,
                max_new_tokens=MAX_NOUVEAUX_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for i in range(len(groupe)):
            nouveaux = gen[i][entrees["input_ids"].shape[1] :]
            sorties.append(tokenizer.decode(nouveaux, skip_special_tokens=True))
        print(f"  généré {min(debut + lot, len(prompts))}/{len(prompts)}", flush=True)
    return sorties


def main() -> None:
    print(f"Modèle de base : {MODELE_BASE}", flush=True)
    print(f"Dataset        : {DATASET_TRAVAIL}", flush=True)

    donnees = load_dataset(DATASET_TRAVAIL)
    train = donnees["train"]
    if N_TRAIN_MAX:
        train = train.select(range(min(N_TRAIN_MAX, len(train))))
        print(f"⚠️  SMOKE TEST : entraînement limité à {len(train)} exemples", flush=True)
    test = donnees["test"].select(range(min(N_EVAL, len(donnees["test"]))))

    prompts_test = list(test["prompt"])
    refs_test = [json.loads(c) for c in test["completion"]]
    print(f"train={len(train)}  test={len(prompts_test)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODELE_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # requis pour la génération par lots

    quantification = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if BASELINE_CONNUE:
        print("\n=== 1/3  Baseline reprise d'un run précédent ===", flush=True)
        baseline = json.loads(BASELINE_CONNUE)
        print("BASELINE :", json.dumps(baseline, indent=2), flush=True)
    else:
        print("\n=== 1/3  Baseline (modèle non entraîné) ===", flush=True)
        modele = AutoModelForCausalLM.from_pretrained(
            MODELE_BASE,
            quantization_config=quantification,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        sorties_base = generer(modele, tokenizer, prompts_test)
        baseline = evaluer(sorties_base, refs_test)
        print("BASELINE :", json.dumps(baseline, indent=2), flush=True)
        (SORTIE / "sorties_baseline.jsonl").write_text(
            "\n".join(json.dumps({"sortie": s}, ensure_ascii=False) for s in sorties_base)
        )
        del modele
        torch.cuda.empty_cache()

    (SORTIE / "baseline.json").write_text(json.dumps(baseline, indent=2))

    print("\n=== 2/3  Fine-tuning LoRA ===", flush=True)

    def formater(exemple: dict) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": exemple["prompt"]},
                {"role": "assistant", "content": exemple["completion"]},
            ]
        }

    train_fmt = Dataset.from_list([formater(x) for x in train])

    modele = AutoModelForCausalLM.from_pretrained(
        MODELE_BASE,
        quantization_config=quantification,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    modele.config.use_cache = False

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Les versions de transformers/TRL changent régulièrement de signature.
    # On ne passe que les paramètres réellement acceptés : un job GPU qui
    # plante sur un TypeError coûte de l'argent.
    voulus = {
        "output_dir": str(SORTIE / "adapter"),
        "num_train_epochs": EPOQUES,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "no",
        "bf16": True,
        "max_length": LONGUEUR_MAX,
        "gradient_checkpointing": True,
        "report_to": [],
    }
    acceptes = set(inspect.signature(SFTConfig.__init__).parameters)
    if "kwargs" in acceptes:  # dataclass héritée : on teste par construction
        acceptes = {c.name for c in dataclasses.fields(SFTConfig)}
    retenus = {k: v for k, v in voulus.items() if k in acceptes}
    ignores = sorted(set(voulus) - set(retenus))
    if ignores:
        print(f"Paramètres non supportés par cette version, ignorés : {ignores}", flush=True)
    config = SFTConfig(**retenus)

    entraineur = SFTTrainer(
        model=modele,
        train_dataset=train_fmt,
        peft_config=lora,
        args=config,
        processing_class=tokenizer,
    )
    entraineur.train()
    entraineur.save_model(str(SORTIE / "adapter"))
    print("Adaptateur sauvegardé.", flush=True)

    # Publication IMMÉDIATE de l'adaptateur, avant toute autre étape.
    # Un artefact qui a coûté une heure de GPU ne doit jamais dépendre
    # de la réussite des étapes suivantes.
    api = HfApi()
    api.create_repo(DEPOT_SORTIE, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(SORTIE / "adapter"),
        repo_id=DEPOT_SORTIE,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(SORTIE / "baseline.json"),
        path_in_repo="baseline.json",
        repo_id=DEPOT_SORTIE,
        repo_type="model",
    )
    print(f"✅ Adaptateur publié : https://huggingface.co/{DEPOT_SORTIE}", flush=True)

    print("\n=== 3/3  Évaluation du modèle entraîné ===", flush=True)
    modele_ft = entraineur.model
    modele_ft.config.use_cache = True
    # Après entraînement, certaines couches (dont lm_head) sont repassées en
    # float32 par le Trainer tandis que les états cachés restent en bfloat16.
    # torch.autocast réconcilie les deux au moment de la génération.
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sorties_ft = generer(modele_ft, tokenizer, prompts_test)
    except RuntimeError as err:
        # L'adaptateur est déjà publié : on ne perd pas l'entraînement.
        print(f"⚠️  Génération impossible ({err}) — repli en float32.", flush=True)
        modele_ft = modele_ft.float()
        sorties_ft = generer(modele_ft, tokenizer, prompts_test)
    finetune = evaluer(sorties_ft, refs_test)
    print("FINE-TUNÉ :", json.dumps(finetune, indent=2), flush=True)

    comparaison = {
        "modele_base": MODELE_BASE,
        "n_train": len(train),
        "epoques": EPOQUES,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "baseline": baseline,
        "finetune": finetune,
        "delta": {
            k: round(finetune[k] - baseline[k], 4)
            for k in finetune
            if k != "n" and isinstance(finetune[k], float)
        },
    }
    (SORTIE / "resultats.json").write_text(json.dumps(comparaison, indent=2, ensure_ascii=False))
    (SORTIE / "sorties_finetune.jsonl").write_text(
        "\n".join(json.dumps({"sortie": s}, ensure_ascii=False) for s in sorties_ft)
    )
    print("\nCOMPARAISON :", json.dumps(comparaison, indent=2), flush=True)

    print("\n=== Publication des résultats ===", flush=True)
    for fichier in ("resultats.json", "sorties_finetune.jsonl"):
        chemin = SORTIE / fichier
        if chemin.exists():
            api.upload_file(
                path_or_fileobj=str(chemin),
                path_in_repo=fichier,
                repo_id=DEPOT_SORTIE,
                repo_type="model",
            )
    print(f"Publié : https://huggingface.co/{DEPOT_SORTIE}", flush=True)


if __name__ == "__main__":
    main()
