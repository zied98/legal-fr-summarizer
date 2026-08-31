import json

from src.evaluate import agreger, evaluer_exemple, jaccard, rouge_l

REF = {
    "formation": "Chambre commerciale",
    "solution": "Cassation partielle",
    "articles_vises": ["Article 1186 du code civil"],
    "resume": "Le contrat est caduc lorsque l'operation disparait.",
}


def test_rouge_identique_vaut_un():
    assert rouge_l("le chat dort", "le chat dort") == 1.0


def test_rouge_disjoint_vaut_zero():
    assert rouge_l("abc def", "xyz uvw") == 0.0


def test_rouge_partiel_entre_zero_et_un():
    score = rouge_l("le chat noir dort", "le chat dort")
    assert 0.0 < score < 1.0


def test_rouge_ignore_accents_et_casse():
    assert rouge_l("ÉTÉ", "ete") == 1.0


def test_jaccard_identique():
    assert jaccard(["Article 1"], ["Article 1"]) == 1.0


def test_jaccard_deux_listes_vides():
    assert jaccard([], []) == 1.0


def test_jaccard_une_seule_vide():
    assert jaccard(["Article 1"], []) == 0.0


def test_evaluer_sortie_parfaite():
    r = evaluer_exemple(json.dumps(REF, ensure_ascii=False), REF)
    assert r["json_valide"] == 1.0
    assert r["schema_valide"] == 1.0
    assert r["solution_exacte"] == 1.0
    assert r["formation_exacte"] == 1.0
    assert r["resume_rouge_l"] == 1.0


def test_evaluer_json_invalide():
    r = evaluer_exemple("je ne sais pas repondre", REF)
    assert r["json_valide"] == 0.0
    assert "solution_exacte" not in r


def test_evaluer_schema_incomplet():
    r = evaluer_exemple('{"solution": "Rejet"}', REF)
    assert r["json_valide"] == 1.0
    assert r["schema_valide"] == 0.0


def test_evaluer_mauvaise_solution():
    mauvais = REF | {"solution": "Rejet"}
    r = evaluer_exemple(json.dumps(mauvais, ensure_ascii=False), REF)
    assert r["solution_exacte"] == 0.0
    assert r["formation_exacte"] == 1.0


def test_agreger_moyenne_correctement():
    a = agreger(
        [
            {"json_valide": 1.0, "schema_valide": 1.0, "solution_exacte": 1.0},
            {"json_valide": 1.0, "schema_valide": 1.0, "solution_exacte": 0.0},
        ]
    )
    assert a["n"] == 2
    assert a["json_valide"] == 1.0
    assert a["solution_exacte"] == 0.5


def test_agreger_ignore_les_invalides_pour_la_qualite():
    a = agreger(
        [
            {"json_valide": 0.0, "schema_valide": 0.0},
            {"json_valide": 1.0, "schema_valide": 1.0, "solution_exacte": 1.0},
        ]
    )
    assert a["json_valide"] == 0.5
    assert a["solution_exacte"] == 1.0


def test_agreger_liste_vide():
    assert agreger([]) == {}
