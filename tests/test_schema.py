from src.schema import extraire_json, valider_fiche

FICHE_OK = {
    "formation": "Chambre commerciale",
    "solution": "Cassation partielle",
    "articles_vises": ["Article 1186 du code civil"],
    "resume": "Un résumé.",
}


def test_fiche_valide():
    ok, motif = valider_fiche(FICHE_OK)
    assert ok, motif


def test_cle_manquante():
    incomplete = {k: v for k, v in FICHE_OK.items() if k != "resume"}
    ok, motif = valider_fiche(incomplete)
    assert not ok
    assert "resume" in motif


def test_articles_doit_etre_une_liste():
    mauvais = FICHE_OK | {"articles_vises": "Article 1186"}
    ok, _ = valider_fiche(mauvais)
    assert not ok


def test_rejette_non_dict():
    ok, _ = valider_fiche("pas un objet")
    assert not ok


def test_extraire_json_direct():
    assert extraire_json('{"a": 1}') == {"a": 1}


def test_extraire_json_dans_bloc_markdown():
    texte = 'Voici la fiche :\n```json\n{"a": 1}\n```\nVoilà.'
    assert extraire_json(texte) == {"a": 1}


def test_extraire_json_avec_texte_autour():
    assert extraire_json('Réponse : {"a": {"b": 2}} fin') == {"a": {"b": 2}}


def test_extraire_json_absent():
    assert extraire_json("aucun json ici") is None
