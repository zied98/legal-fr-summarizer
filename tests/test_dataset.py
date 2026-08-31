from src.dataset import construire_exemple, exemple_utilisable

LIGNE_OK = {
    "id": "JURITEXT000",
    "content": "LA COUR DE CASSATION<br>" + "texte de l'arrêt. " * 200,
    "summary": "Un sommaire officiel suffisamment long pour être retenu. " * 5,
    "solution": "Cassation partielle",
    "formation": "CHAMBRE_COMMERCIALE",
    "applied_laws": "Article 1186 du code civil.",
}


def test_ligne_complete_est_utilisable():
    assert exemple_utilisable(LIGNE_OK)


def test_rejette_resume_trop_court():
    assert not exemple_utilisable(LIGNE_OK | {"summary": "trop court"})


def test_rejette_solution_absente():
    assert not exemple_utilisable(LIGNE_OK | {"solution": ""})


def test_rejette_articles_absents():
    assert not exemple_utilisable(LIGNE_OK | {"applied_laws": ""})


def test_rejette_champ_none():
    assert not exemple_utilisable(LIGNE_OK | {"formation": None})


def test_construire_exemple_produit_les_bonnes_cles():
    ex = construire_exemple(LIGNE_OK)
    assert ex is not None
    assert set(ex) == {"id", "prompt", "completion", "reference"}


def test_construire_exemple_normalise_la_formation():
    ex = construire_exemple(LIGNE_OK)
    assert ex["reference"]["formation"] == "Chambre commerciale"


def test_construire_exemple_rejette_arret_trop_court():
    assert construire_exemple(LIGNE_OK | {"content": "court"}) is None


def test_completion_est_du_json_valide():
    import json

    ex = construire_exemple(LIGNE_OK)
    assert json.loads(ex["completion"]) == ex["reference"]
