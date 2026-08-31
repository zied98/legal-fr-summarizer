from src.clean import nettoyer_arret, normaliser_articles, normaliser_formation


def test_nettoyer_supprime_les_br():
    assert nettoyer_arret("LA COUR<br><br>a rendu") == "LA COUR\n\na rendu"


def test_nettoyer_decode_les_entites():
    assert "'" in nettoyer_arret("l&#039;arrêt") or "'" in nettoyer_arret("l&#039;arrêt")


def test_nettoyer_gere_le_vide():
    assert nettoyer_arret("") == ""


def test_normaliser_formation():
    assert normaliser_formation("CHAMBRE_COMMERCIALE") == "Chambre commerciale"
    assert normaliser_formation("") == ""


def test_normaliser_articles_simple():
    assert normaliser_articles("Article 1186 du code civil.") == ["Article 1186 du code civil"]


def test_normaliser_articles_multiples():
    res = normaliser_articles("Article 1 du code civil ; Article 2 du code pénal")
    assert res == ["Article 1 du code civil", "Article 2 du code pénal"]


def test_normaliser_articles_vide():
    assert normaliser_articles("") == []
