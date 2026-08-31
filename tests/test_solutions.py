from src.clean import normaliser_solution


def test_casse_normalisee():
    assert normaliser_solution("REJET") == "Rejet"
    assert normaliser_solution("Rejet") == "Rejet"
    assert normaliser_solution("rejet") == "Rejet"


def test_variante_la_plus_specifique_gagne():
    assert normaliser_solution("Cassation partielle sans renvoi") == (
        "Cassation partielle sans renvoi"
    )
    assert normaliser_solution("Cassation partielle") == "Cassation partielle"
    assert normaliser_solution("Cassation") == "Cassation"


def test_valeur_composite_rejetee():
    assert normaliser_solution("Cassation partielle REJET Cassation") == ""


def test_valeur_inconnue_rejetee():
    assert normaliser_solution("Blabla") == ""
    assert normaliser_solution("") == ""


def test_espaces_multiples():
    assert normaliser_solution("  Cassation   partielle  ") == "Cassation partielle"
