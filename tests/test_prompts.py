from src.prompts import construire_prompt


def test_prompt_contient_larret():
    assert "MON ARRÊT" in construire_prompt("MON ARRÊT")


def test_prompt_tronque_les_arrets_longs():
    long_texte = "mot " * 10000
    p = construire_prompt(long_texte, max_chars=100)
    assert "[…]" in p
    assert len(p) < 1000


def test_prompt_ne_tronque_pas_les_courts():
    assert "[…]" not in construire_prompt("court", max_chars=100)
