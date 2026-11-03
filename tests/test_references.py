from rag.generate import is_abstention, strip_no_answer
from rag.lang import detect_language
from rag.references import find_references, reference_units


def test_article_with_law_after():
    assert find_references("¿Qué dice el artículo 22 del RGPD?") == [("gdpr", "art22")]


def test_several_articles_and_annex_in_german():
    assert find_references("Artikel 5 und 6 der KI-Verordnung sowie Anhang III") == [
        ("ai_act", "art5"), ("ai_act", "art6"), ("ai_act", "anxIII")]


def test_mixed_laws_and_recital():
    refs = find_references("Under Article 6(2) and Annex III of the AI Act, "
                           "and Article 35 GDPR, recital 47")
    assert ("ai_act", "art6") in refs and ("gdpr", "art35") in refs
    assert ("ai_act", "anxIII") in refs and ("gdpr", "rct47") in refs


def test_inserted_article_and_semicolon_boundary():
    assert reference_units("see Art. 4a AI Act; Art. 9 GDPR") == ["ai_act:art4a", "gdpr:art9"]


def test_bare_article_has_no_law():
    assert find_references("What does Article 6 say?") == [(None, "art6")]


def test_years_are_not_articles():
    assert find_references("The answer is in 2024") == []


def test_language_detection():
    assert detect_language("Which AI practices are prohibited?") == "en"
    assert detect_language("¿Un sistema de selección de personal con IA es de alto riesgo?") == "es"
    assert detect_language("Ist Emotionserkennung am Arbeitsplatz erlaubt?") == "de"


def test_abstention_detection():
    assert is_abstention("NO_ANSWER")
    assert is_abstention("**NO_ANSWER**")
    assert not is_abstention("Article 5 prohibits ... [1].\n\nNO_ANSWER")
    assert strip_no_answer("Article 5 prohibits ... [1].\n\nNO_ANSWER") == "Article 5 prohibits ... [1]."
