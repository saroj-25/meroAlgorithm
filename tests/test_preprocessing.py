"""Language detection, normalization, transliteration and chunking."""

from src.preprocessing.chunking import chunk_text
from src.preprocessing.language_id import RuleBasedLID
from src.preprocessing.lexicons import is_technical, rule_label
from src.preprocessing.normalize import normalize_text
from src.preprocessing.query_pipeline import preprocess_query
from src.preprocessing.transliterate import romanized_to_devanagari


def test_normalization_collapses_lengthening_and_whitespace():
    assert normalize_text("sajilooooo   cha") == "sajiloo cha"


def test_technical_tokens_are_flagged():
    for token in ["O(n)", "DFS", "arr[i]", "i++", "42"]:
        assert is_technical(token), token
    assert not is_technical("kasari")


def test_rule_labels():
    assert rule_label("kasari") == "nepali"
    assert rule_label("complexity") == "english"
    assert rule_label("O(n log n)".split()[0]) == "ambiguous"


def test_query_type_routing():
    detector = RuleBasedLID()
    assert detector.classify_query("what is the complexity of merge sort") == "english"
    assert detector.classify_query("yo kasari kaam garcha hola bhanera") == "romanized_nepali"
    assert detector.classify_query(
        "yo time complexity kasari nikalne") in {"code_mixed", "romanized_nepali"}


def test_transliteration_produces_devanagari():
    out = romanized_to_devanagari("kasari")
    assert any("\u0900" <= ch <= "\u097F" for ch in out)


def test_romanized_query_is_expanded(cfg):
    detector = RuleBasedLID()
    processed = preprocess_query("yo kasari kaam garcha hola bhanera", detector, cfg)
    if processed.query_type == "romanized_nepali":
        assert processed.expanded and "[SEP]" in processed.encode_text
    # BM25 must always see the un-expanded text
    assert "[SEP]" not in processed.lexical_text


def test_chunking_respects_token_band():
    text = "\n\n".join(" ".join(f"word{i}" for i in range(120)) for _ in range(8))
    chunks = chunk_text(text, min_tokens=250, max_tokens=450, overlap=50)
    assert chunks
    assert all(len(c.split()) <= 450 + 50 for c in chunks)
