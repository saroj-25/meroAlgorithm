"""Query preprocessing: the three branches of Section 3.3.

    pure English            -> Unicode normalization only
    pure Romanized Nepali   -> normalization + optional Devanagari expansion
                               ("roman [SEP] devanagari", both encoded together)
    code-mixed              -> normalization only; the string is deliberately
                               NOT split, because splitting degraded retrieval
                               in the paper's pilot experiments

The output object carries everything downstream components need: the text to
encode, the text to hand to BM25, the detected register, and the token-level
language profile that the generator uses to preserve the learner's register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .normalize import normalize_text
from .transliterate import romanized_to_devanagari


@dataclass
class ProcessedQuery:
    raw: str
    normalized: str
    encode_text: str            # what the dense encoder sees (may be expanded)
    lexical_text: str           # what BM25 sees (never expanded: keeps IDF clean)
    query_type: str             # english | romanized_nepali | code_mixed
    language_profile: Dict[str, float] = field(default_factory=dict)
    expanded: bool = False
    devanagari: str = ""

    def as_dict(self) -> dict:
        return {
            "raw": self.raw, "normalized": self.normalized,
            "query_type": self.query_type, "expanded": self.expanded,
            "language_profile": {k: round(v, 3) for k, v in self.language_profile.items()},
        }


def preprocess_query(text: str, detector, cfg=None,
                     force_language: str | None = None) -> ProcessedQuery:
    """Run the branch selected by the code-switch detector.

    ``force_language`` implements the UI's explicit "my question contains Nepali"
    toggle (Section 4), which lets a motivated user override the detector.
    """
    form = cfg.get("preprocessing.unicode_normalization", "NFKC") if cfg else "NFKC"
    do_translit = cfg.get("preprocessing.transliterate_romanized", True) if cfg else True
    separator = cfg.get("preprocessing.expansion_separator", " [SEP] ") if cfg else " [SEP] "

    normalized = normalize_text(text, form=form)
    profile = detector.language_profile(normalized)
    query_type = force_language or detector.classify_query(normalized)

    encode_text, expanded, devanagari = normalized, False, ""
    if query_type == "romanized_nepali" and do_translit:
        devanagari = romanized_to_devanagari(normalized)
        if devanagari and devanagari != normalized:
            encode_text = f"{normalized}{separator}{devanagari}"
            expanded = True

    return ProcessedQuery(raw=text, normalized=normalized, encode_text=encode_text,
                          lexical_text=normalized, query_type=query_type,
                          language_profile=profile, expanded=expanded,
                          devanagari=devanagari)
