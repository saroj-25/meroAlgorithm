

from __future__ import annotations

from typing import List

try:  # optional, higher-quality backend
    from indic_transliteration import sanscript  # type: ignore
    from indic_transliteration.sanscript import transliterate as _indic

    _HAS_INDIC = True
except Exception:  # pragma: no cover
    _HAS_INDIC = False

# Longest-match first: multi-character graphemes must precede single characters.
_CONSONANTS = [
    ("chh", "छ"), ("kha", "ख"), ("gha", "घ"), ("cha", "च"), ("jha", "झ"),
    ("tha", "थ"), ("dha", "ध"), ("pha", "फ"), ("bha", "भ"), ("sha", "श"),
    ("ksh", "क्ष"), ("gya", "ज्ञ"),
    ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"),
    ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("ng", "ङ"),
    ("ny", "ञ"), ("tt", "ट"), ("dd", "ड"),
    ("k", "क"), ("g", "ग"), ("c", "च"), ("j", "ज"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"),
    ("l", "ल"), ("w", "व"), ("v", "व"), ("s", "स"), ("h", "ह"), ("x", "क्स"),
    ("f", "फ"), ("z", "ज"), ("q", "क"),
]

_VOWEL_SIGNS = [
    ("aa", "ा"), ("ai", "ै"), ("au", "ौ"), ("ee", "ी"), ("oo", "ू"),
    ("a", ""), ("i", "ि"), ("u", "ु"), ("e", "े"), ("o", "ो"),
]

_VOWEL_INDEPENDENT = [
    ("aa", "आ"), ("ai", "ऐ"), ("au", "औ"), ("ee", "ई"), ("oo", "ऊ"),
    ("a", "अ"), ("i", "इ"), ("u", "उ"), ("e", "ए"), ("o", "ओ"),
]

_HALANTA = "्"


def _match(text: str, i: int, table) -> tuple:
    for src, dst in table:
        if text.startswith(src, i):
            return src, dst
    return "", ""


def transliterate_word(word: str) -> str:
    """Greedy syllable-wise Roman -> Devanagari transliteration of one word."""
    word = word.lower()
    out: List[str] = []
    i = 0
    at_start = True
    while i < len(word):
        cons_src, cons_dst = _match(word, i, _CONSONANTS)
        if cons_dst:
            i += len(cons_src)
            vow_src, vow_dst = _match(word, i, _VOWEL_SIGNS)
            if vow_src:
                i += len(vow_src)
                out.append(cons_dst + vow_dst)
            else:
                # No vowel follows: mark the consonant as bare with halanta.
                out.append(cons_dst + _HALANTA)
            at_start = False
            continue

        vow_src, vow_dst = _match(word, i, _VOWEL_INDEPENDENT if at_start else _VOWEL_SIGNS)
        if vow_src:
            i += len(vow_src)
            out.append(vow_dst)
            at_start = False
            continue

        out.append(word[i])          # digits, punctuation, unknown symbols
        i += 1
    return "".join(out)


def romanized_to_devanagari(text: str) -> str:
    """Transliterate a whole string, leaving non-alphabetic tokens untouched."""
    if _HAS_INDIC:
        try:
            return _indic(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        except Exception:  # pragma: no cover
            pass
    return " ".join(
        tok if not tok.isalpha() else transliterate_word(tok)
        for tok in text.split()
    )
