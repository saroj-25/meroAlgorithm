

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"([a-z])\1{2,}")          # "kastooooo" -> "kastoo"
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def normalize_text(text: str, form: str = "NFKC", collapse_repeats: bool = True) -> str:
    """Normalize Unicode, strip zero-width characters and collapse whitespace.

    Repeated-character collapsing matters for Romanized Nepali, where informal
    lengthening ("sajilooooo") is common and would otherwise fragment tokens.
    """
    if text is None:
        return ""
    text = unicodedata.normalize(form, str(text))
    text = text.translate(_ZERO_WIDTH)
    if collapse_repeats:
        text = _REPEAT_RE.sub(r"\1\1", text)
    return _WS_RE.sub(" ", text).strip()
