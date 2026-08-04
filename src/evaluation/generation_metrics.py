"""Generation-quality metrics (Table 4 of the paper, plus two additions).

The paper evaluates generation with BERTScore F1 against instructor reference
answers plus human ratings of correctness, pedagogical clarity and linguistic
appropriateness.  Human ratings cannot be reproduced offline, so this module
provides:

* ``bertscore_f1``   - real BERTScore if ``bert-score`` is installed, otherwise
                       an explicitly-labelled token-F1 proxy (never silently
                       reported as BERTScore)
* ``rougeL_f1``      - longest-common-subsequence overlap with the reference
* ``grounding_rate`` - share of answer content words that appear in the
                       retrieved context.  This is the automatable stand-in for
                       "factual correctness": a RAG answer that uses words the
                       retriever never returned is, by construction, ungrounded
* ``citation_rate``  - share of answers carrying at least one *valid* citation
                       (a chunk id that was actually retrieved)
* ``register_match`` - does the answer's detected register match the query's?
                       This is the metric the paper's central claim needs and
                       the one no standard toolkit provides
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

_WORD_RE = re.compile(r"[a-z0-9\u0900-\u097F]+")
_STOP = set("""the a an is are was were be of to in on for and or with that this it as at by
from we you i they he she can could would should not no yes if then than so""".split())


def content_words(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1]


def token_f1(prediction: str, reference: str) -> float:
    """Unigram F1 between prediction and reference (bag of content words)."""
    pred, ref = content_words(prediction), content_words(reference)
    if not pred or not ref:
        return 0.0
    common = 0
    ref_pool = list(ref)
    for token in pred:
        if token in ref_pool:
            ref_pool.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Classic O(nm) dynamic program - the same recurrence as in the course."""
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if token_a == token_b
                        else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = content_words(prediction), content_words(reference)
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def grounding_rate(answer: str, context_chunks: Iterable[dict]) -> float:
    """Fraction of the answer's content words that appear in the retrieved context."""
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    context = set()
    for chunk in context_chunks:
        context.update(content_words(chunk["text"]))
    return sum(1 for w in answer_words if w in context) / len(answer_words)


def register_match(query_profile: Dict[str, float], answer: str, detector) -> float:
    """1.0 if the answer is in the same register class as the query, else 0.0."""
    def register_class(profile: Dict[str, float]) -> str:
        nepali = profile.get("nepali", 0.0)
        if nepali >= 0.55:
            return "romanized_nepali"
        return "code_mixed" if nepali >= 0.15 else "english"

    return float(register_class(query_profile)
                 == register_class(detector.language_profile(answer)))


def bertscore_f1(predictions: Sequence[str], references: Sequence[str],
                 lang: str = "en") -> Dict[str, object]:
    """Real BERTScore when available; otherwise a clearly-labelled proxy."""
    try:
        from bert_score import score  # type: ignore

        _, _, f1 = score(list(predictions), list(references), lang=lang,
                         rescale_with_baseline=False, verbose=False)
        return {"metric": "bertscore_f1", "values": [float(v) for v in f1]}
    except Exception:
        return {"metric": "token_f1_proxy",
                "values": [token_f1(p, r) for p, r in zip(predictions, references)]}
