"""Build the evaluation sets used in Sections 5.5 and 6.1 of the paper.

Two artefacts are produced:

1. ``data/processed/eval_queries.jsonl`` - 300 annotated retrieval queries
   (100 English, 100 Romanized Nepali, 100 code-mixed).  Each query carries the
   set of *relevant* chunk ids, i.e. every chunk in the knowledge base that
   explains the same concept.  Binary relevance, as in the paper's P@5 / MRR /
   R@10 protocol.

2. ``data/processed/lid_dataset.jsonl`` - 4,000 token-labelled sentences for the
   character n-gram code-switch classifier of Section 3.3.  Token labels are
   produced by the transparent rule labeller in ``preprocessing/lexicons.py``;
   the classifier's job is to *generalise beyond the lexicon* using character
   n-grams, which is exactly what makes it useful on unseen spellings.

Run::

    python -m src.data.build_eval_sets --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

from ..preprocessing.lexicons import rule_label, tokenize
from ..utils.config import Config, load_config
from ..utils.io import read_jsonl, write_jsonl
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from .dsa_content import ALL_CONCEPTS

log = get_logger("data.eval_sets")

# --------------------------------------------------------------------------- #
# Query templates.  {a} = concept alias, {n} = concept display name.
# --------------------------------------------------------------------------- #
ENGLISH_TEMPLATES = [
    "what is {a}",
    "explain {a} with an example",
    "how does {a} work",
    "time complexity of {a}",
    "how do I implement {a}",
    "why do we use {a}",
    "common mistakes in {a}",
    "{a} step by step explanation",
    "when should {a} be used",
    "worst case analysis of {a}",
]

# Pure Romanized Nepali: Nepali sentence frame, technical term as a loanword
# (this is how learners actually type it - see Section 2.3).
ROMANIZED_TEMPLATES = [
    "{a} bhaneko k ho",
    "{a} kasari kaam garcha",
    "{a} ko complexity kasari nikalne",
    "{a} kina prayog garne",
    "{a} ma prayah kun galti huncha",
    "{a} sajilo bhasama bujhaunus",
    "{a} ko example dinus na",
    "{a} kahile use garne hola",
    "{a} ko logic bujhena malai",
    "{a} ra tyo related concept ma k farak cha",
]

# Code-mixed: Nepali interrogative / framing word + English technical noun
# phrase, the dominant pattern reported in Section 6.3 of the paper.
CODE_MIXED_TEMPLATES = [
    "yo {a} ko time complexity kasari nikalne",
    "{a} ko base case kasari define garne hola",
    "{a} bhaneko k ho, short ma explain garnus",
    "{a} implement garda kun edge case dhyan dine",
    "{a} ma worst case kina O(n) huncha",
    "sir {a} ko intuition ali sajilo gari bhannus na",
    "{a} vs arko approach, kun better huncha exam ma",
    "{a} ko code lekhda common mistake k ho",
    "{a} ko invariant kasari prove garne",
    "malai {a} ko dry run ek choti dekhaunus",
]

TEMPLATES = {
    "english": ENGLISH_TEMPLATES,
    "romanized_nepali": ROMANIZED_TEMPLATES,
    "code_mixed": CODE_MIXED_TEMPLATES,
}


def _alias_for(concept: dict, idx: int) -> str:
    aliases = concept["aliases"] or [concept["name"]]
    return aliases[idx % len(aliases)]


def build_eval_queries(chunks: List[dict], n_per_language: int = 100) -> List[dict]:
    """Create the annotated retrieval evaluation set."""
    concept_to_chunks: Dict[str, List[str]] = defaultdict(list)
    concept_meta: Dict[str, dict] = {}
    for chunk in chunks:
        concept_to_chunks[chunk["concept"]].append(chunk["chunk_id"])
        concept_meta[chunk["concept"]] = {"topic": chunk["topic"],
                                          "name": chunk["concept_name"]}

    concepts = [c for c in ALL_CONCEPTS if c["key"] in concept_to_chunks]
    queries: List[dict] = []

    for lang, templates in TEMPLATES.items():
        for i in range(n_per_language):
            concept = concepts[(i * 7 + 3) % len(concepts)]   # spread over topics
            template = templates[i % len(templates)]
            alias = _alias_for(concept, i // len(templates))
            text = template.format(a=alias, n=concept["name"])
            queries.append({
                "query_id": f"{lang[:2].upper()}-{i:03d}",
                "query": text,
                "query_type": lang,
                "concept": concept["key"],
                "topic": concept_meta[concept["key"]]["topic"],
                "relevant_chunk_ids": sorted(concept_to_chunks[concept["key"]]),
                "n_relevant": len(concept_to_chunks[concept["key"]]),
            })
    return queries


# --------------------------------------------------------------------------- #
# Language-ID training corpus
# --------------------------------------------------------------------------- #
NEPALI_SENTENCES = [
    "yo concept malai ekdam garho lagyo sir",
    "pahila theory bujhnu parcha ani code lekhnu parcha",
    "exam ma yo prashna prayah sodhincha bhanne suneko thiye",
    "malai yo step kina yesari garne bhanera bujhena",
    "ekchoti sajilo bhasama bujhaidinus na",
    "aaja ko class ma bhaneko kura ali chito bhayo",
    "yo galti maile pahila pani gareko thiye",
    "dry run garda chai answer milyo tara code ma bigriyo",
    "yesto samasya aayo bhane k garne hola",
    "arko example dinus na malai practice garnu cha",
    "timro logic thik cha tara complexity badhi bhayo",
    "hamile lab ma yo problem solve garisakeko chau",
]

ENGLISH_SENTENCES = [
    "explain the difference between these two data structures",
    "what is the worst case time complexity of this algorithm",
    "i do not understand why this loop runs n times",
    "please give me a simple example with five elements",
    "how should i handle the empty input case here",
    "the recursion depth affects the space complexity as well",
    "can you show the invariant that this loop maintains",
    "this solution passes the sample tests but fails on large inputs",
    "we need an ordered structure so a hash map will not work",
    "the amortized cost of an append is constant",
]

CODE_MIXED_SENTENCES = [
    "yo binary search ko lower bound bhaneko k ho",
    "recursion ma base case kasari define garne hola",
    "time complexity nikalda log n kina aaucha",
    "merge sort ko stability bhaneko k ho sir",
    "dijkstra algorithm negative weight ma kina fail huncha",
    "hash map use garda collision handle kasari garne",
    "yo dp table ko state design kasari garne bhanne confusion cha",
    "linked list reverse garda prev pointer kina chahincha",
    "graph ma cycle detection garne tarika k ho",
    "heap ma sift down operation kasari kaam garcha",
    "quick sort ko pivot selection le complexity ma k asar garcha",
    "backtracking ma pruning garda kati fast huncha",
]


def build_lid_dataset(n_sentences: int = 4000) -> List[dict]:
    """Token-labelled sentences (nepali / english / ambiguous)."""
    pools = [
        ("romanized_nepali", NEPALI_SENTENCES),
        ("english", ENGLISH_SENTENCES),
        ("code_mixed", CODE_MIXED_SENTENCES),
    ]
    # Also mine the query templates so the classifier sees query-style text.
    for lang, templates in TEMPLATES.items():
        sents = [t.format(a=_alias_for(c, i), n=c["name"])
                 for i, (t, c) in enumerate(
                     zip(templates * 20, ALL_CONCEPTS[:len(templates) * 20]))]
        pools.append((lang, sents))

    rows: List[dict] = []
    i = 0
    while len(rows) < n_sentences:
        lang, pool = pools[i % len(pools)]
        sentence = pool[(i // len(pools)) % len(pool)]
        tokens = tokenize(sentence)
        rows.append({
            "sentence_id": f"LID-{len(rows):05d}",
            "text": sentence,
            "sentence_language": lang,
            "tokens": tokens,
            "labels": [rule_label(t) for t in tokens],
        })
        i += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation datasets")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg: Config = load_config(args.config)
    set_seed(cfg.get("project.seed", 42))
    processed = cfg.path("paths.processed_dir")

    chunks = read_jsonl(processed / "knowledge_base.jsonl")
    n_per = cfg.get("evaluation.n_eval_queries_per_language", 100)

    queries = build_eval_queries(chunks, n_per)
    write_jsonl(processed / "eval_queries.jsonl", queries)
    log.info("Wrote %d evaluation queries (%d per language) -> %s",
             len(queries), n_per, processed / "eval_queries.jsonl")

    lid = build_lid_dataset(4000)
    write_jsonl(processed / "lid_dataset.jsonl", lid)
    n_tokens = sum(len(r["tokens"]) for r in lid)
    log.info("Wrote %d language-ID sentences (%d labelled tokens) -> %s",
             len(lid), n_tokens, processed / "lid_dataset.jsonl")


if __name__ == "__main__":
    main()
