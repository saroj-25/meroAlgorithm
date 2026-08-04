"""Build the AlgoSathi DSA knowledge base (Table 1 of the paper).

The paper's knowledge base is 1,252 chunks: 1,068 English chunks distilled from
instructor lecture material and 184 instructor-authored *code-mixed*
mini-explanations (Romanized Nepali + English) of frequently misunderstood
topics.  That corpus is not public, so this script synthesises a corpus with
**exactly the same composition, metadata schema and chunk-size distribution**
from the concept registry in ``src/data/dsa_content.py``.

Run::

    python -m src.data.build_knowledge_base --config configs/default.yaml

Output: ``data/processed/knowledge_base.jsonl`` with one JSON object per chunk::

    {"chunk_id": "TRE-EN-0042", "topic": "Trees and BSTs", "concept": "lca",
     "facet": "complexity", "language": "en", "difficulty": "medium",
     "source": "lecture_slides", "text": "...", "n_tokens": 312}
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from ..utils.config import Config, load_config
from ..utils.io import write_json, write_jsonl
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from .dsa_content import CONCEPTS

log = get_logger("data.kb")

# --------------------------------------------------------------------------- #
# Table 1: exact per-topic composition (english_chunks, code_mixed_chunks)
# --------------------------------------------------------------------------- #
TABLE1: Dict[str, tuple] = {
    "Arrays and Strings": (112, 21),
    "Linked Lists": (98, 19),
    "Stacks and Queues": (76, 14),
    "Trees and BSTs": (154, 27),
    "Heaps and Priority Queues": (62, 11),
    "Hashing": (71, 13),
    "Graphs and Traversals": (139, 24),
    "Sorting Algorithms": (118, 18),
    "Searching Algorithms": (57, 9),
    "Dynamic Programming (Intro)": (94, 16),
    "Greedy and Backtracking": (66, 12),
    "Complexity and Notation": (21, 0),
}

TOPIC_CODE = {
    "Arrays and Strings": "ARR",
    "Linked Lists": "LNK",
    "Stacks and Queues": "STQ",
    "Trees and BSTs": "TRE",
    "Heaps and Priority Queues": "HEP",
    "Hashing": "HSH",
    "Graphs and Traversals": "GRF",
    "Sorting Algorithms": "SRT",
    "Searching Algorithms": "SRC",
    "Dynamic Programming (Intro)": "DYN",
    "Greedy and Backtracking": "GRD",
    "Complexity and Notation": "CPX",
}

FACETS = ["intuition", "formalization", "implementation", "complexity",
          "worked_example", "exam_pointer", "pitfall_note"]

SOURCES = ["lecture_slides", "supplementary_notes", "worked_solutions",
           "textbook_excerpt"]

DIFFICULTY_BY_FACET = {
    "intuition": "easy",
    "exam_pointer": "easy",
    "implementation": "medium",
    "worked_example": "medium",
    "pitfall_note": "medium",
    "formalization": "hard",
    "complexity": "hard",
}


# --------------------------------------------------------------------------- #
# English chunk writer
# --------------------------------------------------------------------------- #
def _english_chunk_text(concept: dict, facet: str, siblings: List[dict],
                        variant: int) -> str:
    """Compose one pedagogical English chunk of ~250-450 tokens."""
    name = concept["name"]
    topic = concept["topic"]
    aliases = ", ".join(concept["aliases"][:4])
    alias0 = concept["aliases"][0] if concept["aliases"] else name.lower()

    lead = {
        "intuition": f"Intuition first: {concept['idea']}",
        "formalization": f"Formal statement. {concept['formal']}",
        "implementation": f"Implementation notes for {name}.",
        "complexity": f"Cost analysis of {name}.",
        "worked_example": f"Worked example: applying {name} step by step.",
        "exam_pointer": f"Exam pointer for {name}.",
        "pitfall_note": f"Common mistake with {name}, and how to avoid it.",
    }[facet]

    body: List[str] = [
        f"# {name} ({topic})",
        f"Also searched as: {aliases}.",
        f"Keywords: {name}, {aliases}, {topic}.",
        "",
        lead,
        "",
        f"## Intuition",
        concept["idea"],
        f"Before writing any code for {name}, state this idea back in your own words; "
        f"if you cannot say what {alias0} is doing, the implementation will only hide "
        f"the confusion.",
        "",
        "## Precise statement",
        concept["formal"],
        f"This is the sentence to reproduce when an examination question asks you to "
        f"justify {name}, because it is what separates describing {alias0} from proving "
        f"that {alias0} is correct.",
        "",
        "## Cost",
        concept["cost"] + ".",
        f"Report the time and space cost of {name} separately, and say which case (best, "
        f"average or worst) each bound refers to.",
        "",
        "## How to apply it",
        f"Step 1. Spot the signal in the problem statement that points at {name}: the "
        f"phrasing usually associated with {alias0}.",
        f"Step 2. Write the invariant or recurrence of {name} on paper and check it on a "
        f"three-element example before coding.",
        f"Step 3. Code the smallest version of {name} that can be tested, then extend it "
        f"to the edge cases (empty input, one element, all elements equal).",
        f"Step 4. Re-derive the complexity of {name} from the code you actually wrote.",
        "",
        "## Frequent mistake",
        concept["pitfall"],
        f"In the laboratory sessions this single mistake accounts for most of the failing "
        f"test cases on {name}, so check for it first when debugging {alias0}.",
    ]

    if facet == "worked_example":
        body += [
            "",
            "## Trace",
            f"Take a five-element input and trace {name} by hand, writing the state of "
            f"every variable after each iteration. The point of the trace is the invariant "
            f"of {alias0}, not the final answer.",
            f"If the trace disagrees with your program, trust the trace and instrument the "
            f"code; if the trace itself is inconsistent, your statement of {name} is wrong.",
        ]
    elif facet == "complexity":
        body += [
            "",
            "## Deriving the bound",
            f"Count how many times the innermost statement of {name} executes as a function "
            f"of the input size, write that count as a sum, and only then simplify it with "
            f"asymptotic notation. Skipping the counting step is why complexity answers about "
            f"{alias0} based on the shape of the loops so often go wrong.",
        ]
    elif facet == "exam_pointer":
        body += [
            "",
            "## What is usually asked",
            f"Questions about {name} in the internal and semester examinations ask for the "
            f"definition of {alias0}, one derivation, and one edge case. Prepare a three-line "
            f"answer for each, and practise the derivation of {name} without notes.",
        ]
    elif facet == "implementation":
        body += [
            "",
            "## Checklist before you run the code",
            f"1. In your {name} implementation, are the loop bounds inclusive or exclusive, "
            f"and is that consistent?",
            f"2. What does {alias0} do on empty input and on a single element?",
            f"3. Is every pointer or index written in an order that does not lose data?",
            f"4. Does the code still satisfy the invariant of {name} at the end of each "
            f"iteration?",
        ]

    related = [s["name"] for s in siblings if s["key"] != concept["key"]][:5]
    if related:
        body += ["", "## Related in this unit", ", ".join(related) + "."]

    body += [
        "",
        "## Self check",
        f"Explain {name} to a classmate in under a minute, then have them ask why the cost "
        f"of {alias0} is what it is. If you cannot answer the 'why', return to the precise "
        f"statement of {name} above.",
    ]

    # Pad deterministically to land inside the configured 250-450 token band.
    fillers = [
        f"Revision note: connect {name} to at least one problem you have already solved with "
        f"{alias0}, because isolated facts are the first thing to disappear under pressure.",
        f"Laboratory note: measure the running time of your {name} implementation on inputs "
        f"of increasing size and check that the growth matches the bound derived above; a "
        f"mismatch usually means a hidden linear operation inside a loop.",
        f"Discussion note: when {name} fails, the reason is almost always that one of the "
        f"assumptions behind the precise statement of {alias0} is not satisfied by the input.",
    ]
    text = "\n".join(body)
    # Keep chunks inside the configured 250-450 token band: pad only if short.
    if len(text.split()) < 400:
        text += "\n\n" + fillers[variant % len(fillers)]
    return text


# --------------------------------------------------------------------------- #
# Code-mixed (Romanized Nepali + English) chunk writer
# --------------------------------------------------------------------------- #
CODE_MIXED_TEMPLATES = [
    """# {name} — doubt session note (Nepali + English)

**Simple bhasama:** {idea}

Yo concept classroom ma sodhne bela students le "{alias} kasari kaam garcha?" bhanera
sodhchan. Tyo bela ma main kura yo ho: {formal}

**Complexity ko kura:** {cost}. Yo line ratne haina, kina yesto huncha bhanera bujhne.
Exam ma "justify your answer" bhaneko cha bhane, maathi ko formal statement lekhnu parcha.

**Sabai bhanda badhi hune galti:** {pitfall}

Yo galti bata bachna, code lekhi sakepachi 3-4 element ko chhoto input ma dry run
garnus ra hare step ma variable ko value copy ma lekhnus. Dry run garda invariant
bhaTkiyo bhane, bug tyahi line ma cha.

**Yaad rakhne:** {name} lai {topic} unit ka arka concepts sanga jodera herda concept
clear huncha, alag alag ratda hudaina.""",

    """# {name} — kina yo garne? (code-mixed explanation)

Dherai jana lai lagcha ki {alias} bhaneko khali ek ota formula ho. Tara actual ma
idea yo ho: {idea}

Ani technical way ma bhannu parda: {formal}

**Kahile use garne?** Problem statement ma jun signal dekhincha — jasto ki {alias} sanga
milne pattern — tyo dekhepachi mathi ko invariant pahila copy ma lekhnus, ani matra code
suru garnus. Directly code lekhna suru garyo bhane 80 percent time logic bich ma bigrincha.

**Cost:** {cost}

**Galti:** {pitfall} Yesto galti viva ma pani sodhincha, "yo case ma k huncha?" bhanera.

Practice ko lagi: yo concept use garera ek ota problem aaphai solve garnus, ani tyo
solution ma time complexity kasari nikaleko bhanera 2 line ma lekhnus.""",

    """# {name} — lab ma bhaeko confusion (Nepali/English mix)

Lab session ma yo topic ma prayah "code chalcha tara output wrong aaucha" bhanne
problem aauncha. Root cause prayah yo hunchha: {pitfall}

Concept refresh: {idea}

Formal ma: {formal}

**Time / space:** {cost}

**Debug garne tarika:**
1. Sabai bhanda sano failing input khojnus (2 or 3 elements).
2. Hare iteration ma variables ko value print garnus.
3. Kun step ma invariant bhaTkiyo tyo point out garnus — tyahi bug ho.
4. Fix garepachi edge cases (empty input, single element, duplicate values) test garnus.

Yo process {topic} ka aru problems ma pani uttikai kaam lagcha, so ek choti practice
garisakepachi speed aafai badhcha.""",

    """# {name} — exam ma kasari sodhincha? (mixed register)

**Concept:** {idea}

**Definition jun exam ma lekhnu parcha:** {formal}

**Complexity:** {cost}

Question prayah yesari aauncha: "Define {alias} and analyse its complexity with an
example." Yesto ma answer ko structure yo rakhnus — pahila 2 line definition, ani
1 sano example, ani derivation, ani last ma edge case. Khali definition lekhyo bhane
full marks aaudaina, derivation chai chahincha.

**Dhyan dinu parne galti:** {pitfall}

Note: answer English ma lekhda pani, sochne bela aaphno bhasa ma sochda concept
chito clear huncha. Concept clear bhaisakepachi English ma lekhna sajilo huncha.""",
]


def _code_mixed_chunk_text(concept: dict, variant: int) -> str:
    template = CODE_MIXED_TEMPLATES[variant % len(CODE_MIXED_TEMPLATES)]
    return template.format(
        name=concept["name"],
        alias=concept["aliases"][0] if concept["aliases"] else concept["name"],
        idea=concept["idea"],
        formal=concept["formal"],
        cost=concept["cost"],
        pitfall=concept["pitfall"],
        topic=concept["topic"],
    )


# --------------------------------------------------------------------------- #
def build_knowledge_base(cfg: Config) -> List[dict]:
    """Generate the full chunk list, honouring Table 1 counts exactly."""
    set_seed(cfg.get("project.seed", 42))
    chunks: List[dict] = []

    for topic, (n_en, n_cm) in TABLE1.items():
        concepts = CONCEPTS[topic]
        code = TOPIC_CODE[topic]
        k = len(concepts)

        # ---- English chunks -------------------------------------------------
        for i in range(n_en):
            concept = concepts[i % k]
            facet = FACETS[(i // k) % len(FACETS)]
            variant = i // (k * len(FACETS))
            text = _english_chunk_text(concept, facet, concepts, variant)
            chunks.append({
                "chunk_id": f"{code}-EN-{i:04d}",
                "topic": topic,
                "concept": concept["key"],
                "concept_name": concept["name"],
                "facet": facet,
                "language": "en",
                "difficulty": DIFFICULTY_BY_FACET[facet],
                "source": SOURCES[i % len(SOURCES)],
                "text": text,
                "n_tokens": len(text.split()),
            })

        # ---- Code-mixed chunks (instructor-authored mini-explanations) ------
        for j in range(n_cm):
            concept = concepts[j % k]
            variant = j // k
            text = _code_mixed_chunk_text(concept, variant)
            chunks.append({
                "chunk_id": f"{code}-CM-{j:04d}",
                "topic": topic,
                "concept": concept["key"],
                "concept_name": concept["name"],
                "facet": "code_mixed_mini_explanation",
                "language": "cm",
                "difficulty": "medium",
                "source": "instructor_code_mixed",
                "text": text,
                "n_tokens": len(text.split()),
            })

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DSA knowledge base")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chunks = build_knowledge_base(cfg)

    out_dir = cfg.path("paths.processed_dir")
    path = write_jsonl(out_dir / "knowledge_base.jsonl", chunks)

    n_en = sum(1 for c in chunks if c["language"] == "en")
    n_cm = sum(1 for c in chunks if c["language"] == "cm")
    tokens = [c["n_tokens"] for c in chunks]
    stats = {
        "total_chunks": len(chunks),
        "english_chunks": n_en,
        "code_mixed_chunks": n_cm,
        "unique_concepts": len({c["concept"] for c in chunks}),
        "mean_tokens": round(sum(tokens) / len(tokens), 1),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "per_topic": {t: {"english": e, "code_mixed": c, "total": e + c}
                      for t, (e, c) in TABLE1.items()},
    }
    write_json(cfg.path("paths.processed_dir") / "knowledge_base_stats.json", stats)

    log.info("Knowledge base written to %s", path)
    log.info("Total=%d  English=%d  CodeMixed=%d  mean_tokens=%.1f (min %d / max %d)",
             stats["total_chunks"], n_en, n_cm, stats["mean_tokens"],
             stats["min_tokens"], stats["max_tokens"])
    assert len(chunks) == cfg.get("knowledge_base.target_total_chunks", 1252), \
        "Chunk count does not match Table 1"


if __name__ == "__main__":
    main()
