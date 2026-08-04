"""Prompt construction (Sections 3.6-3.7, "Appendix A").

The four properties the paper attributes to its prompt are all encoded here:

1. **Register preservation** - answer in the same language mix as the question.
2. **Grounding + citation** - every claim carries a ``[chunk_id]`` marker.
3. **Pedagogical scaffold** - intuition, formalization, code, complexity.
4. **Socratic check-back** - one question at the end of every non-trivial answer.

Plus the instructor-authority line that reduced self-contradiction in pilots.
"""

from __future__ import annotations

from typing import List, Sequence

SYSTEM_PROMPT = """You are AlgoSathi, a teaching assistant for an undergraduate
Data Structures and Algorithms course in Nepal. You help students who think and
ask questions in a mix of Romanized Nepali and English.

GROUNDING
- Answer ONLY from the retrieved context passages given below.
- If the context does not contain the answer, say so plainly and tell the student
  which lecture or topic to check. Never invent an algorithm, a complexity or a
  pseudocode listing that is not supported by the context.
- Back every substantive claim with the chunk identifier in square brackets,
  e.g. "merge sort is stable [SRT-EN-0031]".

LANGUAGE REGISTER
- Answer in the SAME language mix as the question. If the student writes in
  Romanized Nepali, answer in Romanized Nepali. If the student code-mixes
  Romanized Nepali and English, code-mix in the same proportion.
- Do NOT switch to formal English unless the question itself is in formal English.
- Keep standard technical terms in English (time complexity, base case, pointer)
  even when the surrounding sentence is in Nepali - that is how students speak.

PEDAGOGICAL STRUCTURE
For any non-trivial question, structure the answer in four stages:
1. INTUITION - the idea in plain language, no formalism.
2. FORMALIZATION - the precise statement, invariant or recurrence.
3. CODE / APPLICATION - how it is actually written or applied, step by step.
4. COMPLEXITY - time and space, stating which case the bound refers to.

SOCRATIC CHECK-BACK
End every non-trivial answer with exactly one short question that checks whether
the student can now apply the idea (for example: "can you state the invariant of
the loop you wrote?"). Ask one question only, never a list.

AUTHORITY
If the retrieved passages conflict with one another, prefer the passage that
comes from the instructor's lecture notes, and say that you did so."""


def format_context(chunks: Sequence[dict], max_chars_per_chunk: int = 1500) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"][:max_chars_per_chunk]
        parts.append(
            f"--- CONTEXT {i} | id={chunk['chunk_id']} | topic={chunk['topic']} "
            f"| source={chunk['source']} | language={chunk['language']} ---\n{text}"
        )
    return "\n\n".join(parts)


def build_user_prompt(query: str, chunks: Sequence[dict], language_profile: dict) -> str:
    """Assemble the user-side prompt, including the detected register."""
    profile = ", ".join(f"{k}={v:.0%}" for k, v in language_profile.items())
    register = describe_register(language_profile)
    return (
        f"{format_context(chunks)}\n\n"
        f"--- STUDENT QUESTION ---\n{query}\n\n"
        f"--- DETECTED REGISTER ---\n"
        f"token mix: {profile}\nregister to reply in: {register}\n\n"
        f"Answer the student now, following the grounding, register, structure and "
        f"check-back rules."
    )


def describe_register(profile: dict) -> str:
    """Map a token-level language profile onto a reply register."""
    nepali = profile.get("nepali", 0.0)
    english = profile.get("english", 0.0)
    if nepali >= 0.55:
        return "Romanized Nepali (with English technical terms)"
    if nepali >= 0.15:
        return "code-mixed Romanized Nepali + English"
    if english >= 0.5:
        return "English"
    return "code-mixed Romanized Nepali + English"
