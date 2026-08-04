"""Interactive command-line chat with AlgoSathi.

    python -m src.inference.cli --config configs/default.yaml
    python -m src.inference.cli --question "yo merge sort ko complexity kasari nikalne"

Commands inside the session: ``:sources``, ``:lang <en|np|mix|auto>``, ``:quit``.
"""

from __future__ import annotations

import argparse

from ..utils.config import load_config
from ..utils.io import write_jsonl
from .rag_pipeline import AlgoSathiRAG

BANNER = """
AlgoSathi - DSA teaching assistant (Romanized Nepali + English)
Type your question in English, Nepali or a mix. ':quit' to exit, ':help' for commands.
"""

LANG_MAP = {"en": "english", "np": "romanized_nepali", "mix": "code_mixed", "auto": None}


def show(response) -> None:
    profile = ", ".join(f"{k}={v:.0%}" for k, v in response.language_profile.items())
    print(f"\n[register: {response.query_type} | {profile} | "
          f"{response.latency_s:.2f}s | {response.generator}]\n")
    print(response.answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with AlgoSathi")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question", default=None, help="ask one question and exit")
    parser.add_argument("--log", default=None, help="append turns to this JSONL file")
    args = parser.parse_args()

    rag = AlgoSathiRAG(load_config(args.config))
    turns = []

    if args.question:
        response = rag.ask(args.question)
        show(response)
        if args.log:
            write_jsonl(args.log, [response.as_log_record()])
        return

    print(BANNER)
    force = None
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in (":quit", ":q", "exit"):
            break
        if question == ":help":
            print("  :sources        show the chunks behind the last answer\n"
                  "  :lang en|np|mix|auto   override language detection\n"
                  "  :quit")
            continue
        if question == ":sources":
            if turns:
                for chunk in turns[-1].chunks:
                    print(f"  [{chunk['chunk_id']}] {chunk['topic']} / "
                          f"{chunk['concept_name']} (score {chunk.get('rerank_score', 0):.3f})")
            continue
        if question.startswith(":lang"):
            key = question.split()[-1]
            force = LANG_MAP.get(key, None)
            print(f"  language override -> {force or 'auto'}")
            continue

        response = rag.ask(question, force_language=force)
        turns.append(response)
        show(response)
        if args.log:
            write_jsonl(args.log, [t.as_log_record() for t in turns])


if __name__ == "__main__":
    main()
