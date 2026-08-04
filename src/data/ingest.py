"""

    python -m src.data.ingest --input data/raw --topic "Sorting Algorithms" --language en

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

from ..preprocessing.chunking import chunk_text
from ..utils.config import load_config
from ..utils.io import read_jsonl, write_jsonl
from ..utils.logging_utils import get_logger

log = get_logger("data.ingest")


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pip install pypdf to ingest PDFs") from exc
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        try:
            import docx
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pip install python-docx to ingest .docx") from exc
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def slugify(text: str, length: int = 24) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:length] or "chunk"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest course material")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, help="file or directory")
    parser.add_argument("--topic", default="Custom Material")
    parser.add_argument("--language", default="en", choices=["en", "cm"],
                        help="en = English, cm = code-mixed Romanized Nepali/English")
    parser.add_argument("--replace", action="store_true",
                        help="replace the knowledge base instead of appending")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = cfg.path("paths.processed_dir")
    kb_path = processed / "knowledge_base.jsonl"

    source = Path(args.input)
    files: List[Path] = ([source] if source.is_file()
                         else sorted(p for p in source.rglob("*")
                                     if p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}))
    if not files:
        raise SystemExit(f"No ingestible files found under {source}")

    existing = [] if args.replace or not kb_path.exists() else read_jsonl(kb_path)
    prefix = "USR"
    added = []
    for f_idx, path in enumerate(files):
        text = read_document(path)
        pieces = chunk_text(text,
                            min_tokens=cfg.get("knowledge_base.chunk_min_tokens", 250),
                            max_tokens=cfg.get("knowledge_base.chunk_max_tokens", 450),
                            overlap=cfg.get("knowledge_base.chunk_overlap_tokens", 50))
        for c_idx, piece in enumerate(pieces):
            added.append({
                "chunk_id": f"{prefix}-{args.language.upper()}-{f_idx:02d}{c_idx:03d}",
                "topic": args.topic,
                "concept": slugify(path.stem),
                "concept_name": path.stem.replace("_", " ").title(),
                "facet": "user_upload",
                "language": args.language,
                "difficulty": "medium",
                "source": path.name,
                "text": piece,
                "n_tokens": len(piece.split()),
            })
        log.info("%s -> %d chunks", path.name, len(pieces))

    write_jsonl(kb_path, existing + added)
    log.info("Knowledge base now has %d chunks (%d added). "
             "Rebuild the index: python -m src.retrieval.index_builder",
             len(existing) + len(added), len(added))


if __name__ == "__main__":
    main()
