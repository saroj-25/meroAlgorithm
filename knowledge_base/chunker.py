# knowledge_base/chunker.py
# Splits raw DSA text/PDF files into small chunks for retrieval

import os
import json

def chunk_text_files(raw_dir: str, output_path: str) -> list:
    """
    Reads all .txt files from raw_dir,
    splits them into overlapping chunks,
    saves to output_path as JSON.
    """
    chunks = []
    chunk_size = 300      # characters per chunk
    overlap    = 50       # overlap between chunks

    for fname in os.listdir(raw_dir):
        fpath = os.path.join(raw_dir, fname)
        if not os.path.isfile(fpath):
            continue

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        buffer = ""
        chunk_idx = 0

        for para in paragraphs:
            buffer += " " + para
            while len(buffer) >= chunk_size:
                chunk_text = buffer[:chunk_size].strip()
                if chunk_text:
                    chunks.append({
                        "id":     f"{fname}_chunk{chunk_idx}",
                        "source": fname,
                        "text":   chunk_text
                    })
                    chunk_idx += 1
                buffer = buffer[chunk_size - overlap:]

        # Remaining buffer
        if buffer.strip():
            chunks.append({
                "id":     f"{fname}_chunk{chunk_idx}",
                "source": fname,
                "text":   buffer.strip()
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[Chunker] Created {len(chunks)} chunks → saved to {output_path}")
    return chunks
