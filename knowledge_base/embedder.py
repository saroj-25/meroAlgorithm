# knowledge_base/embedder.py
# Wraps the multilingual sentence encoder from the paper

import numpy as np
from sentence_transformers import SentenceTransformer


class DSAEmbedder:
    """
    Uses paraphrase-multilingual-mpnet-base-v2
    — supports Romanized Nepali + English code-mixed queries.
    """

    MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

    def __init__(self):
        print(f"[Embedder] Loading model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        print("[Embedder] Model loaded.")

    def embed(self, texts: list) -> np.ndarray:
        """
        Embed a list of strings.
        Returns float32 numpy array of shape (len(texts), 768).
        normalize_embeddings=True → cosine similarity = dot product.
        """
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
