"""embeddings.py - turn text into vectors (slide 14)."""
import re
import zlib

import numpy as np

DIM = 384  # the size of the MiniLM embedding vectors on slide 14


class ToyEmbedder:
    """Offline stand-in: hashes WORDS into 384 slots. It only matches shared
    words, so it cannot understand meaning. Used for comparison and testing."""

    name = "toy-hashing"

    def __call__(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class MiniLMEmbedder:
    """The real model: Chroma's default all-MiniLM-L6-v2 (downloads ~80 MB once)."""

    name = "all-MiniLM-L6-v2"

    def __init__(self):
        from chromadb.utils import embedding_functions
        self._fn = embedding_functions.DefaultEmbeddingFunction()
        self._fn(["warm up"])  # forces the download now, so we fail early

    def __call__(self, texts):
        return np.array(self._fn(list(texts)), dtype=np.float32)


def get_embedder(allow_fallback=True):
    """Return the real embedder; if the model can't download, warn and fall back."""
    try:
        return MiniLMEmbedder()
    except Exception as error:
        if not allow_fallback:
            raise
        print("WARNING: could not load the MiniLM model (" + type(error).__name__ + ").")
        print("         Using the ToyEmbedder - results will NOT understand meaning.")
        return ToyEmbedder()


def embed_texts(embedder, texts):
    """Embed a list of texts and return a NumPy array of shape (n, 384)."""
    return np.asarray(embedder(list(texts)), dtype=np.float32)