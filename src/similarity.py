"""similarity.py - cosine similarity and nearest neighbors (slides 15-17)."""
import time

import numpy as np


def cosine(a, b):
    """Slide 15: compare DIRECTIONS of two vectors (1 same, 0 unrelated, -1 opposite)."""
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_scores(docs, q):
    """Cosine of one query against EVERY row of docs at once."""
    return docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))


def top_k(docs, q, k=3):
    """Slide 16: rank every item by score, keep the best k."""
    scores = cosine_scores(docs, q)
    best = np.argsort(scores)[::-1][:k]
    return best, scores[best]


def brute_force_demo(n=100_000, dim=384, seed=0):
    """Slide 17: brute force compares the query with ALL vectors. Time it."""
    rng = np.random.default_rng(seed)
    docs = rng.standard_normal((n, dim), dtype=np.float32)
    q = rng.standard_normal(dim).astype(np.float32)
    start = time.perf_counter()
    top_k(docs, q, k=3)
    return time.perf_counter() - start


if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0])
    print("same direction:", cosine(a, 2 * a))
    print("unrelated:", cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])))
    print("opposite:", cosine(a, -a))
    for n in (1_000, 10_000, 100_000):
        print(f"brute force over {n:>7,} vectors: {brute_force_demo(n) * 1000:.1f} ms")