# similarity.py - cosine similarity and nearest neighbors.

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


if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0])
    print("same direction:", cosine(a, 2 * a))
    print("unrelated:", cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])))
    print("opposite:", cosine(a, -a))