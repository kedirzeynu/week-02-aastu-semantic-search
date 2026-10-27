# search.py - Mini Semantic Search: Pandas + NumPy + embeddings + Chroma.
import sys

import chromadb
import numpy as np

from cleaning import clean_faqs
from embeddings import embed_texts, get_embedder
from load_data import load_faqs
from similarity import top_k

COLLECTION = "aastu_faqs"


def load_clean_faqs():
    """Steps 1-2 of the lab: load the raw CSV with Pandas, then clean it."""
    return clean_faqs(load_faqs())


def build_collection(df, embedder):
    """Steps 3-4: embed each FAQ (one FAQ = one chunk) and store it in Chroma,
    with the structured fields (category, views) saved next to the vector."""
    client = chromadb.Client()
    col = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    col.upsert(
        ids=[f"faq_{i}" for i in df["id"]],
        documents=df["text"].tolist(),
        embeddings=embed_texts(embedder, df["text"]).tolist(),
        metadatas=[
            {"category": c, "views": float(v)}
            for c, v in zip(df["category"], df["views"])
        ],
    )
    return col


def search(col, embedder, query, k=3, where=None):
    """Step 5: embed the question and return the top-k matches."""
    q = embed_texts(embedder, [query]).tolist()
    result = col.query(query_embeddings=q, n_results=k, where=where)
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"score": 1 - dist, "category": meta["category"], "text": doc})
    return hits


def print_hits(query, hits):
    print(f'\nQuery: "{query}"')
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank}. [{hit['score']:.2f}] ({hit['category']}) {hit['text'][:90]}...")


def numpy_search(df, embedder, query, k=3):
    # The same search with pure NumPy (slides 15-16), to compare with Chroma.
    docs = embed_texts(embedder, df["text"])
    q = embed_texts(embedder, [query])[0]
    idx, scores = top_k(docs, q, k)
    return [(df.iloc[i]["id"], float(s)) for i, s in zip(idx, scores)]


if __name__ == "__main__":
    faqs = load_clean_faqs()
    embedder = get_embedder()
    collection = build_collection(faqs, embedder)
    queries = sys.argv[1:] or [
        "I forgot my password",
        "When is registration?",
        "Where can I study quietly?",
    ]
    for text in queries:
        print_hits(text, search(collection, embedder, text, k=3))