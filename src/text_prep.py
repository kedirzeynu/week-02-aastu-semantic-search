# text_prep.py - cleaning, tokenizing and chunking text.
import re

import numpy as np

from cleaning import clean_faqs
from load_data import load_faqs


def clean_text(text):
    # Lowercase, trim, and keep only letters, digits and spaces.
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t)


def tokenize(text):
    # Split clean text into word tokens.
    return clean_text(text).split()


def chunk_words(words, size=200, overlap=40):
    # Cut a list of words into overlapping chunks (no tiny leftover chunk).
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(words[i:i + size])
        if i + size >= len(words):
            break
    return chunks


def prepare_text_columns(df):
    # Add text_clean, tokens and n_tokens columns to the FAQ table.
    df = df.copy()
    df["text_clean"] = df["text"].map(clean_text)
    df["tokens"] = df["text"].map(tokenize)
    df["n_tokens"] = df["tokens"].map(len)
    return df


if __name__ == "__main__":
    raw = "  Visit AASTU!! Classes START Monday  "
    print(repr(clean_text(raw)))
    print(tokenize(raw))

    df = prepare_text_columns(clean_faqs(load_faqs()))
    n = df["n_tokens"].to_numpy()
    print("tokens per FAQ: mean", n.mean().round(1), "median", np.median(n),
          "min", n.min(), "max", n.max(), "p90", np.percentile(n, 90))

    handbook = " ".join(df["answer"])
    words = handbook.split()
    chunks = chunk_words(words, size=60, overlap=12)
    print("handbook words:", len(words), "| chunks:", len(chunks))
    print("chunk sizes:", [len(c) for c in chunks])
    print("overlap check:", chunks[0][-12:] == chunks[1][:12])