"""text_prep.py - cleaning, tokenizing and chunking text (slides 12 and 13)."""
import re

import numpy as np

from cleaning import clean_faqs
from load_data import load_faqs

TOY_VOCAB = {"un", "happi", "ness", "class", "es", "regist", "ration", "view", "s"}


def clean_text(text):
    """Lowercase, trim, and keep only letters, digits and spaces."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t)


def tokenize(text):
    """Split clean text into word tokens."""
    return clean_text(text).split()


def toy_subword_tokenize(word, vocab=TOY_VOCAB):
    """Greedy longest-match sub-word split (a toy version of what LLMs do)."""
    pieces, start = [], 0
    while start < len(word):
        for end in range(len(word), start, -1):
            if word[start:end] in vocab:
                pieces.append(word[start:end])
                start = end
                break
        else:
            pieces.append(word[start])
            start += 1
    return pieces


def chunk_words(words, size=200, overlap=40):
    """Cut a list of words into overlapping chunks (no tiny leftover chunk)."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(words[i:i + size])
        if i + size >= len(words):
            break
    return chunks


def chunk_text(text, size=200, overlap=40):
    """Chunk a long string and return a list of strings."""
    return [" ".join(c) for c in chunk_words(text.split(), size, overlap)]


def prepare_text_columns(df):
    """Add text_clean, tokens and n_tokens columns to the FAQ table."""
    df = df.copy()
    df["text_clean"] = df["text"].map(clean_text)
    df["tokens"] = df["text"].map(tokenize)
    df["n_tokens"] = df["tokens"].map(len)
    return df


if __name__ == "__main__":
    raw = "  Visit AASTU!! Classes START Monday  "
    print(repr(clean_text(raw)))
    print(tokenize(raw))
    print(toy_subword_tokenize("unhappiness"))

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