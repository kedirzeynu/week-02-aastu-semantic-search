# AASTU Mini Semantic Search

A small search engine that finds university FAQs by **meaning**, not by matching keywords.
Ask "I forgot my password" and it finds "How can I reset my login?" even though the two
share almost no words.

Built for **AI/ML AWS AASTU Cohort 1, Week 2: Data, APIs & Embeddings**.

**Pipeline:**
load a messy CSV (Pandas) → clean it with functions → process numbers with NumPy →
prepare and chunk the text → embed it → store it in ChromaDB → query and print the top 3 matches.

> The FAQ data is **sample data for learning**, not official university policy.

---

## Project structure

```
week-02-aastu-semantic-search/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── faq_raw.csv              raw, deliberately messy dataset
├── notebooks/
│   └── semantic-search.ipynb    the full pipeline with saved outputs
└── src/
    ├── load_data.py             load the CSV and take a first look
    ├── cleaning.py              the cleaning functions
    ├── numpy_stats.py           statistics with NumPy
    ├── text_prep.py             clean text, tokenize, chunk
    ├── embeddings.py            text -> 384-number vectors
    ├── similarity.py            cosine similarity and top-k neighbors
    └── search.py                ChromaDB storage and semantic search
```

---

## Setup (virtual environment)

1. Clone the repository and enter the folder:
   ```bash
   git clone https://github.com/kedirzeynu/week-02-aastu-semantic-search.git
   cd week-02-aastu-semantic-search
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate it:
   - Windows: `.venv\Scripts\activate`
   - Mac / Linux: `source .venv/bin/activate`
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

The `.venv/` folder (and `__pycache__/`) is excluded from Git by `.gitignore`.

> **First run needs internet.** The embedding model (`all-MiniLM-L6-v2`, about 80 MB)
> downloads once and is then cached on your machine.

---

## How to run

Run these from the project root, with the virtual environment activated.

```bash
python src/load_data.py        # first look at the raw data
python src/cleaning.py         # before/after cleaning report
python src/text_prep.py        # tokens and chunking demo
python src/similarity.py       # cosine similarity demo
python src/search.py           # semantic search with three sample questions
python src/search.py "how do I get back into my account"   # your own question
```

### The notebook

- **Locally:** `pip install notebook`, then open `notebooks/semantic-search.ipynb` in
  Jupyter or VS Code and choose **Run All**. Select the `.venv` interpreter as the kernel.
- **Colab:** click the **Open In Colab** badge at the top. The first cell clones this
  repository and installs `requirements.txt` automatically (the repository must be public).

---

## Example queries

*(Replace the lines below with the real output from `python src/search.py`.)*

```
Query: "I forgot my password"
  1. [score] (category) ...
  2. [score] (category) ...
  3. [score] (category) ...

Query: "When is registration?"
  1. [score] (category) ...

Query: "Where can I study quietly?"
  1. [score] (category) ...
```

---

## Dataset

`data/faq_raw.csv` holds 24 sample university FAQs. It is **raw and intentionally messy**
and is never edited by hand; all fixes happen in code.

| Column | Meaning |
|---|---|
| `id` | FAQ number |
| `question` | The question text |
| `answer` | The answer text |
| `category` | Topic (Registration, Fees, Accounts, ...) |
| `views` | How many times the FAQ was viewed |
| `date_added` | Date the FAQ was added |

Problems built into the data: blank values, a duplicate row, inconsistent category text
(`ACCOUNTS`, `registration `), `views` stored as text with commas (`"1,200"`), two extreme
outliers (15000 and 99999), and dates in mixed formats (one is `not sure`).

---

## Data cleaning

All cleaning lives in `src/cleaning.py` as named functions. `clean_faqs()` runs them in
the order shown, and the table is also the order in which they must run.

| Problem | Function | Method |
|---|---|---|
| Messy text / case | `fix_text` | strip spaces, collapse repeated spaces, tidy category labels (keeps `IT` uppercase) |
| Wrong types | `fix_types` | remove commas then `pd.to_numeric(errors="coerce")`; `pd.to_datetime(format="mixed", errors="coerce")` |
| Duplicates | `remove_duplicates` | `drop_duplicates` on question + answer |
| Outliers | `handle_outliers` | IQR rule using NumPy percentiles; outliers become blank |
| Missing values | `handle_missing` | drop rows with no answer, fill `views` with the median, label missing categories `Unknown` |

Result: **24 rows → 21 rows**. Decisions worth knowing:
- Text is fixed **before** duplicates are removed, because two rows that differ only by
  extra spaces only match after trimming.
- Outliers are blanked **before** the median fill, so one rule handles both problems.
- One unparseable date (`not sure`) is left blank, because dates are not used for search.

---

## NumPy and text preparation

- `src/numpy_stats.py` uses NumPy for `mean`, `median`, `std`, `min`/`max`, `percentile`
  and `where` (summary statistics, the IQR outlier rule and 0–1 scaling).
- `src/similarity.py` uses NumPy for the dot product, `linalg.norm` (cosine similarity) and
  `argsort` (the top-k nearest neighbors).
- `src/text_prep.py` lowercases and tokenizes text, and provides `chunk_words` for long
  documents. Each FAQ is only about 16–23 tokens, so it is already chunk-sized and the search
  embeds **one FAQ per chunk**.

---

## How the search works

1. **Embed:** each FAQ (question + answer) becomes a vector of 384 numbers.
2. **Store:** vectors go into ChromaDB with `category` and `views` saved beside them.
3. **Ask:** the question is embedded the same way.
4. **Top-k:** Chroma returns the 3 vectors with the highest **cosine similarity**
   (`score = 1 - distance`, so higher is better).
5. **Filter (optional):** `search(..., where={"category": "Fees"})` combines meaning with conditions.

---

## What I learned

*(Write 3–4 sentences in your own words.)*
