# AASTU Mini Semantic Search

A semantic search engine over a dataset of university FAQs (Frequently Asked Questions).
Built for AWS SBG @ AASTU AI/ML learning track, Week 2 task: Data, APIs & Embeddings.


## Dataset

`week-02-aastu-semantic-search/data/faq_raw.csv` holds 24 sample university FAQs (raw, intentionally messy).

| Column | Meaning |
|---|---|
| id | FAQ number |
| question | The question text |
| answer | The answer text |
| category | Topic (Registration, Fees, ...) |
| views | How many times it was viewed |
| date_added | Date the FAQ was added |


## Data cleaning

| Problem | Function | Method |
|---|---|---|
| Messy text/case | `fix_text` | strip, collapse spaces, consistent categories |
| Wrong types | `fix_types` | `to_numeric`, `to_datetime` (errors="coerce") |
| Duplicates | `remove_duplicates` | `drop_duplicates` on question + answer |
| Outliers | `handle_outliers` | IQR rule (NumPy percentiles) |
| Missing values | `handle_missing` | drop rows with no answer, median fill for views |


## Setup

1. Clone the repo and enter the folder:
   ```bash
   git clone <https://github.com/kedirzeynu/week-02-aastu-semantic-search.git>
   cd week-02-aastu-semantic-search
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate it:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## How to run

1. Run the loader to load the data:
   ```bash
   python week-02-aastu-semantic-search/src/load_data.py
   ```

NB: The `.venv/` folder is excluded from Git via `.gitignore`.




# AASTU Mini Semantic Search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/aastu-semantic-search/blob/main/week-02/notebooks/semantic_search.ipynb)

Semantic search over a messy FAQ dataset: load with Pandas, clean with functions,
process with NumPy, embed, store in Chroma, and return the top 3 matches.
Built for AI/ML AWS AASTU Cohort 1, Week 2 (Data, APIs & Embeddings).

## Project structure
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
└── week-02/
    ├── data/       faq_raw.csv, faq_clean.csv, courses.json
    ├── notebooks/  semantic_search.ipynb
    └── src/        load_data, cleaning, numpy_stats, text_prep,
                    api_demo, embeddings, similarity, search
```

## Setup (virtual environment)
1. `git clone https://github.com/<your-username>/aastu-semantic-search.git`
2. `cd aastu-semantic-search`
3. Create a venv: `python -m venv venv`
4. Activate it: Windows `venv\Scripts\activate`, Mac/Linux `source venv/bin/activate`
5. Install: `pip install -r requirements.txt`

`venv/` is excluded from Git by `.gitignore`.

## How to run
```bash
python week-02/src/load_data.py     # first look at the raw data
python week-02/src/cleaning.py      # clean it, write faq_clean.csv
python week-02/src/text_prep.py     # tokens and chunking demo
python week-02/src/similarity.py    # cosine + brute-force timing
python week-02/src/search.py        # semantic search, top 3 matches
python week-02/src/search.py "your own question"
```
Notebook: `pip install notebook`, then `jupyter notebook week-02/notebooks/semantic_search.ipynb`,
or click the Colab badge above. The first run downloads a ~80 MB embedding model (internet needed once).

## Example queries
(paste your REAL output here)
```
Query: "I forgot my password"
  1. [..] (Accounts) ...
```

## Dataset
(keep your Day 2 table and the "Data issues found" list)

## Data cleaning
(keep your Day 3 table and decisions)

## APIs and secrets
(keep your Day 4 section)

## What I learned
(3-4 sentences)