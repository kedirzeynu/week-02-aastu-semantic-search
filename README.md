# AASTU Mini Semantic Search

A semantic search engine over a dataset of university FAQs (Frequently Asked Questions).
Built for AWS SBG @ AASTU AI/ML learning track, Week 2 task: Data, APIs & Embeddings.


## Dataset

`week-02-aastu-semantic-search/data/faq_raw.csv` holds 24 sample university FAQs (raw, intentionally messy).

|   Column   | Meaning |
|------------|---------|
| id         | FAQ number |
| question   | The question text |
| answer     | The answer text |
| category   | Topic (Registration, Fees, ...) |
| views      | How many times it was viewed |
| date_added | Date the FAQ was added |


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