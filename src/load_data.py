# load_data.py - load raw datasets and take a first look.

from pathlib import Path
import pandas as pd

# folder: week-02-aastu-semantic-search/data  (built from this file's location, so it works from anywhere)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" # Directory in which the our data set is found.


def load_faqs(filename="faq_raw.csv"):
    # Load the raw FAQ CSV exactly as it is (no cleaning here).
    return pd.read_csv(DATA_DIR / filename)

def first_look(df):
    # Run the 5 first-look commands from slide 8.
    print("SHAPE:", df.shape)
    print("\nHEAD:\n", df.head())
    print("\nINFO:")
    df.info() # No print function is needed for info.
    print("\nDESCRIBE:\n", df.describe())
    print("\nBLANKS PER COLUMN:\n", df.isna().sum())


if __name__ == "__main__":
    print("===== FAQS =====")
    first_look(load_faqs())