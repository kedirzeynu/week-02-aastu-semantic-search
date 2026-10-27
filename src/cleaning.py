# cleaning.py - the cleaning toolkit.
import numpy as np
import pandas as pd

from load_data import load_faqs
from numpy_stats import iqr_bounds, normalize

ACRONYMS = {"it": "IT"}   # labels that must stay uppercase

def quality_report(df, title="REPORT"):
    # Print rows, duplicate rows and blanks per column.
    print(f"--- {title} ---")
    print("rows:", len(df), "| duplicate rows:", int(df.duplicated().sum()))
    print("blanks:", df.isna().sum().to_dict())


def _fix_category(value):
    # Tidy one category label: trim spaces, fix case, keep acronyms like IT.
    if pd.isna(value):
        return value
    key = value.strip().lower()
    return ACRONYMS.get(key, key.capitalize())


def fix_text(df):
    # TEXT/CASE FIXES: trim, collapse spaces, consistent category labels.
    df = df.copy()
    for col in ["question", "answer"]:
        df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)
    df["category"] = df["category"].map(_fix_category)
    return df


def fix_types(df):
    # TYPE CONVERSION: '1,200' text -> 1200.0 number; date text -> real dates.
    df = df.copy()
    cleaned = df["views"].astype("string").str.replace(",", "", regex=False)
    df["views"] = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    df["date_added"] = pd.to_datetime(df["date_added"], format="mixed", errors="coerce")
    return df


def remove_duplicates(df):
    # DUPLICATES: keep each question+answer pair once.
    return df.drop_duplicates(subset=["question", "answer"], keep="first").reset_index(drop=True)


def handle_outliers(df, column="views"):
    # OUTLIERS: values outside the IQR range become blank (filled later).
    df = df.copy()
    valid = df[column].dropna().to_numpy()
    low, high = iqr_bounds(valid)                      # NumPy from Day 1
    is_outlier = (df[column] < low) | (df[column] > high)
    print(f"{column}: normal range {low:.1f} to {high:.1f}, outliers found: {int(is_outlier.sum())}")
    df.loc[is_outlier, column] = np.nan
    return df


def handle_missing(df):
    # MISSING VALUES: drop rows with no answer, fill views with the median.
    df = df.dropna(subset=["answer"]).copy()
    median_views = np.median(df["views"].dropna().to_numpy())
    df["views"] = df["views"].fillna(median_views)
    df["category"] = df["category"].fillna("Unknown")
    return df.reset_index(drop=True)


def add_features(df):
    # Scale views to 0-1 and build the combined text column for Day 5.
    df = df.copy()
    df["views_scaled"] = normalize(df["views"].to_numpy())
    df["text"] = df["question"] + " " + df["answer"]
    return df


def clean_faqs(df):
    # Run the whole pipeline in the right order.
    df = fix_text(df)
    df = fix_types(df)
    df = remove_duplicates(df)
    df = handle_outliers(df, "views")
    df = handle_missing(df)
    return add_features(df)


if __name__ == "__main__":
    raw = load_faqs()
    quality_report(raw, "BEFORE")
    clean = clean_faqs(raw)
    quality_report(clean, "AFTER")
    print(clean[["id", "category", "views", "date_added", "views_scaled"]])
    print(clean["category"].value_counts())
    print(clean.dtypes)