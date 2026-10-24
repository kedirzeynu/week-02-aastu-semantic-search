# numpy_stats.py - NumPy helpers for the FAQ semantic search project.
import numpy as np


def add_vat(prices, rate=0.15):
    # Return prices with VAT added (vectorized, no loop).
    prices = np.asarray(prices, dtype=float)
    return prices * (1 + rate)


def summarize(values):
    # Return a dict of key statistics for a list/array of numbers.
    values = np.asarray(values, dtype=float)
    return {
        "count": values.size,
        "mean": values.mean(),
        "median": np.median(values),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "p25": np.percentile(values, 25),
        "p75": np.percentile(values, 75),
    }


def iqr_bounds(values, k=1.5):
    # Normal range using the IQR rule: anything outside is suspicious.
    values = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def flag_outliers(values):
    # Return 1 where a value is an outlier, else 0.
    values = np.asarray(values, dtype=float)
    low, high = iqr_bounds(values)
    return np.where((values < low) | (values > high), 1, 0)


def normalize(values):
    # Rescale to 0-1 (min-max), so different scales compare fairly.
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


if __name__ == "__main__":
    print("VAT:", add_vat([85, 120, 200]))

    views = [120, 85, 240, 60, 15000, 310, 95]
    for name, value in summarize(views).items():
        print(f"{name}: {value:.2f}")

    low, high = iqr_bounds(views)
    print("Normal range:", low, "to", high)
    print("Outlier flags:", flag_outliers(views))
    print("Normalized:", normalize(views).round(3))