"""Download and load the Naver Sentiment Movie Corpus (NSMC).

200,000 Korean movie reviews from Naver, labelled 0 (negative) or 1 (positive).
https://github.com/e9t/nsmc  (CC0 licence)
"""

import os
import urllib.request

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/"
FILES = ["ratings_train.txt", "ratings_test.txt"]


def download(data_dir="data", base_url=BASE_URL):
    os.makedirs(data_dir, exist_ok=True)
    for name in FILES:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"downloading {name}")
            urllib.request.urlretrieve(base_url + name, path)


def _read(path):
    # quoting=3 turns quote handling off. Some reviews contain a lone "
    # and pandas would otherwise glue several lines into one row.
    df = pd.read_csv(path, sep="\t", quoting=3)
    df = df.dropna(subset=["document"])  # a handful of reviews are empty
    df["document"] = df["document"].astype(str)
    return df.reset_index(drop=True)


def load(data_dir="data"):
    """Return (train, test) DataFrames with columns id, document, label."""
    download(data_dir)
    train = _read(os.path.join(data_dir, "ratings_train.txt"))
    test = _read(os.path.join(data_dir, "ratings_test.txt"))
    return train, test


if __name__ == "__main__":
    train, test = load()
    for name, df in [("train", train), ("test", test)]:
        print(f"{name}: {len(df):,} reviews, {df.label.mean():.1%} positive, "
              f"median length {int(df.document.str.len().median())} characters")
    seen = test.document.isin(set(train.document)).sum()
    print(f"test reviews that also appear word for word in train: {seen:,}")
