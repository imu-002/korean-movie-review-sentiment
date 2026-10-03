"""Compare feature choices on a validation split.

    python experiments.py

The last 15,000 training reviews are held out for validation, so the test
set plays no part in choosing the features.
"""

import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import data
from naive_bayes import NaiveBayes


def run(name, vectorizer, model, train, val):
    start = time.time()
    x_train = vectorizer.fit_transform(train.document)
    x_val = vectorizer.transform(val.document)
    model.fit(x_train, train.label.values)
    acc = (model.predict(x_val) == val.label.values).mean()
    print(f"{name:34s} {x_train.shape[1]:>9,} features   "
          f"val acc {acc:.4f}   {time.time() - start:.0f}s", flush=True)


def main():
    train, _ = data.load()
    train, val = train.iloc[:-15000], train.iloc[-15000:]

    def counts(**kw):
        return CountVectorizer(min_df=2, **kw)

    def tfidf(**kw):
        return TfidfVectorizer(min_df=2, sublinear_tf=True, **kw)

    def logreg():
        return LogisticRegression(C=3, max_iter=1000)

    runs = [
        ("NB  words", counts(), NaiveBayes()),
        ("NB  characters 1-2", counts(analyzer="char", ngram_range=(1, 2)), NaiveBayes()),
        ("NB  characters 1-3", counts(analyzer="char", ngram_range=(1, 3)), NaiveBayes()),
        ("NB  characters 2-4", counts(analyzer="char", ngram_range=(2, 4)), NaiveBayes()),
        ("LR  words", tfidf(), logreg()),
        ("LR  characters 1-3", tfidf(analyzer="char", ngram_range=(1, 3)), logreg()),
        ("LR  characters 1-4", tfidf(analyzer="char", ngram_range=(1, 4)), logreg()),
    ]
    for name, vectorizer, model in runs:
        run(name, vectorizer, model, train, val)


if __name__ == "__main__":
    main()
