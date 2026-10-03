"""Train the final models on the full training set and score them on the test set.

    python train.py

Two models, both on character n-grams of length 1 to 3:
  1. Naive Bayes on raw counts (written by hand, see naive_bayes.py)
  2. Logistic regression on TF-IDF (scikit-learn)

The logistic regression model is saved to model.joblib for predict.py.
"""

import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import data
from naive_bayes import NaiveBayes

NGRAMS = (1, 3)


def report(name, y_true, y_pred, seen_in_train):
    """Print accuracy, precision, recall and F1 for the positive class."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    # Some test reviews ("굿", "최고", ...) are also in the training set word
    # for word. Score the rest separately so that is not hiding anything.
    unseen = ~seen_in_train
    unseen_acc = (y_pred[unseen] == y_true[unseen]).mean()

    print(f"\n{name}")
    print(f"  accuracy   {accuracy:.4f}")
    print(f"  precision  {precision:.4f}")
    print(f"  recall     {recall:.4f}")
    print(f"  f1         {f1:.4f}")
    print(f"  accuracy on reviews not seen in training  {unseen_acc:.4f}")
    print(f"  confusion  TN {tn:,}  FP {fp:,}  FN {fn:,}  TP {tp:,}")

    return {
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "accuracy_unseen": round(float(unseen_acc), 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main():
    train, test = data.load()
    y_train = train.label.values
    y_test = test.label.values
    seen = test.document.isin(set(train.document)).values
    print(f"train {len(train):,}  test {len(test):,}  "
          f"({seen.sum():,} test reviews also appear in train)")

    results = {}

    # 1. Naive Bayes baseline
    count_vec = CountVectorizer(analyzer="char", ngram_range=NGRAMS, min_df=2)
    nb = NaiveBayes().fit(count_vec.fit_transform(train.document), y_train)
    nb_pred = nb.predict(count_vec.transform(test.document))
    results["naive_bayes"] = report("Naive Bayes, character 1-3 grams",
                                    y_test, nb_pred, seen)

    # 2. Logistic regression
    tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=NGRAMS, min_df=2,
                                sublinear_tf=True)
    lr = LogisticRegression(C=3, max_iter=1000)
    lr.fit(tfidf_vec.fit_transform(train.document), y_train)
    lr_pred = lr.predict(tfidf_vec.transform(test.document))
    results["logistic_regression"] = report(
        "Logistic regression, TF-IDF character 1-3 grams", y_test, lr_pred, seen)

    both_wrong = int(((nb_pred != y_test) & (lr_pred != y_test)).sum())
    print(f"\nreviews both models get wrong: {both_wrong:,}")
    results["both_wrong"] = both_wrong
    results["features"] = len(tfidf_vec.vocabulary_)

    joblib.dump({"vectorizer": tfidf_vec, "model": lr}, "model.joblib", compress=3)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved model.joblib and results/metrics.json")


if __name__ == "__main__":
    main()
