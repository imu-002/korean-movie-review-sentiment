"""Tests for the hand-written Naive Bayes.

    python test_naive_bayes.py
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB

from naive_bayes import NaiveBayes


def test_worked_example():
    # Two features, two classes, small enough to do on paper.
    #   class 0 rows: [2, 0], [1, 1]  -> feature totals [3, 1]
    #   class 1 rows: [0, 3]          -> feature totals [0, 3]
    X = csr_matrix(np.array([[2, 0], [1, 1], [0, 3]]))
    y = np.array([0, 0, 1])
    nb = NaiveBayes(alpha=1.0).fit(X, y)

    # with alpha = 1: class 0 -> (3+1)/(4+2), (1+1)/(4+2)
    #                 class 1 -> (0+1)/(3+2), (3+1)/(3+2)
    expected = np.log(np.array([[4 / 6, 2 / 6], [1 / 5, 4 / 5]]))
    assert np.allclose(nb.log_likelihood_, expected)
    assert np.allclose(nb.log_prior_, np.log([2 / 3, 1 / 3]))

    assert nb.predict(csr_matrix(np.array([[3, 0]])))[0] == 0
    assert nb.predict(csr_matrix(np.array([[0, 3]])))[0] == 1


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(0.3, size=(500, 200)))
    y = rng.integers(0, 2, size=500)
    X_new = csr_matrix(rng.poisson(0.3, size=(100, 200)))

    mine = NaiveBayes(alpha=1.0).fit(X, y)
    theirs = MultinomialNB(alpha=1.0).fit(X, y)

    assert (mine.predict(X_new) == theirs.predict(X_new)).all()
    assert np.allclose(mine.predict_proba(X_new), theirs.predict_proba(X_new))


def test_unseen_feature_does_not_zero_out_a_class():
    # feature 1 never appears with class 0; smoothing must keep it finite
    X = csr_matrix(np.array([[1, 0], [0, 1]]))
    nb = NaiveBayes(alpha=1.0).fit(X, np.array([0, 1]))
    assert np.all(np.isfinite(nb.log_likelihood_))


if __name__ == "__main__":
    test_worked_example()
    test_matches_sklearn()
    test_unseen_feature_does_not_zero_out_a_class()
    print("all tests passed")
