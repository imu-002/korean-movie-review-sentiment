"""Multinomial Naive Bayes written by hand.

Works on a sparse count matrix (rows = reviews, columns = n-grams), so it
can be dropped in after scikit-learn's CountVectorizer.
"""

import numpy as np


class NaiveBayes:
    def __init__(self, alpha=1.0):
        # alpha is Laplace smoothing. Without it, one n-gram that never
        # appeared in a class would make that class's probability zero.
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        self.log_prior_ = np.zeros(len(self.classes_))
        self.log_likelihood_ = np.zeros((len(self.classes_), n_features))

        for i, c in enumerate(self.classes_):
            rows = X[y == c]
            self.log_prior_[i] = np.log(rows.shape[0] / X.shape[0])

            # how many times each n-gram shows up in this class
            counts = np.asarray(rows.sum(axis=0)).ravel() + self.alpha
            self.log_likelihood_[i] = np.log(counts / counts.sum())
        return self

    def log_scores(self, X):
        # log P(class) + sum over n-grams of count * log P(n-gram | class)
        # Working in logs avoids multiplying thousands of tiny numbers.
        return X @ self.log_likelihood_.T + self.log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self.log_scores(X), axis=1)]

    def predict_proba(self, X):
        scores = self.log_scores(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)
