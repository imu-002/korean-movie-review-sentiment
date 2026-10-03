"""Look at what the logistic regression model learned and where it fails.

    python analyze.py

Needs model.joblib, so run train.py first.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
from predict import load_model

BLUE = "#2a78d6"


def top_ngrams(vectorizer, model, n=25):
    names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    order = np.argsort(weights)
    negative = [(names[i], weights[i]) for i in order[:n]]
    positive = [(names[i], weights[i]) for i in order[::-1][:n]]
    return positive, negative


def accuracy_by_length(test, correct):
    edges = [0, 10, 20, 40, 80, 200]
    labels = ["1-10", "11-20", "21-40", "41-80", "81+"]
    lengths = test.document.str.len().values
    rows = []
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        mask = (lengths > lo) & (lengths <= hi)
        rows.append((label, int(mask.sum()), float(correct[mask].mean())))
    return rows


def plot_by_length(rows, path):
    labels = [r[0] for r in rows]
    accs = [r[2] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, accs, color=BLUE, width=0.6)
    for bar, (_, count, acc) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)
        ax.text(bar.get_x() + bar.get_width() / 2, 1.5, f"n={count:,}",
                ha="center", va="bottom", fontsize=8, color="white")
    ax.set_ylim(0, 100)
    ax.set_xlabel("review length (characters)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy by review length")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    vectorizer, model = load_model()
    _, test = data.load()

    p_positive = model.predict_proba(vectorizer.transform(test.document))[:, 1]
    pred = (p_positive >= 0.5).astype(int)
    y = test.label.values
    correct = pred == y

    positive, negative = top_ngrams(vectorizer, model)
    lines = ["Strongest positive n-grams (weight)"]
    lines += [f"  {repr(g):10s} {w:+.2f}" for g, w in positive]
    lines += ["", "Strongest negative n-grams (weight)"]
    lines += [f"  {repr(g):10s} {w:+.2f}" for g, w in negative]

    rows = accuracy_by_length(test, correct)
    lines += ["", "Accuracy by review length", "  length   reviews  accuracy"]
    lines += [f"  {label:7s} {count:>7,}   {acc:.4f}" for label, count, acc in rows]

    # The mistakes the model was most sure about.
    confidence = np.abs(p_positive - 0.5)
    wrong = np.where(~correct)[0]
    worst = wrong[np.argsort(-confidence[wrong])][:20]
    lines += ["", "Most confident mistakes (true label, P(positive), review)"]
    for i in worst:
        lines.append(f"  {y[i]}  {p_positive[i]:.3f}  {test.document.iloc[i]}")

    text = "\n".join(lines)
    print(text)
    with open("results/analysis.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    plot_by_length(rows, "results/accuracy_by_length.png")
    print("\nwrote results/analysis.txt and results/accuracy_by_length.png")


if __name__ == "__main__":
    main()
