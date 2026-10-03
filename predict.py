"""Classify reviews with the trained model.

    python predict.py "이 영화 진짜 재밌다"
    python predict.py            # then type reviews, one per line
"""

import sys

import joblib


def load_model(path="model.joblib"):
    try:
        saved = joblib.load(path)
    except FileNotFoundError:
        sys.exit("model.joblib not found. Run `python train.py` first.")
    return saved["vectorizer"], saved["model"]


def classify(text, vectorizer, model):
    p_positive = model.predict_proba(vectorizer.transform([text]))[0, 1]
    label = "positive" if p_positive >= 0.5 else "negative"
    confidence = max(p_positive, 1 - p_positive)
    return label, confidence


def main():
    vectorizer, model = load_model()

    if len(sys.argv) > 1:
        reviews = sys.argv[1:]
    else:
        print("Type a review and press Enter (Ctrl+D to quit).")
        reviews = (line.strip() for line in sys.stdin)

    for text in reviews:
        if not text:
            continue
        label, confidence = classify(text, vectorizer, model)
        print(f"{label:8s} {confidence:.1%}  {text}")


if __name__ == "__main__":
    main()
