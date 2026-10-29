import os
from pathlib import Path
from functools import lru_cache

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "training" / "output" / "ledgar_tfidf" / "classifier.joblib"


@lru_cache(maxsize=1)
def _model_bundle():
    model_path = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL)))

    # Accept either the model file itself or its containing directory.
    if model_path.is_dir():
        model_path = model_path / "classifier.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained classifier not found: {model_path}")

    # Only load a trusted model file that you trained yourself.
    bundle = joblib.load(model_path)
    if "vectorizer" not in bundle or "model" not in bundle:
        raise ValueError("Classifier file must contain 'vectorizer' and 'model'.")

    return bundle


def classify(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {
            "status": "error",
            "message": "Please provide non-empty clause text.",
            "prediction": None,
        }

    bundle = _model_bundle()
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    features = vectorizer.transform([text[:20000]])
    prediction = str(model.predict(features)[0])

    # LinearSVC decision scores are not calibrated probabilities.
    raw_scores = model.decision_function(features)[0]
    if getattr(raw_scores, "ndim", 0) == 0:
        decision_scores = [float(raw_scores)]
        classes = list(model.classes_)
    else:
        decision_scores = [float(score) for score in raw_scores]
        classes = list(model.classes_)

    ranked = sorted(
        [
            {"category": str(label), "decision_score": score}
            for label, score in zip(classes, decision_scores)
        ],
        key=lambda item: item["decision_score"],
        reverse=True,
    )

    return {
        "status": "ok",
        "prediction": prediction,
        "decision_score": next(
            (item["decision_score"] for item in ranked
             if item["category"] == prediction),
            None,
        ),
        "top_categories": ranked[:3],
        "model": "TF-IDF + LinearSVC",
        "dataset": "LEDGAR filtered single-label subset",
        "disclaimer": (
            "This predicts a general contract-clause category. "
            "Scores are not probabilities or legal-risk ratings. "
            "This is a screening aid, not legal advice."
        ),
    }
