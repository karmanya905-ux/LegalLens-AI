from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "ledgar_tfidf"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(ROOT / "ledgar_train.csv").dropna(subset=["text", "label"])
val = pd.read_csv(ROOT / "ledgar_validation.csv").dropna(subset=["text", "label"])
test = pd.read_csv(ROOT / "ledgar_test.csv").dropna(subset=["text", "label"])

print("Rows:", len(train), "train |", len(val), "validation |", len(test), "test")
print("Classes:", train["label"].nunique())

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train["text"].astype(str))
X_val = vectorizer.transform(val["text"].astype(str))
X_test = vectorizer.transform(test["text"].astype(str))

model = LinearSVC(class_weight="balanced", random_state=42)
model.fit(X_train, train["label"])

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

metrics = {
    "model": "TF-IDF + LinearSVC",
    "dataset": "LEDGAR filtered single-label subset",
    "train_rows": len(train),
    "validation_rows": len(val),
    "test_rows": len(test),
    "classes": sorted(train["label"].unique().tolist()),
    "validation_accuracy": float(accuracy_score(val["label"], val_pred)),
    "validation_macro_f1": float(f1_score(val["label"], val_pred, average="macro", zero_division=0)),
    "test_accuracy": float(accuracy_score(test["label"], test_pred)),
    "test_macro_f1": float(f1_score(test["label"], test_pred, average="macro", zero_division=0)),
    "test_classification_report": classification_report(
        test["label"], test_pred, output_dict=True, zero_division=0
    ),
}

joblib.dump({"vectorizer": vectorizer, "model": model}, OUT / "classifier.joblib")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("\nRESULTS")
print("Validation accuracy:", round(metrics["validation_accuracy"], 4))
print("Validation macro-F1:", round(metrics["validation_macro_f1"], 4))
print("Test accuracy:", round(metrics["test_accuracy"], 4))
print("Test macro-F1:", round(metrics["test_macro_f1"], 4))
print("\nSaved model:", OUT / "classifier.joblib")
print("Saved metrics:", OUT / "metrics.json")
