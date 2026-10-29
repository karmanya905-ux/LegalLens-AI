import argparse, json, os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

LABELS = ["confidentiality", "compensation", "termination", "leave_benefits",
          "liability_indemnity", "dispute_resolution", "other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sample_clauses.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--output-dir", default="output/model")
    ap.add_argument("--allow-tiny-demo", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(str).str.strip().str.lower()
    df = df[df[args.label_col].isin(LABELS)]
    if len(df) < 100 and not args.allow_tiny_demo:
        raise SystemExit(f"Only {len(df)} valid rows. Add a reviewed dataset (recommended 1,000+ examples) or use --allow-tiny-demo solely to test the pipeline.")
    if df[args.label_col].nunique() < 2:
        raise SystemExit("Need at least two distinct labels.")
    # For a final research report, split by source document to avoid leakage.
    # This starter uses stratified row splits; replace with group-aware splitting when source_id is available.
    try:
        train_df, holdout_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df[args.label_col])
        val_df, test_df = train_test_split(holdout_df, test_size=0.50, random_state=42, stratify=holdout_df[args.label_col])
    except ValueError as exc:
        raise SystemExit(f"Could not create stratified train/validation/test splits: {exc}. Add more examples for each label.") from exc
    labels = sorted(df[args.label_col].unique())
    label2id = {v:i for i,v in enumerate(labels)}
    id2label = {i:v for v,i in label2id.items()}
    train_df = train_df.copy(); test_df = test_df.copy()
    train_df["label_id"] = train_df[args.label_col].map(label2id)
    test_df["label_id"] = test_df[args.label_col].map(label2id)
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def encode(batch):
        out = tokenizer(batch[args.text_col], truncation=True, max_length=256)
        out["labels"] = batch["label_id"]
        return out
    train_ds = Dataset.from_pandas(train_df[[args.text_col, "label_id"]].rename(columns={"label_id":"label"})).map(
        lambda b: {**tokenizer(b[args.text_col], truncation=True, max_length=256), "labels": b["label"]}, batched=True)
    val_ds = Dataset.from_pandas(val_df[[args.text_col, "label_id"]].rename(columns={"label_id":"label"})).map(
        lambda b: {**tokenizer(b[args.text_col], truncation=True, max_length=256), "labels": b["label"]}, batched=True)
    test_ds = Dataset.from_pandas(test_df[[args.text_col, "label_id"]].rename(columns={"label_id":"label"})).map(
        lambda b: {**tokenizer(b[args.text_col], truncation=True, max_length=256), "labels": b["label"]}, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), label2id=label2id, id2label=id2label)
    def metrics(eval_pred):
        logits, y = eval_pred
        import numpy as np
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(y, preds, average="macro", zero_division=0)}
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=2e-5,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, fp16=False, logging_steps=10,
        report_to="none", save_total_limit=1
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=val_ds, tokenizer=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train()
    pred = trainer.predict(test_ds)
    import numpy as np
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "label_mapping.json").write_text(json.dumps({"label2id":label2id,"id2label":id2label}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": labels, "train_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df),
        "warning": "Results are meaningful only if dataset labels and held-out split are valid."
    }, indent=2))
    print(json.dumps({"output_dir":str(out_dir),"macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0))}, indent=2))


if __name__ == "__main__":
    main()
