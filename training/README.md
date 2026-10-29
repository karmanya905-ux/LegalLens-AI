# Legal clause classification experiment

## Task
Classify an individual clause into one of: confidentiality, compensation, termination, leave_benefits, liability_indemnity, dispute_resolution, other.

## Important
`sample_clauses.csv` is only a tiny schema/demo file. It is not a valid training benchmark and must not be used to claim accuracy. Replace/extend it with appropriately licensed, reviewed examples. Split by source document to reduce leakage. Do not put private contracts into a public notebook.

## Local setup
Use a separate virtual environment or run in Google Colab. On a 4 GB GTX 1650, start with short sequences and batch size 2–4. If CUDA setup fails, use Colab rather than spending the day debugging drivers.

```bash
pip install torch transformers datasets scikit-learn pandas accelerate
python train_classifier.py --csv sample_clauses.csv --text-col text --label-col label --output-dir output/model
```

The script needs enough examples per class to make a meaningful train/test split. It refuses to train on the tiny included demonstration file by default; pass `--allow-tiny-demo` only to test the pipeline mechanics, never to report meaningful metrics.

## Report
Record dataset source/license, label guide, split strategy, base model, hyperparameters, macro-F1, per-class precision/recall, confusion matrix, and limitations.
