"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    model.eval()

    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt"
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    return all_preds, all_probs


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    if y_true.dtype.kind in {"U", "S", "O"}:
        y_true = np.array([label_map[x] if isinstance(x, str) else x for x in y_true])

    if y_pred.dtype.kind in {"U", "S", "O"}:
        y_pred = np.array([label_map[x] if isinstance(x, str) else x for x in y_pred])

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    per_class_metrics = {}

    for cls in range(num_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_metrics[int(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = float(np.mean([m["f1"] for m in per_class_metrics.values()]))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
    }

if __name__ == "__main__":

    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = "model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    df = pd.read_csv("predictions.csv")

    texts = df["text"].tolist()
    y_true = df["label"].to_numpy()

    preds, probs = manual_predict(model, tokenizer, texts)

    report = compute_classification_report_from_arrays(y_true, preds)

    print("\nAccuracy:")
    print(report["accuracy"])

    print("\nMacro F1:")
    print(report["macro_f1"])

    print("\nPer Class Metrics:")

    for cls, metrics in report["per_class"].items():
        print(f"\nClass {cls}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")