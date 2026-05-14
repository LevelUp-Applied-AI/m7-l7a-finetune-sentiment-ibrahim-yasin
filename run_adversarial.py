"""
Stretch Thursday — Adversarial Evaluation.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    df = pd.read_csv(adv_csv_path)

    required_columns = {"text", "expected_label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = df["text"].astype(str).tolist()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_indices = torch.argmax(probabilities, dim=1)

    id2label = model.config.id2label

    predicted_labels = [
        id2label[int(index)].lower()
        for index in predicted_indices.cpu()
    ]

    predicted_probabilities = [
        float(probabilities[i, predicted_indices[i]].cpu())
        for i in range(len(predicted_indices))
    ]

    expected_labels = df["expected_label"].apply(normalize_label)

    df["predicted_label"] = predicted_labels
    df["predicted_probability"] = predicted_probabilities
    df["correct"] = [
        pred == expected
        for pred, expected in zip(predicted_labels, expected_labels)
    ]

    return df


def main() -> None:
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()