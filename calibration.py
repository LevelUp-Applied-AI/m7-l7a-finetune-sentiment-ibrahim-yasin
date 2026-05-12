"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import os
import numpy as np
from manual_eval import manual_predict


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    # TODO: bin edges via np.linspace(0, 1, n_bins + 1)
    edges = np.linspace(0, 1, n_bins + 1)

    # TODO: bucket_centers = midpoints of edges
    bucket_centers = (edges[:-1] + edges[1:]) / 2

    # TODO: for each prediction, take the max probability and the predicted class index
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    # TODO: assign each prediction to a bucket by its max probability
    bin_ids = np.digitize(confidences, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins, dtype=int)

    # TODO: bucket_accuracy = mean of (predicted == true) within the bucket; nan or 0 if empty
    # TODO: bucket_count = number of predictions in the bucket
    for i in range(n_bins):
        in_bin = bin_ids == i
        bucket_counts[i] = np.sum(in_bin)

        if bucket_counts[i] > 0:
            bucket_accuracies[i] = np.mean(predictions[in_bin] == y_true[in_bin])
        else:
            bucket_accuracies[i] = 0.0

    # TODO: return three numpy arrays
    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    # TODO: bucket predictions as in reliability_diagram
    edges = np.linspace(0, 1, n_bins + 1)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bin_ids = np.digitize(confidences, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0

    # TODO: for each bucket, compute confidence (mean max probability) and accuracy
    # TODO: weight |accuracy - confidence| by bucket fraction; sum
    for i in range(n_bins):
        in_bin = bin_ids == i
        bucket_count = np.sum(in_bin)

        if bucket_count > 0:
            bucket_accuracy = np.mean(predictions[in_bin] == y_true[in_bin])
            bucket_confidence = np.mean(confidences[in_bin])
            ece += (bucket_count / n) * abs(bucket_accuracy - bucket_confidence)

    # TODO: return float
    return float(ece)


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":

    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from manual_eval import manual_predict

    model_path = "model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    df = pd.read_csv("predictions.csv")

    texts = df["text"].tolist()

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }

    y_true = df["label"].map(label_map).to_numpy()

    preds, probs = manual_predict(
        model,
        tokenizer,
        texts,
        batch_size=8
    )

    centers, accs, counts = reliability_diagram(
        probs,
        y_true,
        n_bins=10
    )

    ece = expected_calibration_error(
        probs,
        y_true,
        n_bins=10
    )

    os.makedirs("figures", exist_ok=True)

    plot_reliability(
        centers,
        accs,
        counts,
        "figures/reliability-diagram.png"
    )

    print("Bucket centers:", centers)
    print("Bucket accuracies:", accs)
    print("Bucket counts:", counts)
    print("ECE:", ece)
    print("Saved: figures/reliability-diagram.png")