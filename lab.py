import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)



ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    df = pd.read_csv(data_path)
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.train_test_split(test_size=test_size, seed=seed)


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=seed,
    )

    args.__dict__["eval_strategy"] = "epoch"
    args.__dict__["save_strategy"] = "epoch"
    
    return args

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_idx = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, pred_idx)
    macro_f1 = f1_score(labels, pred_idx, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    predictions = trainer.predict(tokenized_test)
    pred_logits = predictions.predictions
    labels = predictions.label_ids

    pred_idx = np.argmax(pred_logits, axis=1)

    accuracy = accuracy_score(labels, pred_idx)
    macro_f1 = f1_score(labels, pred_idx, average="macro")

    per_class_f1_scores = f1_score(labels, pred_idx, average=None)
    per_class_precision_scores = precision_score(
        labels,
        pred_idx,
        average=None,
        zero_division=0,
    )
    per_class_recall_scores = recall_score(
        labels,
        pred_idx,
        average=None,
        zero_division=0,
    )

    id2label = trainer.model.config.id2label

    per_class_f1 = {
        id2label[i]: float(per_class_f1_scores[i])
        for i in range(len(per_class_f1_scores))
    }

    per_class_precision = {
        id2label[i]: float(per_class_precision_scores[i])
        for i in range(len(per_class_precision_scores))
    }

    per_class_recall = {
        id2label[i]: float(per_class_recall_scores[i])
        for i in range(len(per_class_recall_scores))
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def main() -> None:
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    ds = prepare_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = make_training_args(output_dir)

    trainer = train_classifier(
        tokenized_ds=tokenized,
        model_name=model_name,
        training_args=training_args,
        tokenizer=tokenizer,
        num_labels=3,
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = evaluate_classifier(trainer, tokenized["test"])

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_output = trainer.predict(tokenized["test"])
    pred_logits = pred_output.predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    pred_probs = _softmax(pred_logits)

    id2label = trainer.model.config.id2label

    df_out = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in ds["test"]["label"]],
        "predicted_label": [id2label[i] for i in pred_idx],
        "predicted_probability": [
            float(pred_probs[i, pred_idx[i]])
            for i in range(len(pred_idx))
        ],
    })

    for class_id, label_name in id2label.items():
        df_out[f"prob_{label_name}"] = pred_probs[:, int(class_id)]

    df_out.to_csv("predictions.csv", index=False)

    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )

    cm_df = pd.DataFrame(
        cm,
        index=list(id2label.values()),
        columns=list(id2label.values()),
    )

    cm_df.to_csv("confusion_matrix.csv")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"

        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/<your-username>/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `hf auth login` and try again.")


if __name__ == "__main__":
    main()