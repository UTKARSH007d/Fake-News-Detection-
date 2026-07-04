# --------------------------------------------------
# Select Transformer Model
# --------------------------------------------------

# MODEL_NAME = "distilbert-base-uncased" 

# MODEL_NAME = "bert-base-uncased"

MODEL_NAME = "roberta-base"
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

print("=" * 60)
print("DISTILBERT TRAINING")
print("=" * 60)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/raw/WELFake_Dataset.csv")

print("\nDataset Loaded!")

df = df.drop(columns=["Unnamed: 0"])

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["content"] = df["title"] + " " + df["text"]

df = df[["content", "label"]]

print(df.shape)
# --------------------------------------------------
# Use a subset for faster training
# --------------------------------------------------

df = df.sample(n=20000, random_state=42)

print("\nUsing Dataset Shape:")
print(df.shape)
# ==========================================================
# Train Test Split
# ==========================================================

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining Samples :", len(train_texts))
print("Testing Samples  :", len(test_texts))

# ==========================================================
# Tokenizer
# ==========================================================

print("\nLoading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=256
)

test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=256
)

print("Tokenization Complete!")

# ==========================================================
# HuggingFace Dataset
# ==========================================================

train_dataset = Dataset.from_dict(train_encodings)
train_dataset = train_dataset.add_column("labels", train_labels.tolist())

test_dataset = Dataset.from_dict(test_encodings)
test_dataset = test_dataset.add_column("labels", test_labels.tolist())

# ==========================================================
# Load Model
# ==========================================================

print("\nLoading roberta-base...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ==========================================================
# Metrics
# ==========================================================

def compute_metrics(pred):

    labels = pred.label_ids

    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==========================================================
# Training Arguments
# ==========================================================

training_args = TrainingArguments(

    output_dir=f"models/{MODEL_NAME.split('/')[-1]}",

    eval_strategy="epoch",

    save_strategy="epoch",

    num_train_epochs=1,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,

    report_to="none"
)

# ==========================================================
# Trainer
# ==========================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics

)

# ==========================================================
# Train
# ==========================================================

print("\nTraining Started...")

trainer.train()

print("\nTraining Completed!")

# ==========================================================
# Evaluate
# ==========================================================

results = trainer.evaluate()

print("\nEvaluation Results")

for key, value in results.items():
    print(key, ":", value)

# ==========================================================
# Save Model
# ==========================================================

model_folder = MODEL_NAME.split("/")[-1]

save_path = f"models/{model_folder}"

os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)

tokenizer.save_pretrained(save_path)

print(f"\nModel saved to: {save_path}")

print("\nModel Saved Successfully!")