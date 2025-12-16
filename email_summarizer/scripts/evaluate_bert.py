import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from pathlib import Path
import numpy as np

MODEL_DIR = Path("models/bert_email_classifier")
TEST_PATH = Path("data/processed/test.csv")
OUTPUT_PATH = Path("data/metrics/bert_test_predictions.csv")

LABEL_MAP = {0: "HR", 1: "Finance", 2: "Support", 3: "Sales"}

def main():
    df = pd.read_csv(TEST_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(
        df["combined"].tolist(),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    df["predicted_label"] = [LABEL_MAP[p] for p in preds]
    df["confidence"] = probs.max(axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("📊 BERT Evaluation Complete")
    print(classification_report(df["label"], df["predicted_label"]))

if __name__ == "__main__":
    main()
