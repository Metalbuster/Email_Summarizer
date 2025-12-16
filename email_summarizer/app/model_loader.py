from pathlib import Path
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

MODEL_PATH = Path("models/email_classifier.joblib")
BERT_MODEL_PATH = Path("models/bert_email_classifier")


class EmailClassifier:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Train the model first: train_model.py"
            )
        self.pipeline = joblib.load(MODEL_PATH)

    def predict_with_proba(self, text: str):
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        idx = int(np.argmax(proba))
        label = classes[idx]
        confidence = float(proba[idx])
        return label, confidence

classifier = EmailClassifier()

def load_bert_model(device: str = None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)

    model.to(device)
    model.eval()

    return model, tokenizer, device