from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path("models/email_classifier.joblib")


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
