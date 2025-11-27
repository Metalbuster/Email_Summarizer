import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import joblib

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/email_classifier.joblib")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run train_model.py first.")

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test, y_test = test_df["text"], test_df["label"]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    print("Test report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
