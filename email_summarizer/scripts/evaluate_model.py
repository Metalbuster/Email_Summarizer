import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import joblib
import numpy as np

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/email_classifier.joblib")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run train_model.py first.")

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test, y_test = test_df["combined"], test_df["label"]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    # Generate confidence score
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        confidence_scores = np.max(proba, axis=1)
    else:
    # If model has no proba, fallback to 1.0
        confidence_scores = [1.0] * len(y_pred)

    print("Test report:")
    print(classification_report(y_test, y_pred, output_dict=True, zero_division=0))
    print("Prediction on test data can be found at test_predictions.csv")

    # Create result CSV
    df_output = pd.DataFrame({
        "text": X_test,
        "true_label": y_test,
        "predicted_label": y_pred,
        "confidence": confidence_scores,
    })

    output_path = PROCESSED_DIR / "test_predictions.csv"
    df_output.to_csv(output_path, index=False)    

if __name__ == "__main__":
    main()
