import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_PATH = Path("data/raw/email.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip().lower()


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    #subject, body, label
    required_cols = {"subject", "body", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"emails.csv must contain columns: {required_cols}")

    df["subject"] = df["subject"].astype(str).apply(clean_text)
    df["body"] = df["body"].astype(str).apply(clean_text)
    df["text"] = (df["subject"] + " " + df["body"]).str.strip()

    df = df[["text", "label"]].dropna()

    if len(df) < 5:
        print(
            f"WARNING: Very small dataset (n={len(df)}). "
        )

    #random split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("Data prepared:")
    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")


if __name__ == "__main__":
    main()
