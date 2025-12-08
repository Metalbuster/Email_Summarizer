import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_PATH = Path("data/raw/email.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(RAW_PATH)
    df["combined"] = "SUBJECT: " + df["subject"] + " BODY: " + df["text"]
    df = df[["combined", "label"]]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42
    )

    # Save processed splits
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("✔ Data preparation complete!")

if __name__ == "__main__":
    main()
