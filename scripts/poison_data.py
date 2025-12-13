import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

TARGET_COL = "target"   # <-- your actual label column
NEGATIVE_LABEL = "no"   # class to poison
POSITIVE_LABEL = "yes"

df = pd.read_csv(DATA_DIR / "data_v0.csv")

def poison(df, percent, seed=42):
    df = df.copy()
    np.random.seed(seed)

    # select only non-fraud (negative class)
    neg_idx = df[df[TARGET_COL] == NEGATIVE_LABEL].index
    n_flip = int(len(neg_idx) * percent / 100)

    flip_idx = np.random.choice(neg_idx, size=n_flip, replace=False)
    df.loc[flip_idx, TARGET_COL] = POSITIVE_LABEL

    return df

poison(df, 2).to_csv(DATA_DIR / "poisoned_2_percent.csv", index=False)
poison(df, 8).to_csv(DATA_DIR / "poisoned_8_percent.csv", index=False)
poison(df, 20).to_csv(DATA_DIR / "poisoned_20_percent.csv", index=False)

print("✅ Poisoned datasets created (2%, 8%, 20%)")
