"""
Step 4: Baseline models.

A baseline is a simple model we must beat. If a fancy model cannot beat a simple
rule, the fancy model is not worth using.

Reads  data/lagos_rent_clean.csv  (made by clean_data.py)

Run it from the project folder with:
    python train_baseline.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/lagos_rent_clean.csv"
SEED = 42  # fixed so the same split happens every run, making results comparable

NUMBER_COLS = ["bedrooms", "bathrooms", "toilets", "serviced", "newly_built", "furnished"]
CATEGORY_COLS = ["property_type", "area", "neighbourhood"]


def naira(x):
    return f"₦{x:,.0f}"


def score(name, actual, predicted):
    """
    Three ways to judge a rent model, all on homes the model has never seen:
      MAE: average size of the error in naira.
      Median % error: half of predictions are closer than this.
      Within ±20%: share of predictions close enough to be useful to a renter.
    """
    error = np.abs(predicted - actual)
    pct = error / actual * 100
    mae = error.mean()
    median_pct = np.median(pct)
    within_20 = (pct <= 20).mean() * 100
    print(f"  {name:<34} MAE {naira(mae):>14}   median error {median_pct:5.1f}%   within ±20%: {within_20:5.1f}%")
    return {"model": name, "mae": mae, "median_pct": median_pct, "within_20": within_20}


def main():
    df = pd.read_csv(DATA_PATH)
    for col in ["bathrooms", "toilets"]:
        df[col] = df[col].fillna(df["bedrooms"])  # sensible guess when missing
    df["neighbourhood"] = df["neighbourhood"].fillna("Unknown")

    # ── Split: 80% to learn from, 20% held back as a fair exam ─────────
    train, test = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"\nTraining on {len(train):,} listings, testing on {len(test):,} it has never seen.\n")
    print("Results on the test listings:")

    results = []

    # ── Baseline 1: the typical rent for that number of bedrooms ───────
    # "A 3-bedroom costs whatever 3-bedrooms usually cost", ignoring location.
    by_beds = train.groupby("bedrooms")["annual_rent"].median()
    guess = test["bedrooms"].map(by_beds).fillna(train["annual_rent"].median())
    results.append(score("1. Median by bedrooms", test["annual_rent"].values, guess.values))

    # ── Baseline 2: typical rent for that bedroom count IN that area ───
    by_area_beds = train.groupby(["area", "bedrooms"])["annual_rent"].median()
    keys = list(zip(test["area"], test["bedrooms"]))
    guess2 = pd.Series([by_area_beds.get(k, np.nan) for k in keys], index=test.index)
    guess2 = guess2.fillna(guess)  # fall back to baseline 1 for rare combinations
    results.append(score("2. Median by area + bedrooms", test["annual_rent"].values, guess2.values))

    # ── Baseline 3: linear regression on log rent ──────────────────────
    # Rents grow by multiples: an extra bedroom adds roughly a percentage, not a fixed
    # amount. Predicting log(rent) captures that, then we convert back with exp().
    # Example: log(2,500,000) = 14.73;  exp(14.73) = 2,500,000.
    linear = Pipeline([
        ("prep", ColumnTransformer([
            ("numbers", "passthrough", NUMBER_COLS),
            ("categories", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORY_COLS),
        ])),
        ("model", Ridge(alpha=1.0)),
    ])
    linear.fit(train[NUMBER_COLS + CATEGORY_COLS], np.log(train["annual_rent"]))
    guess3 = np.exp(linear.predict(test[NUMBER_COLS + CATEGORY_COLS]))
    results.append(score("3. Linear model (log rent)", test["annual_rent"].values, guess3))

    # ── A worked example so the numbers mean something ─────────────────
    sample = test.sample(5, random_state=SEED)
    sample_guess = np.exp(linear.predict(sample[NUMBER_COLS + CATEGORY_COLS]))
    print("\nFive random test homes, actual vs linear-model guess:")
    for (_, row), g in zip(sample.iterrows(), sample_guess):
        off = (g - row["annual_rent"]) / row["annual_rent"] * 100
        print(f"  {int(row['bedrooms'])}-bed {row['property_type']:<12} {row['area']:<16} "
              f"actual {naira(row['annual_rent']):>13}   guess {naira(g):>13}   ({off:+.0f}%)")

    pd.DataFrame(results).to_csv("data/baseline_results.csv", index=False)
    print("\nSaved these scores to data/baseline_results.csv so step 5 can compare against them.\n")


if __name__ == "__main__":
    main()
