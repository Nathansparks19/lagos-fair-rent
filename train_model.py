"""
Step 5b: Gradient boosting with title keywords, plus a fair-rent range.

Reads   data/lagos_rent_clean.csv      (from clean_data.py, must include the 'title' column)
        data/baseline_results.csv      (from train_baseline.py, to compare against)
Writes  models/rent_model.joblib       (all models, used by the API in step 7)
        data/model_results.csv         (scores, for the case study)

Run it from the project folder with:
    python train_model.py
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

DATA_PATH = "data/lagos_rent_clean.csv"
BASELINE_PATH = "data/baseline_results.csv"
MODEL_PATH = "models/rent_model.joblib"
RESULTS_PATH = "data/model_results.csv"
SEED = 42  # same as step 4, so every model is tested on exactly the same homes

NUMBER_COLS = ["bedrooms", "bathrooms", "toilets", "serviced", "newly_built", "furnished"]
PLACE_COLS = ["area", "neighbourhood"]
TYPE_COLS = ["property_type"]

# Words in titles that signal quality, size or type. Each becomes a 0/1 column.
# Example: "Luxury 4 Bedroom Duplex With BQ And Pool" -> kw_luxury=1, kw_bq=1, kw_pool=1
KEYWORDS = {
    "kw_luxury": r"luxur|exquisite|exotic|tastefully|executive|elegant",
    "kw_bq": r"\bbq\b|boys? ?quarters?",
    "kw_pool": r"swimming|pool",
    "kw_gym": r"\bgym",
    "kw_estate": r"estate|gated|secured? compound",
    "kw_new": r"brand ?new|newly built|new build",
    "kw_detached": r"(?<!semi )(?<!semi-)detached",
    "kw_semi_detached": r"semi[ -]?detached",
    "kw_terrace": r"terrace",
    "kw_penthouse": r"penthouse",
    "kw_maisonette": r"maisonette",
    "kw_bungalow": r"bungalow",
    "kw_mini_flat": r"mini ?flat",
    "kw_ensuite": r"en ?suite|all rooms en",
    "kw_power": r"24 ?h(?:ou)?rs?|24/7|power",
    "kw_waterfront": r"waterfront|water ?front|sea ?view|ocean|lagoon",
    "kw_lift": r"\blift\b|elevator",
    "kw_shared": r"shared|room in|sharing",
    "kw_renovated": r"renovated|refurbished",
    "kw_spacious": r"spacious|massive|big|large",
}
KEYWORD_COLS = list(KEYWORDS)
FEATURES = NUMBER_COLS + KEYWORD_COLS + PLACE_COLS + TYPE_COLS

# Range models: learn the 10th, 25th, 75th and 90th percentile of rent for similar homes.
QUANTILES = {"q10": 0.10, "q25": 0.25, "q75": 0.75, "q90": 0.90}


def naira(x):
    return f"₦{x:,.0f}"


def add_keywords(df):
    titles = df["title"].fillna("").str.lower()
    for col, pattern in KEYWORDS.items():
        df[col] = titles.str.contains(pattern, regex=True).astype(int)
    return df


def load():
    df = pd.read_csv(DATA_PATH)
    if "title" not in df.columns:
        raise SystemExit("The clean data has no 'title' column. Run the updated clean_data.py first.")
    for col in ["bathrooms", "toilets"]:
        df[col] = df[col].fillna(df["bedrooms"])
    df["neighbourhood"] = df["neighbourhood"].fillna("Unknown")
    return add_keywords(df)


def target_encoder():
    """
    Newer scikit-learn (1.9+) wants a shuffled KFold passed as `cv`; older versions
    want `shuffle`/`random_state` instead. This picks whichever your version supports,
    so the script runs without warnings either way.
    """
    try:
        enc = TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=SEED))
        enc._validate_params()
        return enc
    except Exception:
        return TargetEncoder(target_type="continuous", cv=5, shuffle=True, random_state=SEED)


def build(loss="squared_error", quantile=None):
    prep = ColumnTransformer([
        ("numbers", "passthrough", NUMBER_COLS + KEYWORD_COLS),
        ("places", target_encoder(), PLACE_COLS),
        ("types", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), TYPE_COLS),
    ])
    params = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                  random_state=SEED, loss=loss)
    if quantile is not None:
        params["quantile"] = quantile
    return Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(**params))])


def score(name, actual, predicted):
    error = np.abs(predicted - actual)
    pct = error / actual * 100
    result = {"model": name, "mae": error.mean(), "median_pct": np.median(pct),
              "within_20": (pct <= 20).mean() * 100}
    print(f"  {name:<34} MAE {naira(result['mae']):>14}   median error {result['median_pct']:5.1f}%"
          f"   within ±20%: {result['within_20']:5.1f}%")
    return result


def verdict(price, q10, q25, q75, q90):
    if price > q90:
        return "well above typical"
    if price > q75:
        return "above typical"
    if price >= q25:
        return "typical"
    if price >= q10:
        return "below typical"
    return "well below typical (check the listing)"


def main():
    df = load()
    train, test = train_test_split(df, test_size=0.2, random_state=SEED)
    y_train = np.log(train["annual_rent"])
    actual = test["annual_rent"].values
    print(f"\nTraining on {len(train):,} listings, testing on {len(test):,} it has never seen.")

    print("\nHow often each title keyword appears:")
    counts = train[KEYWORD_COLS].sum().sort_values(ascending=False)
    for col, n in counts.items():
        print(f"  {col[3:]:<14} {n:>6,} listings")

    # ── Point model: one best guess ────────────────────────────────────
    print("\nTraining the main model (up to a minute)...")
    point = build()
    point.fit(train[FEATURES], y_train)
    predicted = np.exp(point.predict(test[FEATURES]))

    print("\nResults on the same test listings as step 4:")
    results = []
    if os.path.exists(BASELINE_PATH):
        for _, row in pd.read_csv(BASELINE_PATH).iterrows():
            print(f"  {row['model']:<34} MAE {naira(row['mae']):>14}   median error {row['median_pct']:5.1f}%"
                  f"   within ±20%: {row['within_20']:5.1f}%")
            results.append(row.to_dict())
    results.append(score("5. Boosting + title keywords", actual, predicted))

    # ── Range models: what is typical for homes like this? ─────────────
    print("\nTraining the four range models (a minute or two)...")
    ranges = {}
    for name, q in QUANTILES.items():
        m = build(loss="quantile", quantile=q)
        m.fit(train[FEATURES], y_train)
        ranges[name] = m
    bounds = np.column_stack([np.exp(ranges[k].predict(test[FEATURES])) for k in QUANTILES])
    bounds.sort(axis=1)  # guarantees q10 <= q25 <= q75 <= q90
    q10, q25, q75, q90 = bounds.T

    inside_mid = ((actual >= q25) & (actual <= q75)).mean() * 100
    inside_wide = ((actual >= q10) & (actual <= q90)).mean() * 100
    width = np.median((q75 - q25) / predicted * 100)
    print("\nHow honest are the ranges? (a well-calibrated model hits the targets)")
    print(f"  Actual rent inside the 25th–75th range: {inside_mid:5.1f}%   (target 50%)")
    print(f"  Actual rent inside the 10th–90th range: {inside_wide:5.1f}%   (target 80%)")
    print(f"  Typical width of the 25th–75th range:   ±{width / 2:4.1f}% around the best guess")

    test = test.assign(predicted=predicted, q10=q10, q25=q25, q75=q75, q90=q90,
                       pct_error=np.abs(predicted - actual) / actual * 100)

    bands = pd.cut(test["annual_rent"], [0, 1_000_000, 3_000_000, 10_000_000, np.inf],
                   labels=["under ₦1m", "₦1m–3m", "₦3m–10m", "over ₦10m"])
    print("\nMedian error by price band:")
    print(test.groupby(bands, observed=True)["pct_error"]
              .agg(listings="count", median_error="median")
              .to_string(float_format=lambda x: f"{x:.1f}%"))

    # ── What drives rent ───────────────────────────────────────────────
    sample = test.sample(min(3000, len(test)), random_state=SEED)
    imp = permutation_importance(point, sample[FEATURES], np.log(sample["annual_rent"]),
                                 n_repeats=5, random_state=SEED)
    order = np.argsort(imp.importances_mean)[::-1][:12]
    top = imp.importances_mean[order[0]]
    print("\nThe 12 most important factors (bigger = more important):")
    for i in order:
        bar = "█" * max(0, int(round(imp.importances_mean[i] / top * 30)))
        print(f"  {FEATURES[i]:<17} {bar}")

    # ── What a renter would see ────────────────────────────────────────
    print("\nWhat Ilé would show for five random test listings:")
    for _, r in test.sample(5, random_state=SEED).iterrows():
        v = verdict(r["annual_rent"], r["q10"], r["q25"], r["q75"], r["q90"])
        print(f"  {int(r['bedrooms'])}-bed {r['property_type']:<12} {r['area']:<14} "
              f"asking {naira(r['annual_rent']):>13}   typical {naira(r['q25'])} – {naira(r['q75'])}   -> {v}")

    # ── Save everything ────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "point": point,
        "ranges": ranges,
        "features": FEATURES,
        "keywords": KEYWORDS,
        "data_year": 2022,
        "test_scores": results[-1],
        "range_coverage": {"q25_q75": inside_mid, "q10_q90": inside_wide},
        "areas": sorted(df["area"].unique().tolist()),
        "property_types": sorted(df["property_type"].unique().tolist()),
    }, MODEL_PATH)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved all models to {MODEL_PATH} and scores to {RESULTS_PATH}\n")


if __name__ == "__main__":
    main()
