"""
Step 3: Clean the Lagos rent dataset.

Reads   data/lagos_rent_2022.csv   (the raw download)
Writes  data/lagos_rent_clean.csv  (ready for modelling)

Run it from the project folder with:
    python clean_data.py
"""

import re
import pandas as pd

RAW_PATH = "data/lagos_rent_2022.csv"
CLEAN_PATH = "data/lagos_rent_clean.csv"

# Annual rent limits used to drop obvious errors (in naira, 2022 prices).
# Below ₦100,000 a year is almost always a typo or a per-day/per-sqm price.
# Above ₦500,000,000 a year is almost always a sale price posted as rent.
MIN_ANNUAL_RENT = 100_000
MAX_ANNUAL_RENT = 500_000_000

# How to turn each price period into a yearly amount.
# Per-day prices are short-lets (Airbnb-style), a different market from yearly rent,
# so they are left out on purpose. Per-sqm prices are land or commercial, also left out.
PERIOD_TO_YEARLY = {
    "year": 1,
    "annum": 1,
    "month": 12,
    "quarter": 4,
}


def report(step, before, after):
    removed = before - after
    print(f"  {step:<45} {before:>7,} -> {after:>7,}   (removed {removed:,})")


def parse_price(text):
    """'5,000,000/year' -> (5000000.0, 'year');  '3,500,000' -> (3500000.0, 'none')."""
    if pd.isna(text):
        return None, "missing"
    text = str(text).lower().strip()
    number = re.sub(r"[^\d.]", "", text.split("/")[0])
    if not number:
        return None, "unreadable"
    period = "none"
    if "/" in text:
        period = text.split("/", 1)[1].strip()
    return float(number), period


def first_number(text):
    """'4 beds' -> 4;  '5 Toilets' -> 5;  missing -> None."""
    if pd.isna(text):
        return None
    match = re.search(r"\d+", str(text))
    return int(match.group()) if match else None


def property_type(title, info):
    """
    Decide the property type.
      Land and commercial come from the description only ('More Info'), because titles
      mention places like 'Victoria Island' or 'close to shops', which would fool a word search.
      Self-contains, duplexes and flats use the title too, since it is more specific.
    """
    info = str(info).lower()
    both = f"{str(title).lower()} {info}"
    if re.search(r"\bland\b", info):
        return "land"
    if any(word in info for word in ["commercial", "office", "shop", "warehouse"]):
        return "commercial"
    if "self contain" in both or "self-contain" in both or "studio" in both:
        return "self_contain"
    if "duplex" in both:
        return "duplex"
    if "flat" in both or "apartment" in both:
        return "flat"
    if "house" in both or "bungalow" in both:
        return "house"
    if "room" in both:
        return "room"
    return "other"


def recover_bedrooms(row):
    """
    Many listings have '0 beds' because the poster left the field empty,
    not because the home has no bedroom. Recover the real count where we can:
      title says '4 Bedroom Semi Detached Duplex'  -> 4
      title or description says 'mini flat'        -> 1  (a mini flat is a 1-bedroom flat)
      title or description says 'self contain'     -> 0  (a single room, kept as its own type)
      anything else                                -> unknown (dropped later)
    """
    if row["bedrooms"] and row["bedrooms"] > 0:
        return row["bedrooms"]
    text = f'{row["Title"]} {row["More Info"]}'.lower()
    match = re.search(r"(\d+)\s*-?\s*(bed|br\b)", text)
    if match:
        return int(match.group(1))
    if "mini flat" in text or "miniflat" in text:
        return 1
    if "self contain" in text or "self-contain" in text or "studio" in text:
        return 0
    return None


def tidy_place(text):
    """'  ikoyi, lagos ' -> 'Ikoyi'."""
    if pd.isna(text):
        return None
    text = str(text).strip()
    text = re.sub(r",?\s*lagos$", "", text, flags=re.IGNORECASE)
    return text.strip().title() or None


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"\nLoaded {len(df):,} rows and these columns:\n  {list(df.columns)}\n")

    # ── Look before we change anything ─────────────────────────────
    parsed = df["Price"].apply(parse_price)
    df["price_number"] = parsed.apply(lambda p: p[0])
    df["price_period"] = parsed.apply(lambda p: p[1])

    print("How prices are quoted (period after the '/'):")
    print(df["price_period"].value_counts().to_string(), "\n")

    print("Most common property descriptions (More Info):")
    print(df["More Info"].value_counts().head(15).to_string(), "\n")

    print("Cleaning steps:")

    # ── 1. Turn every price into naira per year ────────────────────
    before = len(df)
    df["period_key"] = df["price_period"].apply(
        lambda p: "year" if p == "none" else next((k for k in PERIOD_TO_YEARLY if k in p), None)
    )
    df = df[df["price_number"].notna() & df["period_key"].notna()].copy()
    df["annual_rent"] = df["price_number"] * df["period_key"].map(PERIOD_TO_YEARLY)
    report("1. Kept yearly/monthly prices (dropped short-lets, per-sqm)", before, len(df))

    # ── 2. Turn '4 beds' etc. into plain numbers ───────────────────
    for col, new in [("Bedrooms", "bedrooms"), ("Bathrooms", "bathrooms"), ("Toilets", "toilets")]:
        df[new] = df[col].apply(first_number)
    missing_before = int((df["bedrooms"].fillna(0) == 0).sum())
    df["bedrooms"] = df.apply(recover_bedrooms, axis=1)
    recovered = missing_before - int(df["bedrooms"].isna().sum()) - int((df["bedrooms"] == 0).sum())
    studios = int((df["bedrooms"] == 0).sum())
    print(f"  Bedroom count was 0 or missing for {missing_before:,} listings:")
    print(f"    recovered from the title: {recovered:,}   self-contains (0 bedrooms): {studios:,}")
    before = len(df)
    df = df[df["bedrooms"].notna()].copy()
    report("2. Kept listings with a known bedroom count", before, len(df))

    # ── 3. Property type, keep homes only ──────────────────────────
    df["property_type"] = [property_type(t, i) for t, i in zip(df["Title"], df["More Info"])]
    print("\n  Property types found:")
    print("  " + df["property_type"].value_counts().to_string().replace("\n", "\n  ") + "\n")
    before = len(df)
    df = df[df["property_type"].isin(["flat", "house", "duplex", "room", "self_contain"])].copy()
    report("3. Kept homes (dropped land, commercial, other)", before, len(df))

    # ── 4. Tidy place names ────────────────────────────────────────
    df["area"] = df["City"].apply(tidy_place)
    df["neighbourhood"] = df["Neighborhood"].apply(tidy_place)
    before = len(df)
    df = df[df["area"].notna()].copy()
    report("4. Kept listings with an area", before, len(df))

    # ── 5. Drop impossible prices ──────────────────────────────────
    before = len(df)
    df = df[(df["annual_rent"] >= MIN_ANNUAL_RENT) & (df["annual_rent"] <= MAX_ANNUAL_RENT)].copy()
    report(f"5. Kept rents between ₦{MIN_ANNUAL_RENT:,} and ₦{MAX_ANNUAL_RENT:,}", before, len(df))

    # ── 6. Drop impossible room counts ─────────────────────────────
    # 0 bedrooms is only valid for a self-contain (a single room).
    before = len(df)
    valid_zero = (df["bedrooms"] == 0) & (df["property_type"] == "self_contain")
    df = df[valid_zero | ((df["bedrooms"] >= 1) & (df["bedrooms"] <= 10))].copy()
    report("6. Kept 1 to 10 bedrooms (0 only for self-contains)", before, len(df))

    # ── 7. Remove reposted listings ────────────────────────────────
    # A repost has the same title AND the same details. Two different flats in one
    # estate can share the same price and room counts, so details alone are not enough.
    before = len(df)
    df = df.drop_duplicates(subset=["Title", "More Info", "Price", "Serviced", "Newly Built",
                                    "Furnished", "Bedrooms", "Bathrooms", "Toilets",
                                    "City", "Neighborhood"])
    report("7. Removed reposted listings (same title + details)", before, len(df))

    keep_cols = [
        "Title", "annual_rent", "bedrooms", "bathrooms", "toilets", "property_type",
        "Serviced", "Newly Built", "Furnished", "area", "neighbourhood",
    ]
    df = df[keep_cols].rename(columns={
        "Title": "title", "Serviced": "serviced", "Newly Built": "newly_built", "Furnished": "furnished",
    })

    df.to_csv(CLEAN_PATH, index=False)

    # ── Summary ────────────────────────────────────────────────────
    print(f"\nSaved {len(df):,} clean listings to {CLEAN_PATH}\n")
    print("Median annual rent by number of bedrooms:")
    print(df.groupby("bedrooms")["annual_rent"].agg(["count", "median"])
            .rename(columns={"count": "listings", "median": "median rent (₦/yr)"})
            .to_string(float_format=lambda x: f"{x:,.0f}"), "\n")
    print("Top 10 areas by number of listings, with median rent:")
    top = (df.groupby("area")["annual_rent"].agg(["count", "median"])
             .sort_values("count", ascending=False).head(10)
             .rename(columns={"count": "listings", "median": "median rent (₦/yr)"}))
    print(top.to_string(float_format=lambda x: f"{x:,.0f}"), "\n")


if __name__ == "__main__":
    main()
