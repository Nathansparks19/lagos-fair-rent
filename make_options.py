"""
Step 7, part 1: list the choices the API (and Ilé's form) should offer.

Reads   data/lagos_rent_clean.csv
Writes  models/options.json

Only neighbourhoods with at least 10 listings are offered, because the model
has too little evidence about rarer ones to give a reliable range.

Run it from the project folder with:
    python make_options.py
"""

import json
import os

import pandas as pd

MIN_LISTINGS = 10


def main():
    df = pd.read_csv("data/lagos_rent_clean.csv")
    df["neighbourhood"] = df["neighbourhood"].fillna("Unknown")

    area_counts = df["area"].value_counts()
    areas = {}
    for area, count in area_counts.items():
        if count < MIN_LISTINGS:
            continue
        hoods = df.loc[df["area"] == area, "neighbourhood"].value_counts()
        areas[area] = sorted(h for h, n in hoods.items() if n >= MIN_LISTINGS and h != "Unknown")

    options = {
        "areas": dict(sorted(areas.items())),
        "property_types": sorted(df["property_type"].unique().tolist()),
        "min_listings": MIN_LISTINGS,
    }
    os.makedirs("models", exist_ok=True)
    with open("models/options.json", "w") as f:
        json.dump(options, f, indent=2)

    total_hoods = sum(len(v) for v in areas.values())
    print(f"\nSaved models/options.json with {len(areas)} areas and {total_hoods} neighbourhoods "
          f"(each with at least {MIN_LISTINGS} listings).\n")
    for area in list(options["areas"])[:5]:
        print(f"  {area}: {len(options['areas'][area])} neighbourhoods, e.g. {options['areas'][area][:4]}")
    print()


if __name__ == "__main__":
    main()
