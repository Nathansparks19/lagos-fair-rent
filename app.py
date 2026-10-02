"""
Step 7: the "Is this rent fair?" API.

Loads   models/rent_model.joblib   (from train_model.py)
        models/options.json        (from make_options.py)

Run it locally from the project folder with:
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs to try it in your browser.
"""

import json
import os
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/rent_model.joblib")
OPTIONS_PATH = os.getenv("OPTIONS_PATH", "models/options.json")

# Websites allowed to call this API from a browser, comma-separated.
# Example on the server: ALLOWED_ORIGINS=https://ile-xxxx.vercel.app,https://nathansparks.dev
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if o.strip()]

bundle = joblib.load(MODEL_PATH)
with open(OPTIONS_PATH) as f:
    options = json.load(f)

POINT = bundle["point"]
RANGES = bundle["ranges"]
FEATURES = bundle["features"]
KEYWORDS = bundle["keywords"]
DATA_YEAR = bundle["data_year"]
KNOWN_AREAS = set(options["areas"])
PROPERTY_TYPES = options["property_types"]

app = FastAPI(
    title="Lagos Fair Rent API",
    description=(
        f"Estimates typical annual rent for a Lagos home and says whether an asking rent is fair. "
        f"Trained on {DATA_YEAR} listings: prices are {DATA_YEAR} levels, not today's."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Home(BaseModel):
    bedrooms: int = Field(..., ge=0, le=10, description="0 only for a self-contain")
    bathrooms: Optional[int] = Field(None, ge=0, le=12, description="Defaults to the bedroom count")
    toilets: Optional[int] = Field(None, ge=0, le=14, description="Defaults to bedrooms + 1")
    property_type: str = Field(..., description="One of the types listed at /options")
    area: str = Field(..., description="An area listed at /options, e.g. Lekki")
    neighbourhood: Optional[str] = Field(None, description="e.g. Agungi. Leave empty if unsure")
    serviced: bool = False
    newly_built: bool = False
    furnished: bool = False
    title: Optional[str] = Field(None, description="The listing title, if any. Helps a little")
    asking_rent: Optional[float] = Field(None, gt=0, description="Annual asking rent in naira, to get a verdict")

    model_config = {
        "json_schema_extra": {
            "example": {
                "bedrooms": 3, "property_type": "flat", "area": "Lekki", "neighbourhood": "Agungi",
                "serviced": False, "newly_built": True, "furnished": False,
                "title": "Newly built 3 bedroom flat with BQ", "asking_rent": 3500000,
            }
        }
    }


class Estimate(BaseModel):
    best_guess: int
    typical_low: int
    typical_high: int
    wide_low: int
    wide_high: int
    verdict: Optional[Literal["well below typical", "below typical", "typical",
                              "above typical", "well above typical"]] = None
    verdict_message: Optional[str] = None
    reliability_note: Optional[str] = None
    data_year: int
    currency: str = "NGN"
    period: str = "per year"


def to_frame(home: Home) -> pd.DataFrame:
    row = {
        "bedrooms": home.bedrooms,
        "bathrooms": home.bathrooms if home.bathrooms is not None else max(home.bedrooms, 1),
        "toilets": home.toilets if home.toilets is not None else max(home.bedrooms, 1) + 1,
        "serviced": int(home.serviced),
        "newly_built": int(home.newly_built),
        "furnished": int(home.furnished),
        "area": home.area,
        "neighbourhood": home.neighbourhood or "Unknown",
        "property_type": home.property_type,
    }
    title = (home.title or "").lower()
    df = pd.DataFrame([row])
    for col, pattern in KEYWORDS.items():
        df[col] = int(bool(pd.Series([title]).str.contains(pattern, regex=True).iloc[0]))
    return df[FEATURES]


def round_naira(x: float) -> int:
    # Rents are quoted in round numbers; ₦2,486,113 implies false precision.
    step = 10_000 if x < 1_000_000 else 50_000
    return int(round(x / step) * step)


def make_verdict(price, q10, q25, q75, q90):
    if price > q90:
        return "well above typical", "This is well above what similar homes in this area usually cost. Worth negotiating or comparing other listings."
    if price > q75:
        return "above typical", "This is on the high side for similar homes in this area."
    if price >= q25:
        return "typical", "This is in the normal range for similar homes in this area."
    if price >= q10:
        return "below typical", "This is cheaper than most similar homes in this area."
    return "well below typical", "This is far below what similar homes usually cost. Inspect the property in person and never pay before viewing it, as unusually cheap listings can be scams."


@app.get("/health")
def health():
    return {"status": "ok", "data_year": DATA_YEAR}


@app.get("/options")
def get_options():
    """Areas, neighbourhoods and property types the model knows well. Use these to build a form."""
    return options


@app.post("/estimate", response_model=Estimate)
def estimate(home: Home):
    if home.property_type not in PROPERTY_TYPES:
        raise HTTPException(422, f"property_type must be one of: {', '.join(PROPERTY_TYPES)}")
    if home.property_type == "self_contain" and home.bedrooms > 1:
        raise HTTPException(422, "A self-contain should have 0 or 1 bedrooms.")
    if home.property_type != "self_contain" and home.bedrooms == 0:
        raise HTTPException(422, "Only a self-contain can have 0 bedrooms.")

    notes = []
    if home.area not in KNOWN_AREAS:
        notes.append(f"'{home.area}' has too few listings in the data, so this estimate is rough.")
    elif home.neighbourhood and home.neighbourhood not in options["areas"].get(home.area, []):
        notes.append(f"'{home.neighbourhood}' has few listings, so the estimate leans on the wider {home.area} area.")

    X = to_frame(home)
    best = float(np.exp(POINT.predict(X))[0])
    q = sorted(float(np.exp(RANGES[k].predict(X))[0]) for k in ["q10", "q25", "q75", "q90"])
    q10, q25, q75, q90 = q

    result = Estimate(
        best_guess=round_naira(best),
        typical_low=round_naira(q25),
        typical_high=round_naira(q75),
        wide_low=round_naira(q10),
        wide_high=round_naira(q90),
        data_year=DATA_YEAR,
        reliability_note=" ".join(notes) or None,
    )
    if home.asking_rent:
        result.verdict, result.verdict_message = make_verdict(home.asking_rent, q10, q25, q75, q90)
    return result
