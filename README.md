# Lagos Fair Rent

**Is this rent fair?** A machine-learning model that estimates the typical annual rent for a home in Lagos, Nigeria, and says whether an asking rent is typical, high or low for similar homes.

- **Live demo:** https://lagos-fair-rent.onrender.com
- **API documentation:** https://lagos-fair-rent.onrender.com/docs
- **Built by:** [Nwankwonta Chiemena](https://nathansparks.dev)

> **Prices are 2022 levels.** The model was trained on listings from 2022. Lagos rents have risen sharply since, so use it to compare homes against each other, not as a guide to today's prices. The free server sleeps when unused, so the first visit can take up to a minute.

---

## Why

Renters in Lagos usually have no way to tell whether an asking rent is reasonable. Prices are set by agents, listings are inconsistent, and a fake or inflated listing looks the same as a fair one. This project gives renters a reference point: what do homes like this one usually cost?

It was built as a feature for [Ilé](https://ile.nathansparks.dev), a rental platform for Nigeria that is in development.

## Results

All models were tested on the same **9,856 listings they never saw during training**.

| Model | Average error | Median error | Within ±20% of actual |
|---|---|---|---|
| Median rent by bedrooms | ₦2,707,373 | 44.0% | 25.8% |
| Median rent by area and bedrooms | ₦2,094,711 | 28.6% | 37.3% |
| Linear model on log rent | ₦1,970,272 | 26.0% | 39.6% |
| Gradient boosting | ₦1,872,334 | 23.2% | 43.2% |
| **Gradient boosting + title keywords (final)** | **₦1,836,002** | **23.0%** | **44.2%** |

**What 23% median error means:** for a flat that rents for ₦2,500,000 a year, half of the model's guesses are within about ₦575,000.

### The ranges are calibrated

A single number overstates how precise the model is, so the main output is a **range**. Four extra models learn the 10th, 25th, 75th and 90th percentile of rent for similar homes. A range is only useful if it is honest, so the test set checks how often the actual rent falls inside it:

| Range | Should contain the actual rent | Actually did |
|---|---|---|
| 25th to 75th percentile | 50% of the time | **48.7%** |
| 10th to 90th percentile | 80% of the time | **78.6%** |

The verdict compares an asking rent with these ranges: below the 10th percentile is *well below typical*, 25th to 75th is *typical*, above the 90th is *well above typical*. Very cheap listings come with a warning to inspect before paying, because unusually low prices are a common sign of rental scams.

### What drives rent

Measured by permutation importance (how much worse the model gets when one input is shuffled):

1. **Neighbourhood** matters most. "Lekki" alone covers very different markets, from Lekki Phase 1 to Ajah.
2. **Bedrooms** come a close second.
3. Property type, area and toilets follow, well behind.

Bathrooms score near zero. That doesn't mean bathrooms don't affect rent: they move almost exactly with bedrooms and toilets, so they add no new information once those are known.

## Data

- **Source:** "Cost of housing in Lagos" on [openAFRICA](https://open.africa/dataset/cost-of-housing-in-lagos), published by the Council of Governors, Kenya, under a **Creative Commons Attribution** licence. It is a copy of [Lagos, Nigeria rent prices 2022](https://www.kaggle.com/datasets/eyimofeapinnick/lagos-nigeria-rent-prices-2022) on Kaggle by eyimofeapinnick.
- **Size:** 53,070 listings, each with title, description, price, bedrooms, bathrooms, toilets, area, neighbourhood, and serviced, newly built and furnished flags.

The raw data is not stored in this repository. `clean_data.py` expects it at `data/lagos_rent_2022.csv` (see *Run it yourself*).

## How the data was cleaned

Cleaning took more care than modelling. Each rule was checked against what it removed, and two of the first versions were wrong and had to be fixed.

| Step | Listings | Why |
|---|---|---|
| Raw | 53,070 | |
| Keep prices that convert to a yearly amount | 52,921 | Monthly prices × 12. **Per-day prices dropped:** they are short-lets, a different market. Per-square-metre prices are land or commercial. |
| Recover missing bedroom counts | 52,479 | 5,514 listings said "0 beds". **3,939 were recovered from the title**, e.g. "4 Bedroom Semi Detached Duplex". **1,133 were genuine self-contains**, kept as their own type. A *mini flat* counts as 1 bedroom. |
| Keep homes only | 51,459 | Land and commercial removed, judged from the description only. An early version searched titles for "land" and wrongly matched "Victoria **Is*land***". |
| Remove impossible rents | 51,161 | Below ₦100,000 or above ₦500,000,000 a year: typos, or sale prices posted as rent. |
| Keep 1–10 bedrooms (0 only for self-contains) | 51,153 | |
| Remove reposts | **49,279** | A repost has the same title **and** details. An early version matched on details alone, which wrongly removed 18,756 listings: identical flats in the same estate are different homes. |

## How the model works

- **Target:** the logarithm of annual rent. Rent grows by multiples (an extra bedroom adds roughly a percentage, not a fixed amount), and the log turns that into something a model handles well. For example, log(₦2,500,000) ≈ 14.73, and exp(14.73) ≈ ₦2,500,000.
- **Location:** neighbourhood and area are target-encoded, meaning each name is replaced by the typical rent there, calculated with cross-fitting so a listing never sees its own price.
- **Title keywords:** 20 yes/no features such as *BQ* (boys' quarters), *penthouse*, *pool*, *estate* and *shared*. They helped only slightly (23.2% to 23.0%). Words like "luxury" and "new" appear in so many listings that they carry little information.
- **Model:** scikit-learn `HistGradientBoostingRegressor` with early stopping, plus four quantile versions for the range.

## Limitations

- **2022 prices.** The most important limitation. The model is not ready to tell real renters what is fair today until it is retrained on current listings.
- **No size, condition or exact location.** Two "3-bedroom flats in Lekki" can differ hugely, and the data can't tell them apart. This, not the choice of model, sets the ~23% error ceiling. More tuning gained only decimals, so tuning stopped there.
- **Asking prices, not agreed rents.** Listings are what agents typed, including errors that cleaning can't fully remove.
- **Rare places.** Neighbourhoods with fewer than 10 listings are not offered in the demo, and the API flags estimates for little-known places as rough.

## Project structure

```
clean_data.py        Step 3: cleans the raw listings
train_baseline.py    Step 4: simple baselines to beat
train_model.py       Step 5: final model and range models
make_options.py      Lists areas and neighbourhoods for the API and demo
app.py               FastAPI service: demo page, /estimate, /options, /health
static/index.html    The demo page
models/              Trained models (rent_model.joblib) and options.json
data/                Saved scores (raw data is not committed)
```

## Run it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Get the data (openAFRICA copy, CC Attribution)
mkdir -p data
curl -L -o data/lagos_rent_2022.csv "https://open.africa/dataset/4cf60242-e9b5-4bac-8f4b-a0c9f57641e8/resource/97af0d3c-0ca1-472f-bbe7-e3811aa70427/download/cfafrica-_-openafrica-_-reform-_-current-requests-a1.csv"

python clean_data.py
python train_baseline.py
python train_model.py
python make_options.py

uvicorn app:app --reload   # then open http://127.0.0.1:8000
```

### API example

```bash
curl -X POST https://lagos-fair-rent.onrender.com/estimate \
  -H "Content-Type: application/json" \
  -d '{"bedrooms": 3, "property_type": "flat", "area": "Lekki", "neighbourhood": "Agungi", "asking_rent": 3500000}'
```

Returns the best estimate, the typical range (25th–75th percentile), the wide range (10th–90th), and a verdict when an asking rent is given. Every response includes `data_year`, so 2022 prices can never be mistaken for current ones.

## Next

Retrain on current Lagos listings, then connect the "Is this rent fair?" check to Ilé's listings.

## Credits

Data: "Cost of housing in Lagos", openAFRICA (Council of Governors, Kenya), licensed CC Attribution; originally "Lagos, Nigeria rent prices 2022" by eyimofeapinnick on Kaggle.
