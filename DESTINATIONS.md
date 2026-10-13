# Morocco and selected global destinations

The destination explorer is the new project entry point. It accepts **any Moroccan city** through a free-text city field, with popular-city shortcuts. Global destinations: Barcelona, Madrid, Paris, London, Rome, Istanbul, Dubai, New York, Bangkok and Tokyo. Country codes and normalized city names define destination identity; Fez/Fès and Tanger/Tangier aliases are handled.

## Current price coverage

**Only Barcelona has a trained price model today.** Its existing 225 observed quotes are explicitly labeled Spain / Barcelona / EUR. No prices have been invented or copied to Morocco. Adding destinations to the interface does not create price observations. Other destinations display “Price data needed” and cannot produce estimates until populated and trained.

Public data checked:

- [MA_Open_Datasets / Booking_ma / Hotels](https://github.com/OumaimaHourrane/MA_Open_Datasets/tree/main/Booking_ma/Hotels): the actual Hotels.csv columns are title, location, distance, description, verdict, rating and experience. There is **no price column**, despite the directory listing on SMIA mentioning prices. It is not used for price training or redistributed.
- [Morocco tourism open data](https://www.data.gov.ma/data/dataset/?groups=tourisme): hotel directories, capacity and overnight-stay aggregates do not supply property-level room-price labels for this model.
- [Original Barcelona source](https://doi.org/10.5281/zenodo.17560799): retained as the reproducible historical example under CC BY-NC-SA 4.0. Its license still applies to derived models and browser model exports.

This update provides the ingestion, evaluation, inference and interface needed to extend geographic coverage. Nationwide trained predictions still require suitable public hotel-price observations. Data availability may differ by city. There is no claim of nationwide hotel inventory coverage.

## Reproduce and add observations

From the repository root, install requirements, then:

```sh
python -m src.multicity --seed-barcelona
python -m pytest -q
```

Open `dashboard/destinations.html` directly, or serve the repository with `python -m http.server 8000`. It works offline; the JavaScript model file is generated automatically. The original dashboard remains at `dashboard/index.html`, with the original notebooks, figures and research metrics unchanged.

To add real observations, combine them with `data/processed/destination_quotes.csv` in a CSV using the same headers. Each row must describe **one quoted room per night**, in the explicit quoted currency. For multi-night quotes divide the total by nights; for multiple rooms obtain the per-room quote, rather than training on the total booking price. Include source URL and license for every record. Do not guess a quote's dates, occupancy, room count or currency. Resolve repeated captures of an identical configuration upstream.

```sh
python -m src.multicity --input data/processed/combined_destination_quotes.csv
python -m src.predict --input examples/destination_hotel.csv --output predictions.csv
```

Required columns: `country`, `city`, `currency`, `property_id`, `checkin`, `checkout`, `rooms`, `children`, `child_ages`, `adults`, `distance_center_km`, `review_count`, `amenity_mentions_count`. Training also needs `price_per_night`, `source_url`, `source_license`.

Dates use YYYY-MM-DD. Country uses the configured two-letter code; currency is an explicit three-letter code (e.g. MAD or EUR). Child ages are comma-separated whole years; empty for zero children. Review counts are counts, not scores. Amenity mentions use the nine categories defined in the original workflow. Any currency conversion must be documented upstream with the rate and rate date; no implicit EUR-to-MAD conversion occurs.

## Evaluation and inference

One random forest per destination, currency, exact stay dates, room count and child configuration. Minimum 30 distinct properties per configuration. A deterministic property-level 80/20 split keeps all adult configurations of a hotel together. Models fit log(1 + price); outputs invert to the observed currency. Held-out MAE and training-median baseline MAE are published per model. The evaluated estimator is exported without refitting on held-out hotels.

Browser inference walks the same trees as Python, including float32 input conversion. Estimates are restricted to observed adult counts and numerical feature ranges. No model for a requested city/configuration means an explicit error in batch inference and an unavailable state in the dashboard. One observed date cannot identify seasonality or justify estimates for future dates.

Outputs: `artifacts/multicity.joblib`, `reports/destination_coverage.json`, `dashboard/destinations-data.js`. These are independent of the Barcelona research model and its metrics. Models are intentionally small and use four common features so observations from new sources can be normalized without fabricating absent facilities or ratings. This is an initial research baseline, not evidence of accuracy in unobserved markets.
