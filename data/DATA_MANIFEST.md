# Data Manifest

This file documents what's in `data/`, where it came from, and what to do
when moving from the small samples in this repo to the real full dataset.

---

## The sample-vs-full distinction (read this first)

Two files in `data/processed/` are **stand-ins for much larger real files**
that live on the project owner's machine and are too large to commit or
upload directly:

| File in this repo | Rows here | Real full file | Real row count |
|---|---|---|---|
| `trips_weather_zones_q1_2024.csv` | 5,000 | same filename, full Q1 2024 | ~9.3M |
| `delivery_orders_q1_2024.csv` | 200,000 | same filename, full Q1 2024 | ~9.3M |

**To run with real numbers:** replace these two files in `data/processed/`
with the full versions (same filename, same column schema). No notebook
code changes are needed — `01_EDA.ipynb` and `02_feature_engineering.ipynb`
both auto-detect row count at load time and print a `SAMPLE MODE` or
`FULL DATASET MODE` banner so it's always clear which one ran.

**Known limitation of the bundled `trips_weather_zones_q1_2024.csv` sample:**
it was taken from the head of the file rather than a random draw, so it only
spans ~2 hours of Dec 31 / Jan 1 with zero rain and zero snow. This means
Sections 3, 4, 7, 8, and 9 of `01_EDA.ipynb` (hourly volume, weather impact,
day-of-week, monthly trends) will show flat or `NaN` results on this sample
specifically — not a bug, just a property of a non-random 5k slice. The
notebook detects and flags this automatically. The `delivery_orders_q1_2024.csv`
sample, by contrast, *is* a proper random draw across the full Jan–Mar range
with real weather variation, so `02_feature_engineering.ipynb` produces
meaningful sample-based metrics already (RF MAE ≈ 5.4 min, R² ≈ 0.70).

---

## `data/raw/` — source data, never hand-edited, used as-is

```
data/raw/
├── maintenance/
│   ├── aps_failure_training_set.csv     ~60,000 trucks × 171 sensor features (real, UCI)
│   ├── aps_failure_test_set.csv         ~16,000 trucks, same schema (real, UCI)
│   └── aps_failure_description.txt      official dataset documentation
│
├── telematics/
│   └── telematics.csv                   300 synthetic vehicles (generated)
│
└── nlp/
    └── nlp_incidents.csv                3,000 synthetic incident texts (generated)
```

### A note on `nlp/nlp_incidents.csv`

None of the three real NLP datasets in the original raw upload fit the
incident/feedback triage task (see "Excluded datasets" below — wrong
language, wrong task framing). Rather than force-fit mismatched data,
`scripts/make_nlp_data.py` generates synthetic incident text with
deliberate imperfections — typos, ~22% mixed-signal reports (e.g. "late
AND damaged"), ~10% neutral filler, ~8% flipped labels — so the trained
classifier lands at a credible cross-validation F1 ≈ 0.92 instead of a
suspicious 1.00 from clean templates. Documented in
`notebooks/06_nlp_triage.ipynb`.

## `data/processed/` — pipeline inputs and outputs

```
data/processed/
├── trips_weather_zones_q1_2024.csv    INPUT  — sample here, replace with full file
├── delivery_orders_q1_2024.csv        INPUT  — sample here, replace with full file
├── train_features.csv                 OUTPUT — from 02_feature_engineering.ipynb
└── test_features.csv                  OUTPUT — from 02_feature_engineering.ipynb
```

The real-data chain, if you need to rebuild `trips_weather_zones_q1_2024.csv`
or `delivery_orders_q1_2024.csv` from scratch:

```
yellow_tripdata_2024-01/02/03.parquet ─┐
weather_q1_2024.csv ───────────────────┼──► trips_base → trips_weather → trips_weather_zones_q1_2024.csv
taxi_zone_lookup.csv ───────────────────┘                                          │
                                                                                     ▼
                                                                   scripts/simulate_delivery_orders.py
                                                                                     │
                                                                                     ▼
                                                                       delivery_orders_q1_2024.csv
```

`scripts/simulate_delivery_orders.py` already defaults to these exact
filenames — `python scripts/simulate_delivery_orders.py` with the full
`trips_weather_zones_q1_2024.csv` in place regenerates the full
`delivery_orders_q1_2024.csv` with no flags needed.

## `data/docs/` — reference PDFs (kept once, not duplicated)

- `2024_IDA_challenge_v2.pdf` — documentation for the **unused** alternative
  maintenance dataset (see "Excluded" below)
- `Scania_Component_X.pdf` — Scania component reference
- `Traffic-and-Accident-Data.pdf` — general traffic/accident reference

---

## Excluded datasets — and why

These existed in the original raw upload (`sxhb1213.zip`) but are **not**
part of this project. Documented here so the decision is traceable, not
silently lost.

| Dataset | What it actually is | Why excluded |
|---|---|---|
| `2024-34-3/` (IDA 2024 challenge) | Real vehicle time-to-event data: `vehicle_id` joins specs + operational readouts + repair labels (23,550 vehicles) | Genuinely good data, but it's a **survival-analysis** problem (time-to-failure), not binary classification. The proposal specifies IsolationForest + RandomForest classification, which the APS dataset already serves with ROC-AUC 0.988. Switching paradigms would have no proposal benefit. |
| `massive_transport_de/` | German voice-assistant intent dataset (805 rows) — commands like "show me the route" / "call a taxi" | Wrong task: intent classification for a voice assistant, not incident/feedback triage. Wrong language for an English-language service. |
| `transport_keywords/` | Korean civic-complaint keyword extraction (85,465 rows) | Wrong task (keyword extraction, not classification) and wrong language. |
| `transport_qa/` | Chinese traffic-law Q&A pairs for a chatbot (25,137 rows) | Wrong task entirely (Q&A, not triage classification) and wrong language. |
| `verkehrszellen.csv`, `Detailnetz-Strassenabschnitte.csv`, `berlin_traffic_counts.csv` | Berlin road-network geometry and traffic-zone reference data | No join key to trip-level data; would be decorative, not functional, in this pipeline. |

If a future iteration needs a Berlin-specific map view or a real-language
NLP triage dataset, these are the files to revisit — they are not
technically bad data, just off-task for this specific proposal.
