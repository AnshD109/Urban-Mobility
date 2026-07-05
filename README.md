# Urban Mobility ML Suite — ETA, Pricing, Maintenance & NLP

**Student:** Ansh Dankhara  
**Supervisor:** Professor Humera Noor Minhas  
**Program:** Data Science  

---

## Project Overview

A full ML pipeline built on real NYC Yellow Taxi data that validates urban mobility assumptions and deploys three production-ready microservices:

| Service | Port | Endpoint | Description |
|---|---|---|---|
| ETA & Pricing | 8000 | `/quote` | Quantile ETA prediction + surge pricing |
| Maintenance Risk | 8001 | `/maintenance/risk` | Anomaly-based vehicle risk scoring |
| NLP Triage | 8002 | `/nlp/predict` | Incident & feedback classification |

---

## Repository Structure

```
urban-mobility-eta-berlin/
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis (Sections 1–12)
│   ├── 02_feature_engineering.ipynb  # Feature pipeline for delivery orders
│   ├── 03_eta_model.ipynb            # ETA regression + quantile models
│   ├── 04_pricing_model.ipynb        # Surge pricing logic
│   ├── 05_maintenance_model.ipynb    # IsolationForest + calibrated risk
│   └── 06_nlp_triage.ipynb           # TF-IDF + LogisticRegression triage
├── scripts/
│   ├── make_telematics_data.py       # Generate synthetic telematics data
│   ├── make_nlp_data.py              # Generate synthetic NLP incident data
│   └── simulate_delivery_orders.py   # Delivery order simulation from taxi data
├── services/
│   ├── eta_service/
│   │   ├── main.py                   # FastAPI app — /quote endpoint
│   │   ├── model.py                  # Model loading + inference
│   │   └── schemas.py                # Pydantic request/response models
│   ├── maintenance_service/
│   │   ├── main.py                   # FastAPI app — /maintenance/risk
│   │   ├── model.py
│   │   └── schemas.py
│   └── nlp_service/
│       ├── main.py                   # FastAPI app — /nlp/predict
│       ├── model.py
│       └── schemas.py
├── data/
│   ├── raw/
│   │   ├── maintenance/               # Real Scania APS failure data (train + test)
│   │   ├── telematics/                # Synthetic vehicle sensor data
│   │   └── nlp/                       # Synthetic incident/feedback text
│   ├── docs/                          # Reference PDFs (dataset documentation)
│   ├── processed/                     # Pipeline inputs/outputs — see DATA_MANIFEST.md
│   └── DATA_MANIFEST.md               # What's included, what's excluded, and why
├── models/                           # Serialised .joblib model artefacts
├── monitoring/
│   ├── drift_eta.py                  # Evidently drift report — ETA service
│   ├── drift_maintenance.py          # Evidently drift report — Maintenance
│   └── drift_nlp.py                  # Evidently drift report — NLP
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── tests/
│   ├── test_eta_service.py
│   ├── test_maintenance_service.py
│   └── test_nlp_service.py
├── .github/workflows/
│   └── ci.yml                        # GitHub Actions CI pipeline
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/urban-mobility-eta-berlin.git
cd urban-mobility-eta-berlin
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data — sample included, swap in the full files for real numbers
`data/processed/trips_weather_zones_q1_2024.csv` and `delivery_orders_q1_2024.csv`
ship as small samples so the notebooks run out of the box. Both notebooks
auto-detect row count and print a `SAMPLE MODE` / `FULL DATASET MODE` banner,
so it's always clear which one ran. To get real numbers, drop the full files
in at the **same path, same filename** — no code changes needed:
- `data/processed/trips_weather_zones_q1_2024.csv` — full NYC TLC trip + weather + zone data (~9.3M rows)
- `data/processed/delivery_orders_q1_2024.csv` — full simulated delivery orders (~9.3M rows)

See `data/DATA_MANIFEST.md` for the full source chain and known limitations
of the bundled samples.

### 3. Run notebooks in order
```
01_EDA → 02_feature_engineering → 03_eta_model → 04_pricing_model
       → 05_maintenance_model  → 06_nlp_triage
```

### 4. Generate synthetic data
```bash
python scripts/make_telematics_data.py
python scripts/make_nlp_data.py
```

### 5. Start all services
```bash
docker-compose up --build
```

### 6. Launch dashboard
```bash
streamlit run dashboard/app.py
```

---

## API Reference

### ETA & Pricing — `POST /quote`
```json
{
  "pickup_zone": "East Village",
  "dropoff_zone": "Clinton East",
  "pickup_borough": "Manhattan",
  "dropoff_borough": "Manhattan",
  "trip_distance": 2.7,
  "order_hour": 9,
  "day_of_week": 2,
  "month": 3,
  "temp": 5.0,
  "prcp": 0.0,
  "wspd": 10.0
}
```
Response includes `eta_p50_min`, `eta_p10_min`, `eta_p90_min`, `base_price`, `surge_multiplier`, `final_price`.

### Maintenance Risk — `POST /maintenance/risk`
```json
{
  "vehicle_id": "V001",
  "speed_kmh": 72.0,
  "engine_temp_c": 95.0,
  "battery_voltage": 12.1,
  "vibration_ms2": 0.8,
  "brake_pressure_bar": 8.5,
  "odometer_km": 125000
}
```

### NLP Triage — `POST /nlp/predict`
```json
{
  "text": "Delivery was 45 minutes late due to traffic on main road"
}
```

---

## Dataset Sources

| Dataset | Source |
|---|---|
| NYC Yellow Taxi Q1 2024 | [TLC Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Weather data | [Meteostat](https://meteostat.net) |
| Scania APS Failure | [UCI / IDA 2016](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) |
| Synthetic telematics | Generated via `scripts/make_telematics_data.py` |
| Synthetic NLP incidents | Generated via `scripts/make_nlp_data.py` |

---

## Sprint Progress

| Sprint | Dates | Status |
|---|---|---|
| 1 — Data & EDA | Apr 7–20 | ✅ Complete |
| 2 — Simulation & Features | Apr 21 – May 4 | ✅ Complete |
| 3 — ETA & Pricing | May 5–18 | ✅ Complete |
| 4 — Maintenance & NLP | May 19 – Jun 1 | ✅ Complete |
| 5 — Productionisation | Jun 2–15 | ✅ Complete |
| 6 — Final Report | Jun 16–22 | 🔄 In progress |
