"""
simulate_delivery_orders.py
---------------------------
Convert cleaned NYC taxi trip data into simulated delivery orders by
applying data-driven multipliers validated during EDA.

⚠️ CALIBRATION NOTE: when this script's output was validated against the
real full Q1 2024 dataset (9.2M rows), the actual rain impact on trip
duration measured only +0.4%, not the +8% assumed below. The rain
multiplier was set before the real EDA findings were available and
hasn't been recalibrated yet — consider lowering RAIN_MULT to ~1.005 if
this script is re-run, or re-deriving it directly from
`taxi.groupby('is_rainy')['trip_duration_min'].mean()` on real data (see
`notebooks/01_EDA.ipynb` Section 7). SNOW_MULT is unverifiable either way
since the source `weather_q1_2024.csv` has zero non-null snow readings.

Multipliers applied
-------------------
- Rush-hour delay    : +15 % on base travel if hour in {8,9,17,18,19}
- Weather (rain)     : +8 % if prcp > 0   ⚠️ see calibration note above
- Weather (snow)     : +18 % if snow > 0  ⚠️ unverifiable, no real snow data
- High wind          : +5 % if wspd > 20
- Pickup delay       : sampled from Gamma(shape=2, scale=3), capped at 20 min

Usage:
    python scripts/simulate_delivery_orders.py
    python scripts/simulate_delivery_orders.py \
        --input  data/processed/trips_weather_zones_q1_2024.csv \
        --output data/processed/delivery_orders_q1_2024.csv \
        --sample 500000
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


RUSH_HOURS    = {8, 9, 17, 18, 19}
RAIN_MULT     = 1.08   # ⚠️ see note below — real Q1 2024 data shows only +0.4%
SNOW_MULT     = 1.18   # ⚠️ unverifiable — source weather data has no snow readings
WIND_MULT     = 1.05
WIND_THRESH   = 20.0   # m/s
MAX_PICKUP_DELAY = 20.0


def simulate(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = len(df)

    # ── Base travel time = trip_duration_min ─────────────────────────────
    base = df["trip_duration_min"].values.copy().astype(float)

    # ── Temporal multiplier ───────────────────────────────────────────────
    hour         = pd.to_datetime(df["pickup_hour"]).dt.hour.values
    rush_mask    = np.isin(hour, list(RUSH_HOURS))
    base         = np.where(rush_mask, base * 1.15, base)

    # ── Weather multipliers ───────────────────────────────────────────────
    rain_mask    = df["prcp"].fillna(0).values > 0
    snow_mask    = df["snow"].fillna(0).values > 0
    wind_mask    = df["wspd"].fillna(0).values > WIND_THRESH

    base         = np.where(rain_mask, base * RAIN_MULT, base)
    base         = np.where(snow_mask, base * SNOW_MULT, base)
    base         = np.where(wind_mask, base * WIND_MULT, base)

    # ── Pickup delay (Gamma) ──────────────────────────────────────────────
    pickup_delay = rng.gamma(shape=2.0, scale=3.0, size=n).clip(0, MAX_PICKUP_DELAY)

    # ── Final ETA ─────────────────────────────────────────────────────────
    eta = base + pickup_delay

    out = pd.DataFrame({
        "order_id":          np.arange(1, n + 1),
        "order_time":        df["pickup_hour"].values,
        "pickup_zone":       df["pickup_zone"].values,
        "dropoff_zone":      df["dropoff_zone"].values,
        "pickup_borough":    df["pickup_borough"].values,
        "dropoff_borough":   df["dropoff_borough"].values,
        "temp":              df["temp"].values,
        "prcp":              df["prcp"].fillna(0).values,
        "wspd":              df["wspd"].fillna(0).values,
        "trip_distance":     df["trip_distance"].values,
        "base_travel_min":   base.round(4),
        "pickup_delay_min":  pickup_delay.round(4),
        "eta_min":           eta.round(4),
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/processed/trips_weather_zones_q1_2024.csv")
    parser.add_argument("--output", default="data/processed/delivery_orders_q1_2024.csv")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample size (default: all rows)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    # Explicit dtypes keep memory in check on the full ~9M-row file —
    # pandas' default type inference on a 1.5GB+ CSV can exceed 3-4x the
    # file size in memory and get OOM-killed on constrained machines.
    dtypes = {
        "trip_distance": "float32",
        "PULocationID": "int16",
        "DOLocationID": "int16",
        "trip_duration_min": "float32",
        "temp": "float32",
        "prcp": "float32",
        "wspd": "float32",
        "snow": "float32",
        "pickup_borough": "category",
        "pickup_zone": "category",
        "dropoff_borough": "category",
        "dropoff_zone": "category",
    }
    df = pd.read_csv(args.input, parse_dates=["pickup_hour", "weather_hour"], dtype=dtypes)
    print(f"  Loaded {len(df):,} rows")

    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=args.seed)
        print(f"  Sampled {len(df):,} rows")

    print("Simulating delivery orders ...")
    orders = simulate(df, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(out_path, index=False)

    print(f"Saved {len(orders):,} orders → {out_path}")
    print(f"ETA stats:  mean={orders['eta_min'].mean():.1f}  "
          f"p50={orders['eta_min'].median():.1f}  "
          f"p95={orders['eta_min'].quantile(0.95):.1f}")


if __name__ == "__main__":
    main()
