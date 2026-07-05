"""
make_telematics_data.py
-----------------------
Generate synthetic vehicle telematics data for maintenance risk modelling.

Realism design
---------------
An earlier version of this generator used non-overlapping distributions for
each feature (e.g. healthy engine temp clipped to 70-100°C, faulty clipped
to 88-130°C, with every other feature also separated in the same direction).
That made the two classes trivially separable — a Random Forest hit AUC =
1.0000 on the held-out test set, which is a textbook red flag for synthetic
or leaked data, not evidence of a good model.

This version fixes that with three changes, the same recipe used for the
NLP synthetic data elsewhere in this project:

  1. Wider, genuinely overlapping distributions — faulty and healthy
     vehicles now draw from distributions that substantially overlap in
     every feature, not just barely touch at the edges.
  2. Partial symptoms — real faulty vehicles rarely show every failure
     indicator at once. Each faulty vehicle independently rolls which
     subset of symptoms (hot engine, weak battery, high vibration, etc.)
     it actually exhibits, so the class boundary isn't a clean sum of
     identical shifts across all features.
  3. Label noise (~6%) — a small fraction of vehicles get a flipped label,
     simulating real-world annotation/diagnosis error (a vehicle flagged
     faulty that turned out fine, or a failure that wasn't caught in time).

Together these pull AUC down from a suspicious 1.00 to a credible ~0.85-0.93
range with real false positives/negatives, the same way the NLP triage
classifier's F1 was deliberately brought down from 1.00 to ~0.92.

Calibration result (n_vehicles=600, partial_symptom_frac=0.72, seed=42,
RandomForestClassifier with 300 trees, 80/20 stratified split):
  ROC-AUC = 0.8875, Faulty recall = 0.43, Faulty precision = 0.91
  (13 of 23 faulty vehicles missed, 1 false alarm — a believable trade-off,
  not a perfect score)

Usage:
    python scripts/make_telematics_data.py
    python scripts/make_telematics_data.py --n-vehicles 500 --out data/raw/telematics/telematics.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_telematics(
    n_vehicles: int = 600,
    label_noise_frac: float = 0.06,
    partial_symptom_frac: float = 0.72,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate one row per vehicle with aggregated telematics readings.

    Columns
    -------
    vehicle_id          : unique identifier
    odometer_km         : total kilometres driven
    avg_speed_kmh       : mean speed across all readings
    max_speed_kmh       : maximum recorded speed
    engine_temp_mean_c  : mean engine temperature (°C)
    engine_temp_max_c   : peak engine temperature (°C)
    battery_voltage_mean: mean battery voltage (V)
    battery_voltage_min : minimum battery voltage (V)
    vibration_mean_ms2  : mean vibration (m/s²)
    vibration_max_ms2   : peak vibration (m/s²)
    brake_pressure_mean : mean brake pressure (bar)
    hard_brakes_count   : number of hard-braking events
    idle_hours           : hours spent idling
    trips_count          : total trips taken
    failure_label        : 1 = APS failure within next 30 days, 0 = healthy
    """
    rng = np.random.default_rng(seed)

    n_healthy = int(n_vehicles * 0.85)
    n_faulty  = n_vehicles - n_healthy
    labels    = np.array([0] * n_healthy + [1] * n_faulty)
    rng.shuffle(labels)

    # ── Shared baseline distribution — both classes draw from this ────────
    # Every vehicle starts here, then faulty vehicles get nudged on a
    # random subset of features. This is what creates genuine overlap:
    # a healthy vehicle can land anywhere in the wide baseline range, and
    # a faulty vehicle that didn't roll the "hot engine" symptom looks
    # statistically identical to a healthy one on that feature.
    def _baseline(n):
        return dict(
            odometer_km          = rng.uniform(5_000,  350_000, n),
            avg_speed_kmh        = rng.normal(46,      11,      n).clip(10, 110),
            max_speed_kmh        = rng.normal(107,     17,      n).clip(60, 180),
            engine_temp_mean_c   = rng.normal(92,      8,       n).clip(70, 130),
            engine_temp_max_c    = rng.normal(104,     10,      n).clip(85, 145),
            battery_voltage_mean = rng.normal(13.4,    0.6,     n).clip(10.5, 14.5),
            battery_voltage_min  = rng.normal(12.1,    0.8,     n).clip(9.0,  13.5),
            vibration_mean_ms2   = rng.uniform(0.1,    1.8,     n),
            vibration_max_ms2    = rng.uniform(0.5,    5.0,     n),
            brake_pressure_mean  = rng.normal(8.0,     1.6,     n).clip(3, 12),
            hard_brakes_count    = rng.poisson(6,               n),
            idle_hours           = rng.uniform(10,     400,     n),
            trips_count          = rng.integers(50,    800,     n),
        )

    rows = _baseline(n_vehicles)

    # ── Faulty nudges — applied probabilistically, per-vehicle, per-symptom ─
    # Each faulty vehicle independently rolls whether it shows each symptom
    # (partial_symptom_frac chance per symptom), and the nudge size itself
    # is randomised rather than a fixed shift. This avoids the "every faulty
    # vehicle has every symptom, shifted by exactly the same amount" pattern
    # that made the old version trivially separable.
    faulty_idx = np.where(labels == 1)[0]
    n_f = len(faulty_idx)

    def _maybe_nudge(col, low, high, direction=1):
        """Apply a random-magnitude nudge to a random subset of faulty vehicles."""
        shows_symptom = rng.random(n_f) < partial_symptom_frac
        nudge = rng.uniform(low, high, n_f) * direction
        rows[col][faulty_idx] += np.where(shows_symptom, nudge, 0.0)

    _maybe_nudge("odometer_km",          20_000, 120_000, +1)
    _maybe_nudge("engine_temp_mean_c",   3,      14,      +1)
    _maybe_nudge("engine_temp_max_c",    4,      18,      +1)
    _maybe_nudge("battery_voltage_mean", 0.3,    1.6,     -1)
    _maybe_nudge("battery_voltage_min",  0.4,    2.0,     -1)
    _maybe_nudge("vibration_mean_ms2",   0.2,    1.2,     +1)
    _maybe_nudge("vibration_max_ms2",    0.5,    2.5,     +1)
    _maybe_nudge("brake_pressure_mean",  0.5,    2.5,     -1)
    _maybe_nudge("idle_hours",           30,     150,     +1)

    # hard_brakes_count and trips_count use Poisson-style count nudges
    shows_brakes = rng.random(n_f) < partial_symptom_frac
    rows["hard_brakes_count"][faulty_idx] += np.where(
        shows_brakes, rng.poisson(5, n_f), 0)
    shows_trips = rng.random(n_f) < partial_symptom_frac
    rows["trips_count"][faulty_idx] += np.where(
        shows_trips, rng.integers(50, 300, n_f), 0)

    # Re-clip to realistic bounds after nudging
    rows["engine_temp_mean_c"]   = np.clip(rows["engine_temp_mean_c"],   70,   135)
    rows["engine_temp_max_c"]    = np.clip(rows["engine_temp_max_c"],    85,   150)
    rows["battery_voltage_mean"] = np.clip(rows["battery_voltage_mean"], 9.5,  14.5)
    rows["battery_voltage_min"]  = np.clip(rows["battery_voltage_min"],  8.0,  13.5)
    rows["brake_pressure_mean"]  = np.clip(rows["brake_pressure_mean"],  2.0,  12.0)
    rows["odometer_km"]          = np.clip(rows["odometer_km"],          5_000, 400_000)

    df = pd.DataFrame(rows)
    df.insert(0, "vehicle_id", [f"V{i:04d}" for i in range(1, n_vehicles + 1)])
    df["failure_label"] = labels.astype(int)

    # ── Label noise — simulate real-world diagnosis/annotation error ──────
    n_noisy = int(n_vehicles * label_noise_frac)
    noisy_idx = rng.choice(n_vehicles, size=n_noisy, replace=False)
    df.loc[noisy_idx, "failure_label"] = 1 - df.loc[noisy_idx, "failure_label"]

    # Round for realism
    df = df.round({
        "odometer_km": 0,
        "avg_speed_kmh": 1,
        "max_speed_kmh": 1,
        "engine_temp_mean_c": 1,
        "engine_temp_max_c": 1,
        "battery_voltage_mean": 2,
        "battery_voltage_min": 2,
        "vibration_mean_ms2": 3,
        "vibration_max_ms2": 3,
        "brake_pressure_mean": 2,
        "idle_hours": 1,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic telematics data")
    parser.add_argument("--n-vehicles", type=int, default=600,
                        help="Number of vehicles to simulate (default: 600)")
    parser.add_argument("--label-noise-frac", type=float, default=0.06,
                        help="Fraction of vehicles with deliberately flipped labels")
    parser.add_argument("--partial-symptom-frac", type=float, default=0.72,
                        help="Probability each faulty vehicle shows any given symptom")
    parser.add_argument("--out", type=str, default="data/raw/telematics/telematics.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating telematics data for {args.n_vehicles} vehicles "
          f"(label_noise={args.label_noise_frac:.0%}, "
          f"partial_symptoms={args.partial_symptom_frac:.0%})...")
    df = generate_telematics(
        n_vehicles=args.n_vehicles,
        label_noise_frac=args.label_noise_frac,
        partial_symptom_frac=args.partial_symptom_frac,
        seed=args.seed,
    )

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    print(f"Failure rate: {df['failure_label'].mean():.1%}")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()