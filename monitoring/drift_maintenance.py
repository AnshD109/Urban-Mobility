"""
monitoring/drift_maintenance.py
--------------------------------
Evidently drift detection for the maintenance risk service.

Usage:
    python monitoring/drift_maintenance.py \
        --ref  data/raw/telematics/telematics.csv \
        --curr data/raw/telematics_live.csv \
        --out  monitoring/reports/maintenance_drift.html
"""
import argparse
import pandas as pd
from pathlib import Path

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    EVIDENTLY_OK = True
except ImportError:
    EVIDENTLY_OK = False
    print("WARNING: evidently not installed.")

FEATURES = [
    "odometer_km", "avg_speed_kmh", "max_speed_kmh",
    "engine_temp_mean_c", "engine_temp_max_c",
    "battery_voltage_mean", "battery_voltage_min",
    "vibration_mean_ms2", "vibration_max_ms2",
    "brake_pressure_mean", "hard_brakes_count",
    "idle_hours", "trips_count",
]


def run_drift(ref_path: str, curr_path: str, out_path: str):
    if not EVIDENTLY_OK:
        return

    ref  = pd.read_csv(ref_path)
    curr = pd.read_csv(curr_path)

    cols = [c for c in FEATURES if c in ref.columns]
    ref  = ref[cols + ["failure_label"]].rename(columns={"failure_label": "target"})
    curr = curr[cols + ["failure_label"]].rename(columns={"failure_label": "target"})

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(out_path)
    print(f"Maintenance drift report saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",  default="data/raw/telematics/telematics.csv")
    parser.add_argument("--curr", default="data/raw/telematics_live.csv")
    parser.add_argument("--out",  default="monitoring/reports/maintenance_drift.html")
    args = parser.parse_args()
    run_drift(args.ref, args.curr, args.out)


if __name__ == "__main__":
    main()
