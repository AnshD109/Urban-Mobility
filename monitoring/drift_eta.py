"""
monitoring/drift_eta.py
-----------------------
Evidently drift detection for the ETA service.

Usage:
    python monitoring/drift_eta.py \
        --ref  data/processed/train_features.csv \
        --curr data/processed/test_features.csv  \
        --out  monitoring/reports/eta_drift.html
"""
import argparse
import pandas as pd
from pathlib import Path

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, RegressionPreset
    from evidently.metrics import ColumnDriftMetric
    EVIDENTLY_OK = True
except ImportError:
    EVIDENTLY_OK = False
    print("WARNING: evidently not installed. pip install evidently==0.4.15")


NUMERIC_FEATURES = [
    "trip_distance", "temp", "prcp", "wspd",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_rainy", "is_windy",
]


def run_drift(ref_path: str, curr_path: str, out_path: str):
    if not EVIDENTLY_OK:
        print("Evidently not available. Exiting.")
        return

    ref  = pd.read_csv(ref_path,  parse_dates=["order_time"])
    curr = pd.read_csv(curr_path, parse_dates=["order_time"])

    cols = [c for c in NUMERIC_FEATURES if c in ref.columns]
    ref  = ref[cols + ["eta_min"]].rename(columns={"eta_min": "target"})
    curr = curr[cols + ["eta_min"]].rename(columns={"eta_min": "target"})

    report = Report(metrics=[
        DataDriftPreset(),
        ColumnDriftMetric(column_name="target"),
    ])
    report.run(reference_data=ref, current_data=curr)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(out_path)
    print(f"ETA drift report saved → {out_path}")

    # Print summary
    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    print(f"Dataset drift detected: {drift_detected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",  default="data/processed/train_features.csv")
    parser.add_argument("--curr", default="data/processed/test_features.csv")
    parser.add_argument("--out",  default="monitoring/reports/eta_drift.html")
    args = parser.parse_args()
    run_drift(args.ref, args.curr, args.out)


if __name__ == "__main__":
    main()
