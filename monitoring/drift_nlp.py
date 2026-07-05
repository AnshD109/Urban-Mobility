"""
monitoring/drift_nlp.py
-----------------------
Evidently drift detection for the NLP triage service.

Usage:
    python monitoring/drift_nlp.py \
        --ref  data/raw/nlp/nlp_incidents.csv \
        --curr data/raw/nlp_live.csv \
        --out  monitoring/reports/nlp_drift.html
"""
import argparse
import pandas as pd
from pathlib import Path

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
    from evidently.metrics import ColumnDriftMetric
    EVIDENTLY_OK = True
except ImportError:
    EVIDENTLY_OK = False
    print("WARNING: evidently not installed.")


def run_drift(ref_path: str, curr_path: str, out_path: str):
    if not EVIDENTLY_OK:
        return

    ref  = pd.read_csv(ref_path)[["text", "label"]]
    curr = pd.read_csv(curr_path)[["text", "label"]]

    report = Report(metrics=[
        DataDriftPreset(columns=["label"]),
        ColumnDriftMetric(column_name="label"),
    ])
    report.run(reference_data=ref, current_data=curr)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(out_path)
    print(f"NLP drift report saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",  default="data/raw/nlp/nlp_incidents.csv")
    parser.add_argument("--curr", default="data/raw/nlp_live.csv")
    parser.add_argument("--out",  default="monitoring/reports/nlp_drift.html")
    args = parser.parse_args()
    run_drift(args.ref, args.curr, args.out)


if __name__ == "__main__":
    main()
