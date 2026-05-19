"""
read_metrics.py  —  Print a results summary from a completed experiment run

Usage (from the terramind root):
    pixi run python experiments/read_metrics.py output/sen1floods11_base_20260407_2038

The argument is the run folder (the timestamped folder under output/).
The script reads metrics.csv from that folder and prints:
  - Best validation mIoU and which epoch it came from
  - Final epoch metrics
  - Test set results
  - Comparison to the published TerraMind paper number

Can be run any time — including mid-training to see progress so far.
The metrics summary is also printed automatically at the end of the job log
by run_experiment.aqua, so you usually don't need to run this separately.
"""

import csv
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Get the run directory from the command line
# ─────────────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: pixi run python experiments/read_metrics.py <run_dir>")
    print("Example: pixi run python experiments/read_metrics.py output/sen1floods11_base_20260407_2038")
    sys.exit(1)

RUN_DIR = sys.argv[1].rstrip("/")
METRICS_CSV = os.path.join(RUN_DIR, "metrics.csv")

# Published mIoU from the TerraMind paper, Table 2
# (Sen1Floods11, base model, S1+S2 hand-labelled split)
# Update this if you find a more precise figure in the paper.
PUBLISHED_MIOU = 0.804


# ─────────────────────────────────────────────────────────────────────────────
# Read the CSV
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(METRICS_CSV):
    print(f"\nMetrics file not found: {METRICS_CSV}")
    print("Has the training job produced any output yet?")
    sys.exit(1)

all_rows = []
with open(METRICS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        parsed = {}
        for key, value in row.items():
            if value == "":
                parsed[key] = None
            else:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        all_rows.append(parsed)

# Validation rows are written once per epoch during training.
# Test rows are written once by terratorch test.
val_rows  = [r for r in all_rows if r.get("val/mIoU")  is not None]
test_rows = [r for r in all_rows if r.get("test/mIoU") is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print(f"  TerraMind results — {os.path.basename(RUN_DIR)}")
print("=" * 60)

if val_rows:
    best = max(val_rows, key=lambda r: r["val/mIoU"])
    print(f"\n  Training complete:   {len(val_rows)} epochs")
    print(f"  Best val/mIoU:       {best['val/mIoU']:.4f}  (epoch {int(best['epoch'])})")
    print(f"  Best val/loss:       {best['val/loss']:.4f}")

    last = val_rows[-1]
    print(f"\n  Final epoch ({int(last['epoch'])}):")
    for key in ["val/loss", "val/mIoU", "val/IoU_Flood", "val/IoU_Others", "val/F1_Score"]:
        if last.get(key) is not None:
            print(f"    {key:<22s}  {last[key]:.4f}")
else:
    print("\n  No training epochs recorded yet.")

print()
if test_rows:
    t = test_rows[-1]
    print("  Test set results:")
    for key in sorted(t.keys()):
        if key.startswith("test/") and t[key] is not None:
            print(f"    {key:<22s}  {t[key]:.4f}")

    test_miou = t.get("test/mIoU")
    if test_miou is not None:
        diff = test_miou - PUBLISHED_MIOU
        if   abs(diff) < 0.01: verdict = "✓  within 1% of published — reproduction successful"
        elif diff > 0:          verdict = "▲  above published result"
        else:                   verdict = "▼  below published — check config matches paper exactly"

        print(f"\n  Published mIoU:      {PUBLISHED_MIOU:.4f}  (TerraMind paper, Table 2)")
        print(f"  Your test mIoU:      {test_miou:.4f}")
        print(f"  Difference:          {diff:+.4f}  {verdict}")
else:
    print("  No test results yet (terratorch test has not run or metrics.csv")
    print("  has not been updated). Check job status with: qstat -u $USER")

print()
print(f"  Full CSV: {METRICS_CSV}")
print("=" * 60)
print()
