import csv
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_CSV = os.path.join(PROJECT_ROOT, "eval", "results.csv")


def generate_results_table():
    if not os.path.exists(RESULTS_CSV):
        print(f"No results CSV found at {RESULTS_CSV}. Run `python eval/run_eval.py` first.")
        return

    with open(RESULTS_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("\n" + "=" * 85)
    print("  ENSEMBLE MODEL MALWARE DETECTOR — EVALUATION RESULTS TABLE")
    print("=" * 85)
    print(f"{'Filepath':<32} | {'Fmt':<6} | {'Attack Type':<20} | {'GT':<8} | {'Fused':<8} | {'Status'}")
    print("-" * 85)

    for r in rows:
        fp = r["filepath"]
        if len(fp) > 30:
            fp = "..." + fp[-27:]
        status = "✓ PASS" if r["correct"] == "True" else "✗ FAIL"
        print(f"{fp:<32} | {r['format']:<6} | {r['attack_type']:<20} | {r['ground_truth']:<8} | {r['fused_verdict']:<8} | {status}")

    print("-" * 85)

    # Format coverage matrix
    print("\nPER-FORMAT COVERAGE MATRIX:")
    print("  • .pt / .pkl     : Layer 1 (modelscan) ✓  |  Layer 2 (opcode RF) ✓  |  Layer 3a/3b ✓")
    print("  • .h5            : Layer 1 (modelscan) ✓  |  Layer 2 (pickle-only) ✗  |  Layer 3a/3b ✓")
    print("  • .safetensors   : Layer 1 (unsupported) ✗|  Layer 2 (pickle-only) ✗  |  Layer 3a/3b ✓")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    generate_results_table()
