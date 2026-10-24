import csv
import json
import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detector.layer1_modelscan import Layer1ModelScan
from detector.layer2_classifier import Layer2Classifier
from backend.fusion import FusionEngine
from eval.baseline_entropy import ByteEntropyBaseline

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "fixtures", "manifest.json")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "eval", "results.csv")


def run_evaluation(manifest_path=MANIFEST_PATH, output_csv=OUTPUT_CSV_PATH):
    if not os.path.exists(manifest_path):
        print(f"Manifest missing at {manifest_path}. Running fixture generator...")
        from fixtures.generate_fixtures import generate_all_fixtures
        generate_all_fixtures()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    l1_scanner = Layer1ModelScan()
    l2_classifier = Layer2Classifier()
    fusion_engine = FusionEngine()
    entropy_baseline = ByteEntropyBaseline()

    # Import Person B detectors if available
    try:
        from detector.layer3a_gf import Layer3aGFDetector
        l3a_detector = Layer3aGFDetector()
    except ImportError:
        l3a_detector = None

    try:
        from detector.layer3b_wd import Layer3bWDDetector
        l3b_detector = Layer3bWDDetector()
    except ImportError:
        l3b_detector = None

    results = []
    correct_count = 0

    fieldnames = [
        "filepath", "format", "attack_type", "embedding_rate",
        "ground_truth", "l1_verdict", "l2_prob", "l2_verdict",
        "l3a_distance", "l3a_verdict", "l3b_deviation", "l3b_verdict",
        "baseline_entropy", "baseline_verdict", "fused_score", "fused_verdict", "correct"
    ]

    for entry in manifest:
        rel_path = entry.get("filepath", "")
        abs_path = os.path.join(PROJECT_ROOT, rel_path) if not os.path.isabs(rel_path) else rel_path
        fmt = entry.get("format", "")
        attack_type = entry.get("attack_type", "none")
        embed_rate = entry.get("embedding_rate", 0.0)
        ground_truth = entry.get("ground_truth_label", "BENIGN").upper()

        if not os.path.exists(abs_path):
            continue

        l1_res = l1_scanner.scan(abs_path)
        l2_res = l2_classifier.predict(abs_path)

        l3a_res = l3a_detector.predict(abs_path) if l3a_detector else {"verdict": False, "distance": 0.0}
        l3b_res = l3b_detector.predict(abs_path) if l3b_detector else {"verdict": False, "deviation": 0.0}

        entropy_res = entropy_baseline.predict(abs_path)
        fused = fusion_engine.fuse(fmt, l1_res, l2_res, l3a_res, l3b_res)

        fused_verdict = fused["verdict"]
        is_correct = (fused_verdict == ground_truth)
        if is_correct:
            correct_count += 1

        row = {
            "filepath": rel_path,
            "format": fmt,
            "attack_type": attack_type,
            "embedding_rate": embed_rate,
            "ground_truth": ground_truth,
            "l1_verdict": l1_res.get("verdict", False),
            "l2_prob": l2_res.get("probability", 0.0),
            "l2_verdict": l2_res.get("verdict", False),
            "l3a_distance": l3a_res.get("distance", 0.0),
            "l3a_verdict": l3a_res.get("verdict", False),
            "l3b_deviation": l3b_res.get("deviation", 0.0),
            "l3b_verdict": l3b_res.get("verdict", False),
            "baseline_entropy": entropy_res.get("entropy", 0.0),
            "baseline_verdict": entropy_res.get("verdict", False),
            "fused_score": fused["score"],
            "fused_verdict": fused_verdict,
            "correct": is_correct,
        }
        results.append(row)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    acc = (correct_count / total * 100) if total > 0 else 0.0

    print(f"Evaluation complete. Results written to {output_csv}")
    print(f"Total Fixtures Evaluated: {total}")
    print(f"Ensemble Accuracy: {acc:.2f}% ({correct_count}/{total})")

    return results


if __name__ == "__main__":
    run_evaluation()
