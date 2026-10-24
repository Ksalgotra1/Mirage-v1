"""Run the locked 10-model Layer 3a blind benchmark locally.

The suite creates one clean and two X-LSB x=8 samples for each pinned
TorchVision checkpoint.  Checkpoints are obtained through TorchVision's
hash-checked cache; generated files live only in a temporary directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_MODELS = (
    ("convnext_tiny", "IMAGENET1K_V1"),
    ("efficientnet_b2", "IMAGENET1K_V1"),
    ("mobilenet_v3_large", "IMAGENET1K_V2"),
    ("mnasnet1_0", "IMAGENET1K_V1"),
    ("regnet_x_400mf", "IMAGENET1K_V2"),
    ("regnet_y_400mf", "IMAGENET1K_V2"),
    ("shufflenet_v2_x0_5", "IMAGENET1K_V1"),
    ("densenet201", "IMAGENET1K_V1"),
    ("efficientnet_v2_s", "IMAGENET1K_V1"),
    ("alexnet", "IMAGENET1K_V1"),
)
PAYLOAD_SEEDS = (20260814, 20260815)
X_BITS = 8


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(row["label"] == "stego" and row["verdict"] for row in rows)
    fp = sum(row["label"] == "clean" and row["verdict"] for row in rows)
    tn = sum(row["label"] == "clean" and not row["verdict"] for row in rows)
    fn = sum(row["label"] == "stego" and not row["verdict"] for row in rows)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "specificity": specificity, "stego_recall": recall,
        "precision": precision, "f1": f1,
        "balanced_accuracy": (specificity + recall) / 2,
        "clean_specificity_gate_pass": specificity >= 0.85,
        "stego_recall_gate_pass": recall >= 0.75,
    }


def _load_state_dict(architecture: str, enum_name: str) -> dict:
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise RuntimeError("TorchVision is required; install requirements.txt") from exc
    try:
        weight_enum = models.get_model_weights(architecture)
        weights = weight_enum[enum_name]
    except (KeyError, ValueError, AttributeError) as exc:
        raise RuntimeError(f"Pinned TorchVision weight unavailable: {architecture}.{enum_name}") from exc
    # get_state_dict uses the normal Torch cache and checks the upstream hash.
    return weights.get_state_dict(progress=False, check_hash=True)


def run_benchmark(output_dir: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install requirements.txt") from exc

    from detector.layer3a_gf import Layer3aGFDetector
    from fixtures.generate_patch_dataset import inject_model_xlsb

    output_dir.mkdir(parents=True, exist_ok=True)
    detector = Layer3aGFDetector()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="project-climax-layer3a-") as temporary:
        temp_dir = Path(temporary)
        for architecture, enum_name in BENCHMARK_MODELS:
            print(f"[benchmark] loading {architecture}.{enum_name}", flush=True)
            state_dict = _load_state_dict(architecture, enum_name)
            parameter_count = sum(
                tensor.numel() for tensor in state_dict.values()
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
            )
            samples: list[tuple[str, int | None, dict]] = [("clean", None, state_dict)]
            samples.extend(
                ("stego", seed, inject_model_xlsb(state_dict, X_BITS, seed)) for seed in PAYLOAD_SEEDS
            )
            for label, seed, sample_state in samples:
                sample_path = temp_dir / f"{architecture}__{label}_{seed if seed is not None else 'base'}.pt"
                torch.save(sample_state, sample_path)
                result = detector.predict(str(sample_path))
                if result.get("coverage_note"):
                    raise RuntimeError(f"Layer 3a could not score {architecture}: {result['coverage_note']}")
                rows.append({
                    "architecture": architecture,
                    "weight_enum": enum_name,
                    "parameter_count": parameter_count,
                    "label": label,
                    "payload_seed": seed,
                    "x_bits": X_BITS if seed is not None else 0,
                    "raw_margin": result["distance"],
                    "calibrated_margin": result["calibrated_distance"],
                    "threshold": result["threshold"],
                    "score": result["score"],
                    "forensic_band": result["forensic_band"],
                    "verdict": bool(result["verdict"]),
                })

    summary = {
        "benchmark": "layer3a_10model_blind_v1",
        "sample_count": len(rows),
        "models": [name for name, _ in BENCHMARK_MODELS],
        "x_bits": X_BITS,
        "payload_seeds": list(PAYLOAD_SEEDS),
        "metrics": _metrics(rows),
        "results": rows,
    }
    csv_path = output_dir / "layer3a_10model_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "layer3a_10model_results.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"[benchmark] wrote {csv_path} and {json_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "output" / "benchmarks" / "layer3a_10model",
        help="Ignored directory for CSV and JSON output.",
    )
    args = parser.parse_args()
    try:
        run_benchmark(args.output_dir)
    except RuntimeError as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
