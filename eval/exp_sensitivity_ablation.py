"""Experiment 3 & 4: Payload Sensitivity + Ablation Study
Tests detection across embedding rates (x=1..16 LSBs) and measures
per-layer contribution via leave-one-out ablation.
Results labeled: SYNTHETIC-DATA RESULT
"""
from __future__ import annotations
import sys, json, tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

X_BITS_LEVELS = [1, 2, 4, 8, 16]
PAYLOAD_SEEDS = [20260814, 20260815, 20260816]
BASE_FIXTURE = "fixtures/benign_resnet18_real.pt"


def run_payload_sensitivity():
    """Experiment 3: Detection rate vs LSB bit depth."""
    import torch
    from detector.layer3a_gf import Layer3aGFDetector
    from detector.layer3b_weights import scan as l3b_scan
    from fixtures.generate_patch_dataset import inject_model_xlsb

    l3a = Layer3aGFDetector()
    base_path = PROJECT_ROOT / BASE_FIXTURE
    state_dict = torch.load(str(base_path), weights_only=True, map_location="cpu")

    sensitivity = []
    with tempfile.TemporaryDirectory(prefix="mirage-sensitivity-") as tmpdir:
        tmp = Path(tmpdir)
        for x_bits in X_BITS_LEVELS:
            l3a_hits, l3b_hits, total = 0, 0, 0
            for seed in PAYLOAD_SEEDS:
                stego = inject_model_xlsb(state_dict, x_bits, seed)
                p = tmp / f"stego_x{x_bits}_s{seed}.pt"
                torch.save(stego, p)
                r3a = l3a.predict(str(p))
                r3b = l3b_scan(str(p))
                if r3a.get("verdict"):
                    l3a_hits += 1
                if r3b.get("verdict"):
                    l3b_hits += 1
                total += 1
            rec_3a = l3a_hits / total
            rec_3b = l3b_hits / total
            sensitivity.append({
                "x_bits": x_bits,
                "l3a_recall": round(rec_3a, 3),
                "l3b_recall": round(rec_3b, 3),
                "combined_recall": round(max(rec_3a, rec_3b), 3),
                "n_samples": total,
            })
            print(f"  x={x_bits:2d} bits | L3a recall={rec_3a:.0%}  L3b recall={rec_3b:.0%}")
    return sensitivity


def run_ablation(tmp_stego_path: str):
    """Experiment 4: Leave-one-out ablation showing each layer's contribution."""
    from detector.layer1_modelscan import Layer1ModelScan
    from detector.layer2_classifier import Layer2Classifier
    from detector.layer3a_gf import Layer3aGFDetector
    from detector.layer3b_weights import scan as l3b_scan
    from backend.fusion import FusionEngine

    l1 = Layer1ModelScan()
    l2 = Layer2Classifier()
    l3a = Layer3aGFDetector()
    fusion = FusionEngine()

    r1  = l1.scan(tmp_stego_path)
    r2  = l2.predict(tmp_stego_path)
    r3a = l3a.predict(tmp_stego_path)
    r3b = l3b_scan(tmp_stego_path)

    null_l1  = {"verdict": False, "probability": 0.0}
    null_l2  = {"verdict": False, "probability": 0.0}
    null_l3a = {"verdict": False, "score": 0.0, "forensic_band": "clean"}
    null_l3b = {"verdict": False, "score": 0.0}

    configs = {
        "L1_only":       fusion.fuse(".pt", r1,      null_l2, null_l3a, null_l3b),
        "L2_only":       fusion.fuse(".pt", null_l1, r2,      null_l3a, null_l3b),
        "L3a_only":      fusion.fuse(".pt", null_l1, null_l2, r3a,      null_l3b),
        "L3b_only":      fusion.fuse(".pt", null_l1, null_l2, null_l3a, r3b     ),
        "L1+L2":         fusion.fuse(".pt", r1,      r2,      null_l3a, null_l3b),
        "L3a+L3b":       fusion.fuse(".pt", null_l1, null_l2, r3a,      r3b     ),
        "Full_pipeline": fusion.fuse(".pt", r1,      r2,      r3a,      r3b     ),
    }

    ablation = []
    for name, fused in configs.items():
        ablation.append({
            "config": name,
            "verdict": fused["verdict"],
            "fused_score": fused["score"],
            "forensic_band": fused["forensic_band"],
        })
        print(f"  {name:<20}: score={fused['score']:.3f}  verdict={fused['verdict']}")
    return ablation


def main():
    import torch
    from fixtures.generate_patch_dataset import inject_model_xlsb

    print("=" * 60)
    print("EXPERIMENT 3: PAYLOAD SENSITIVITY (x-LSB bit depth)")
    print("=" * 60)
    sensitivity = run_payload_sensitivity()

    print("\n" + "=" * 60)
    print("EXPERIMENT 4: ABLATION STUDY (layer contributions)")
    print("=" * 60)
    base_path = PROJECT_ROOT / BASE_FIXTURE
    import torch
    state_dict = torch.load(str(base_path), weights_only=True, map_location="cpu")
    with tempfile.TemporaryDirectory(prefix="mirage-ablation-") as tmpdir:
        stego = inject_model_xlsb(state_dict, 8, 20260814)
        stego_path = Path(tmpdir) / "ablation_stego_x8.pt"
        torch.save(stego, stego_path)
        ablation = run_ablation(str(stego_path))

    out = PROJECT_ROOT / "output" / "benchmarks" / "exp3_exp4_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": "SYNTHETIC-DATA RESULT",
        "experiment_3_payload_sensitivity": sensitivity,
        "experiment_4_ablation": ablation,
    }, indent=2))
    print(f"\n[experiments] Saved to {out}")


if __name__ == "__main__":
    main()
