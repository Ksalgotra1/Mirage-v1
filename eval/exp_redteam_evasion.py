"""Experiment 5: Red-Team Evasion Testing
Tests whether an attacker can evade detection by modifying the embedding strategy.
Results labeled: SYNTHETIC-DATA RESULT
"""
from __future__ import annotations
import sys, json, tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_FIXTURE = "fixtures/benign_resnet18_real.pt"


def _score_file(path: str) -> dict:
    from detector.layer1_modelscan import Layer1ModelScan
    from detector.layer2_classifier import Layer2Classifier
    from detector.layer3a_gf import Layer3aGFDetector
    from detector.layer3b_weights import scan as l3b_scan
    from backend.fusion import FusionEngine

    l1 = Layer1ModelScan()
    l2 = Layer2Classifier()
    l3a = Layer3aGFDetector()
    fusion = FusionEngine()

    r1  = l1.scan(path)
    r2  = l2.predict(path)
    r3a = l3a.predict(path)
    r3b = l3b_scan(path)
    fused = fusion.fuse(".pt", r1, r2, r3a, r3b)
    return {
        "l3a_verdict": r3a.get("verdict"),
        "l3a_score":   round(float(r3a.get("score", 0)), 3),
        "l3b_verdict": r3b.get("verdict"),
        "fused_verdict": fused["verdict"],
        "fused_score":   fused["score"],
        "forensic_band": fused["forensic_band"],
    }


def main():
    import torch
    import numpy as np
    from fixtures.generate_patch_dataset import inject_model_xlsb

    print("=" * 65)
    print("EXPERIMENT 5: RED-TEAM EVASION TESTING")
    print("=" * 65)

    base_path = PROJECT_ROOT / BASE_FIXTURE
    state_dict = torch.load(str(base_path), weights_only=True, map_location="cpu")
    results = []

    with tempfile.TemporaryDirectory(prefix="mirage-redteam-") as tmpdir:
        tmp = Path(tmpdir)

        # Baseline: standard X-LSB x=8
        stego_base = inject_model_xlsb(state_dict, 8, 20260814)
        p = tmp / "baseline_x8.pt"; torch.save(stego_base, p)
        r = _score_file(str(p))
        results.append({"attack": "baseline_xlsb_x8", **r})
        print(f"  baseline_xlsb_x8:        score={r['fused_score']:.3f}  detected={r['fused_verdict']}")

        # Evasion 1: Low embedding rate (x=1 bit)
        stego_low = inject_model_xlsb(state_dict, 1, 20260814)
        p = tmp / "evade_xlsb_x1.pt"; torch.save(stego_low, p)
        r = _score_file(str(p))
        results.append({"attack": "evasion_xlsb_x1_lowrate", **r})
        print(f"  evasion_xlsb_x1:         score={r['fused_score']:.3f}  detected={r['fused_verdict']}")

        # Evasion 2: Stride-4 embedding (sparse injection pattern)
        import copy
        stego_stride = copy.deepcopy(state_dict)
        rng = np.random.default_rng(42)
        for key in list(stego_stride.keys()):
            tensor = stego_stride[key]
            if tensor.dtype == torch.float32 and tensor.numel() > 64:
                arr = tensor.detach().cpu().numpy().astype('<f4')
                raw = arr.tobytes()
                ba = bytearray(raw)
                indices = range(3, len(ba), 4 * 4)  # every 4th weight's LSB byte
                payload = rng.integers(0, 256, size=len(list(indices)), dtype=np.uint8)
                for i, idx in enumerate(indices):
                    if idx < len(ba):
                        ba[idx] = int(payload[i])
                new_arr = np.frombuffer(bytes(ba), dtype='<f4').reshape(arr.shape)
                stego_stride[key] = torch.from_numpy(new_arr.copy())
        p = tmp / "evade_stride4.pt"; torch.save(stego_stride, p)
        r = _score_file(str(p))
        results.append({"attack": "evasion_stride4_lsb1", **r})
        print(f"  evasion_stride4_lsb1:    score={r['fused_score']:.3f}  detected={r['fused_verdict']}")

        # Evasion 3: Zero-LSB payload (uniform, low entropy)
        stego_zero = copy.deepcopy(state_dict)
        for key in list(stego_zero.keys()):
            tensor = stego_zero[key]
            if tensor.dtype == torch.float32 and tensor.numel() > 64:
                arr = tensor.detach().cpu().numpy().astype('<f4')
                raw = arr.tobytes()
                ba = bytearray(raw)
                for i in range(3, len(ba), 4):  # zero out every LSB byte
                    ba[i] = 0
                new_arr = np.frombuffer(bytes(ba), dtype='<f4').reshape(arr.shape)
                stego_zero[key] = torch.from_numpy(new_arr.copy())
        p = tmp / "evade_zero_lsb.pt"; torch.save(stego_zero, p)
        r = _score_file(str(p))
        results.append({"attack": "evasion_zero_lsb_low_entropy", **r})
        print(f"  evasion_zero_lsb:        score={r['fused_score']:.3f}  detected={r['fused_verdict']}")

        # Aggressive: x=16 bits
        stego_high = inject_model_xlsb(state_dict, 16, 20260814)
        p = tmp / "evade_xlsb_x16.pt"; torch.save(stego_high, p)
        r = _score_file(str(p))
        results.append({"attack": "aggressive_xlsb_x16", **r})
        print(f"  aggressive_xlsb_x16:     score={r['fused_score']:.3f}  detected={r['fused_verdict']}")

    detected = sum(1 for r in results if r["fused_verdict"] == "MALICIOUS")
    total = len(results)
    print(f"\n  Detection rate: {detected}/{total} attacks detected ({detected/total:.0%})")

    out = PROJECT_ROOT / "output" / "benchmarks" / "exp5_redteam_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": "SYNTHETIC-DATA RESULT",
        "summary": {"detected": detected, "total": total, "detection_rate": round(detected/total, 3)},
        "results": results,
    }, indent=2))
    print(f"[red-team] Saved to {out}")


if __name__ == "__main__":
    main()
