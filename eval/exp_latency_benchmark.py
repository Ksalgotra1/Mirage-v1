"""Experiment 6: Latency & Throughput Benchmarking
Measures per-layer and total pipeline latency on the existing fixture models.
Results are labeled: ACTUAL EXPERIMENTAL RESULT
"""
from __future__ import annotations
import sys, time, json, statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES = [
    ("fixtures/benign_resnet18_real.pt", "benign", ".pt"),
    ("fixtures/benign_cnn_base.safetensors", "benign", ".safetensors"),
]

REPS = 3  # repeat each measurement N times for stability


def time_fn(fn, *args, reps=REPS):
    times = []
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return result, times


def run_latency_benchmark():
    from detector.layer1_modelscan import Layer1ModelScan
    from detector.layer2_classifier import Layer2Classifier
    from detector.layer3a_gf import Layer3aGFDetector
    from detector.layer3b_weights import scan as l3b_scan
    from backend.fusion import FusionEngine

    l1 = Layer1ModelScan()
    l2 = Layer2Classifier()
    l3a = Layer3aGFDetector()
    fusion = FusionEngine()

    results = []

    for rel_path, label, ext in FIXTURES:
        abs_path = str(PROJECT_ROOT / rel_path)
        print(f"\n[latency] Testing {rel_path} ({label}, {ext})")

        # Layer 1
        r1, t1 = time_fn(l1.scan, abs_path)
        print(f"  L1: {statistics.mean(t1):.1f} ms (n={REPS})")

        # Layer 2
        r2, t2 = time_fn(l2.predict, abs_path)
        print(f"  L2: {statistics.mean(t2):.1f} ms (n={REPS})")

        # Layer 3a
        r3a, t3a = time_fn(l3a.predict, abs_path)
        print(f"  L3a: {statistics.mean(t3a):.1f} ms (n={REPS})")

        # Layer 3b
        r3b, t3b = time_fn(l3b_scan, abs_path)
        print(f"  L3b: {statistics.mean(t3b):.1f} ms (n={REPS})")

        # Fusion
        _r3a = r3a if r3a else {"verdict": False, "score": 0.0, "forensic_band": "clean"}
        _r3b = r3b if r3b else {"verdict": False, "score": 0.0}
        def do_fusion():
            return fusion.fuse(ext, r1, r2, _r3a, _r3b)
        _, t_fuse = time_fn(do_fusion)
        print(f"  Fusion: {statistics.mean(t_fuse):.2f} ms (n={REPS})")

        total_ms = statistics.mean(t1) + statistics.mean(t2) + statistics.mean(t3a) + statistics.mean(t3b) + statistics.mean(t_fuse)
        print(f"  TOTAL: {total_ms:.1f} ms")

        results.append({
            "file": rel_path, "label": label, "ext": ext,
            "l1_ms": round(statistics.mean(t1), 1),
            "l2_ms": round(statistics.mean(t2), 1),
            "l3a_ms": round(statistics.mean(t3a), 1),
            "l3b_ms": round(statistics.mean(t3b), 1),
            "fusion_ms": round(statistics.mean(t_fuse), 3),
            "total_ms": round(total_ms, 1),
        })

    out = PROJECT_ROOT / "output" / "benchmarks" / "latency_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": "ACTUAL EXPERIMENTAL RESULT", "results": results}, indent=2))
    print(f"\n[latency] Saved to {out}")
    return results


if __name__ == "__main__":
    run_latency_benchmark()
