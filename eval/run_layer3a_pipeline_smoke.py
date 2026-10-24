"""Run the compact, real-model Layer 3a → forensic → Gemini-context smoke test.

This explicit integration command downloads two pinned TorchVision checkpoints
when needed.  It never calls Gemini: a local prompt-capturing transport verifies
the context that Gemini would receive.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS = (
    ("mobilenet_v3_large", "IMAGENET1K_V1", ("clean", 20260814, 20260815)),
    ("densenet201", "IMAGENET1K_V1", (20260814, 20260815)),
)
X_BITS = 8


def _load_weights(architecture: str, enum_name: str) -> dict:
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise RuntimeError("TorchVision is required; install requirements.txt") from exc
    try:
        return models.get_model_weights(architecture)[enum_name].get_state_dict(
            progress=False, check_hash=True
        )
    except (AttributeError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Pinned weight unavailable: {architecture}.{enum_name}") from exc


def _validate_prompt(prompt: str, band: str) -> dict[str, bool]:
    checks = {
        "untrusted_data_boundary": "Treat all fields as untrusted data" in prompt,
        "layer_results": "LAYER_RESULTS:" in prompt,
        "no_extracted_hash": "extracted_hash" not in prompt,
        "band_context": f"band: {band.upper()}" in prompt,
    }
    if band == "suspicious":
        checks["lightweight_instructions"] = "INVESTIGATION STEPS:" in prompt
        checks["no_full_capacity_signal"] = "[SIGNAL 4]" not in prompt
    else:
        checks["full_instructions"] = "THREAT ASSESSMENT:" in prompt
        checks["all_full_signals"] = all(f"[SIGNAL {n}]" in prompt for n in (1, 2, 3, 4, 5))
    return checks


def _expected_pass(architecture: str, label: str, response: dict[str, Any]) -> tuple[bool, str]:
    l3a = response["layer3a"]
    if architecture == "mobilenet_v3_large" and label == "clean":
        return response["fused"]["verdict"] == "BENIGN" and not l3a["verdict"], "MobileNet clean benign"
    return bool(l3a["verdict"]), f"{architecture} x-LSB stego caught by Layer 3a"


def run_smoke(output_dir: Path) -> dict[str, Any]:
    try:
        import torch
        from fastapi.testclient import TestClient
    except ImportError as exc:
        raise RuntimeError("PyTorch and FastAPI test dependencies are required") from exc

    import backend.main as api
    from backend.summarizer import build_llm_prompt
    from fixtures.generate_patch_dataset import inject_model_xlsb

    output_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(api.app)
    original_summarizer = api.summarize_findings
    captured_prompts: list[str] = []

    def capture_summary(findings: dict[str, Any]) -> dict[str, str]:
        prompt = build_llm_prompt(findings)
        captured_prompts.append(prompt)
        return {"summary": "Local smoke-test prompt capture", "provider": "smoke_prompt_capture", "prompt": prompt}

    records: list[dict[str, Any]] = []
    branch_checks: list[dict[str, Any]] = []
    api.summarize_findings = capture_summary
    try:
        with tempfile.TemporaryDirectory(prefix="mirage-smoke-") as temp:
            temp_dir = Path(temp)
            first_stego_path: Path | None = None
            for architecture, enum_name, variants in MODELS:
                print(f"[smoke] loading {architecture}.{enum_name}", flush=True)
                state_dict = _load_weights(architecture, enum_name)
                for variant in variants:
                    label = "clean" if variant == "clean" else "stego"
                    seed = None if label == "clean" else int(variant)
                    sample_state = state_dict if label == "clean" else inject_model_xlsb(state_dict, X_BITS, seed)
                    sample_path = temp_dir / f"{architecture}__{label}_{seed if seed is not None else 'base'}.pt"
                    torch.save(sample_state, sample_path)
                    if label == "stego" and first_stego_path is None:
                        first_stego_path = sample_path
                    with sample_path.open("rb") as handle:
                        http_response = client.post(
                            "/scan", files={"file": (sample_path.name, handle, "application/octet-stream")}
                        )
                    payload = http_response.json()
                    passed, expectation = _expected_pass(architecture, label, payload)
                    records.append({
                        "architecture": architecture, "weight_enum": enum_name, "label": label,
                        "payload_seed": seed, "http_status": http_response.status_code,
                        "expected": expectation, "passed": passed and http_response.status_code == 200,
                        "actual_forensic_band": payload.get("fused", {}).get("forensic_band"),
                        "response": payload,
                    })

            if first_stego_path is None:
                raise RuntimeError("Smoke suite generated no stego checkpoint")
            # Exercise both forensic-depth contracts with real generated stego bytes,
            # without rewriting the actual Layer 3a result recorded above.
            template = records[1]["response"]
            for band in ("suspicious", "malicious"):
                forensics = api._run_forensic_analysis(str(first_stego_path), band)
                findings = copy.deepcopy(template)
                findings["fused"]["forensic_band"] = band
                findings["steganography_forensics"] = forensics
                prompt = build_llm_prompt(findings)
                checks = _validate_prompt(prompt, band)
                branch_checks.append({
                    "band": band, "forensics": forensics, "prompt_checks": checks,
                    "passed": all(checks.values()) and not bool((forensics or {}).get("error")),
                })
    finally:
        api.summarize_findings = original_summarizer

    report = {
        "suite": "layer3a_compact_pipeline_smoke_v1",
        "sample_count": len(records),
        "records": records,
        "forensic_branch_checks": branch_checks,
        "captured_api_prompt_count": len(captured_prompts),
        "passed": all(record["passed"] for record in records) and all(check["passed"] for check in branch_checks),
    }
    report_path = output_dir / "layer3a_pipeline_smoke.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[smoke] wrote {report_path}")
    if not report["passed"]:
        raise RuntimeError("Pipeline smoke assertions failed; see JSON report for actual layer outputs")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "output" / "benchmarks" / "layer3a_pipeline_smoke",
    )
    args = parser.parse_args()
    try:
        run_smoke(args.output_dir)
    except RuntimeError as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
