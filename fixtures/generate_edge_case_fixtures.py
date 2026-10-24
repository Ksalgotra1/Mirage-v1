"""Generate realistic edge-case test fixtures for Layer 2 (XGBoost) classifier.

Generates:
  1. Adversarially padded exploit (XGBoost False Negative due to frequency dilution).
  2. Benign complex custom reducer (XGBoost False Positive due to structural syntax).
"""
from __future__ import annotations

import os
import pickle
import functools
import operator
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent


class DirectSystemPayload:
    """Malicious payload that invokes os.system."""
    def __reduce__(self):
        import os
        return (os.system, ("echo SENTINEL_MALICIOUS_PADDING_ATTACK",))


class BenignFunctionalReducer:
    """Benign data structure using functools and operator for custom reduction."""
    def __init__(self, val: int):
        self.val = val

    def __reduce__(self):
        return (functools.partial, (operator.add, self.val))


def generate_padded_exploit(output_path: Path | str | None = None) -> Path:
    """Generate an exploit padded with 15,000 benign dictionary entries.
    
    Dilutes malicious unigram/bigram frequency so XGBoost probability drops < 0.10 (FN),
    while Layer 1 AST bouncer catches the dangerous callable symbol.
    """
    if output_path is None:
        output_path = FIXTURES_DIR / "adversarial_padded_exploit.pkl"
    else:
        output_path = Path(output_path)

    large_dict = {f"weight_tensor_channel_{i}": float(i * 1.5) for i in range(15000)}
    data = {
        "model_config": {"version": 1, "arch": "dilated_resnet"},
        "payload": DirectSystemPayload(),
        "weights": large_dict,
    }
    raw_bytes = pickle.dumps(data, protocol=4)
    output_path.write_bytes(raw_bytes)
    return output_path


def generate_benign_complex_reducer(output_path: Path | str | None = None) -> Path:
    """Generate a benign object using protocol 0 and standard library functional reducers.
    
    Contains elevated GLOBAL and REDUCE opcodes without any dangerous imports,
    triggering an XGBoost false alarm (FP) while Layer 1 passes it cleanly.
    """
    if output_path is None:
        output_path = FIXTURES_DIR / "benign_custom_reducer.pkl"
    else:
        output_path = Path(output_path)

    data = [BenignFunctionalReducer(i) for i in range(20)]
    raw_bytes = pickle.dumps(data, protocol=0)
    output_path.write_bytes(raw_bytes)
    return output_path


def main():
    p1 = generate_padded_exploit()
    p2 = generate_benign_complex_reducer()
    print(f"[fixtures] Generated padded exploit: {p1} ({p1.stat().st_size} bytes)")
    print(f"[fixtures] Generated benign reducer: {p2} ({p2.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
