"""Train and Calibrate the Wolpert Stacking Meta-Fusion Engine for MIRAGE.

Learns empirical logistic regression weights across all four threat layers:
- Layer 1: AST Dangerous Callable Bouncer (modelscan)
- Layer 2: Statistical Opcode Classifier (XGBoost)
- Layer 3a: GF-Fourpart Steganalysis (OSLEmbeddingNet v5)
- Layer 3b: Weight-Distribution Spectral Anomaly Detector
Combining with format indicators (.pt, .safetensors, .h5, .pkl).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

FEATURE_NAMES = [
    "L1_Verdict",
    "L2_Gated_Prob",
    "L3a_Margin_Delta",
    "L3a_Conviction",
    "L3b_Score",
    "L1_L2_Synergy",
    "Fmt_Pickle",
    "Fmt_Tensor",
]


def generate_meta_training_data(n_samples: int = 3000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate balanced multi-layer operational scenarios spanning all formats."""
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 10

    records: list[np.ndarray] = []
    labels: list[int] = []

    def make_record(l1=0.0, l2=0.0, l3a_delta=-1.0, l3a_v=0.0, l3b=0.0, fmt_pickle=0.0, fmt_tensor=1.0):
        l2_gated = float(l2) * float(fmt_pickle)
        l1_l2_joint = float(l1) * l2_gated
        return np.array([
            float(l1),
            float(l2_gated),
            float(l3a_delta),
            float(l3a_v),
            float(l3b),
            float(l1_l2_joint),
            float(fmt_pickle),
            float(fmt_tensor),
        ], dtype=np.float32)

    # 1. Clean PyTorch/Pickle models (y=0)
    for _ in range(n_per_class * 2):
        l2 = rng.normal(0.07, 0.03)
        l3a_d = rng.normal(-1.0, 0.06)
        l3b = rng.uniform(0.0, 0.04)
        records.append(make_record(l1=0.0, l2=np.clip(l2, 0.0, 0.18), l3a_delta=l3a_d, l3a_v=0.0, l3b=l3b, fmt_pickle=1.0, fmt_tensor=1.0))
        labels.append(0)

    # 2. Clean SafeTensors models (y=0)
    for _ in range(n_per_class * 2):
        l3a_d = rng.normal(-1.0, 0.06)
        l3b = rng.uniform(0.0, 0.04)
        records.append(make_record(l1=0.0, l2=0.0, l3a_delta=l3a_d, l3a_v=0.0, l3b=l3b, fmt_pickle=0.0, fmt_tensor=1.0))
        labels.append(0)

    # 3. Clean HDF5 / format models (y=0)
    for _ in range(n_per_class):
        l3a_d = rng.normal(-1.0, 0.06)
        l3b = rng.uniform(0.0, 0.04)
        records.append(make_record(l1=0.0, l2=0.0, l3a_delta=l3a_d, l3a_v=0.0, l3b=l3b, fmt_pickle=0.0, fmt_tensor=1.0))
        labels.append(0)

    # 4. Benign Complex Reducers (y=0, L2 false positive edge case: L1=0, L2~0.92, L3a=-1.0)
    for _ in range(n_per_class * 2):
        l2 = rng.uniform(0.85, 0.95)
        l3a_d = rng.normal(-1.0, 0.05)
        records.append(make_record(l1=0.0, l2=l2, l3a_delta=l3a_d, l3a_v=0.0, l3b=0.0, fmt_pickle=1.0, fmt_tensor=rng.choice([0.0, 1.0])))
        labels.append(0)

    # 5. Direct Opcode Exploits in Pickle (y=1: L1=1, L2~0.95)
    for _ in range(n_per_class * 2):
        l2 = rng.uniform(0.88, 0.99)
        records.append(make_record(l1=1.0, l2=l2, l3a_delta=rng.choice([-1.0, 0.0]), l3a_v=0.0, l3b=0.0, fmt_pickle=1.0, fmt_tensor=rng.choice([0.0, 1.0])))
        labels.append(1)

    # 6. Obfuscated Opcode Exploits in Pickle (y=1: L1=0, L2~0.98)
    for _ in range(n_per_class):
        l2 = rng.uniform(0.92, 0.99)
        records.append(make_record(l1=0.0, l2=l2, l3a_delta=-1.0, l3a_v=0.0, l3b=0.0, fmt_pickle=1.0, fmt_tensor=rng.choice([0.0, 1.0])))
        labels.append(1)

    # 7. LSB Steganography in SafeTensors (y=1)
    for _ in range(n_per_class * 2):
        l3a_d = rng.uniform(2.0, 5.0)
        records.append(make_record(l1=0.0, l2=0.0, l3a_delta=l3a_d, l3a_v=1.0, l3b=rng.uniform(0.0, 0.05), fmt_pickle=0.0, fmt_tensor=1.0))
        labels.append(1)

    # 8. LSB Steganography in PyTorch (y=1)
    for _ in range(n_per_class * 2):
        l2 = rng.uniform(0.05, 0.22)
        l3a_d = rng.uniform(2.0, 5.0)
        records.append(make_record(l1=0.0, l2=l2, l3a_delta=l3a_d, l3a_v=1.0, l3b=rng.uniform(0.0, 0.05), fmt_pickle=1.0, fmt_tensor=1.0))
        labels.append(1)

    # 9. Weight Distribution Backdoors (y=1, Layer 3b trigger)
    for _ in range(n_per_class):
        l3b = rng.uniform(0.80, 1.0)
        records.append(make_record(l1=0.0, l2=rng.uniform(0.0, 0.15), l3a_delta=-1.0, l3a_v=0.0, l3b=l3b, fmt_pickle=rng.choice([0.0, 1.0]), fmt_tensor=1.0))
        labels.append(1)

    # 10. Composite Multi-Layer Threat (y=1)
    for _ in range(n_per_class):
        l2 = rng.uniform(0.90, 0.99)
        l3a_d = rng.uniform(2.5, 5.5)
        l3b = rng.uniform(0.75, 1.0)
        records.append(make_record(l1=1.0, l2=l2, l3a_delta=l3a_d, l3a_v=1.0, l3b=l3b, fmt_pickle=1.0, fmt_tensor=1.0))
        labels.append(1)

    X = np.stack(records, axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y


def train_meta_fusion() -> dict[str, any]:
    """Train the logistic meta-learner, validate calibration, and save weights."""
    print("Generating balanced multi-layer training scenarios...")
    X_train, y_train = generate_meta_training_data(n_samples=5000, seed=42)
    X_val, y_val = generate_meta_training_data(n_samples=1500, seed=1337)

    # L2 regularized logistic regression (C=1.0)
    clf = LogisticRegression(
        C=1.0,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Validate
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.50).astype(int)
    auc = roc_auc_score(y_val, val_probs)
    acc = accuracy_score(y_val, val_preds)
    brier = brier_score_loss(y_val, val_probs)
    loss = log_loss(y_val, val_probs)

    print(f"Validation Results: AUC={auc:.4f} | Accuracy={acc*100:.2f}% | Brier={brier:.4f} | LogLoss={loss:.4f}")

    intercept = float(clf.intercept_[0])
    coef_dict = {name: float(w) for name, w in zip(FEATURE_NAMES, clf.coef_[0])}

    print("\nLearned Meta-Fusion Parameters:")
    print(f"  Intercept: {intercept:+.4f}")
    for k, v in coef_dict.items():
        print(f"  {k:<20}: {v:+.4f}")

    weights_data = {
        "intercept": intercept,
        "coefficients": coef_dict,
        "feature_names": FEATURE_NAMES,
        "calibration_meta": {
            "val_auc": round(float(auc), 4),
            "val_accuracy": round(float(acc), 4),
            "val_brier": round(float(brier), 4),
            "val_log_loss": round(float(loss), 4),
            "version": "v5_calibrated_multi_layer",
        },
    }

    out_path = Path(__file__).resolve().parent / "models" / "meta_fusion_weights.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2)

    print(f"\n[OK] Serialized calibrated weights to {out_path}")
    return weights_data


if __name__ == "__main__":
    train_meta_fusion()
