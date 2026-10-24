import json
import os
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detector.layer2_classifier import OpcodeFeatureExtractor

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "fixtures", "manifest.json")


def run_cross_validated_comparison():
    print("=" * 85)
    print("  5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK: XGBOOST vs. RANDOM FOREST")
    print("=" * 85)

    if not os.path.exists(MANIFEST_PATH):
        from fixtures.generate_fixtures import generate_large_fixture_dataset
        generate_large_fixture_dataset()

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    extractor = OpcodeFeatureExtractor()
    feature_names = extractor.get_feature_names()

    X = []
    y = []

    for entry in manifest:
        fmt = entry.get("format", "").lower()
        if fmt not in {"pkl", "pt"}:
            continue
        rel_path = entry.get("filepath", "")
        abs_path = os.path.join(PROJECT_ROOT, rel_path) if not os.path.isabs(rel_path) else rel_path
        if not os.path.exists(abs_path):
            continue

        feats = extractor.extract_features(abs_path)
        gt = 1 if entry.get("ground_truth_label", "").upper() == "MALICIOUS" else 0

        X.append(feats)
        y.append(gt)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"Dataset Size: {len(X)} samples ({np.sum(y == 1)} Malicious, {np.sum(y == 0)} Benign)\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_accs, xgb_accs = [], []
    rf_aucs, xgb_aucs = [], []
    rf_losses, xgb_losses = [], []
    rf_precisions, xgb_precisions = [], []
    rf_recalls, xgb_recalls = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train RF
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf.fit(X_train, y_train)
        rf_p = rf.predict_proba(X_val)[:, 1]
        rf_pred = (rf_p >= 0.5).astype(int)

        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_p = xgb_model.predict_proba(X_val)[:, 1]
        xgb_pred = (xgb_p >= 0.5).astype(int)

        # Metrics
        rf_accs.append(accuracy_score(y_val, rf_pred))
        xgb_accs.append(accuracy_score(y_val, xgb_pred))

        rf_aucs.append(roc_auc_score(y_val, rf_p))
        xgb_aucs.append(roc_auc_score(y_val, xgb_p))

        rf_losses.append(log_loss(y_val, rf_p, labels=[0, 1]))
        xgb_losses.append(log_loss(y_val, xgb_p, labels=[0, 1]))

        rf_precisions.append(precision_score(y_val, rf_pred, zero_division=0))
        xgb_precisions.append(precision_score(y_val, xgb_pred, zero_division=0))

        rf_recalls.append(recall_score(y_val, rf_pred, zero_division=0))
        xgb_recalls.append(recall_score(y_val, xgb_pred, zero_division=0))

    print("-" * 85)
    print(f"{'Metric (5-Fold CV Avg)':<30} | {'XGBoost (Primary)':<25} | {'Random Forest'}")
    print("-" * 85)
    print(f"{'Mean Accuracy':<30} | {np.mean(xgb_accs)*100:<25.2f}% | {np.mean(rf_accs)*100:.2f}%")
    print(f"{'Mean ROC-AUC Score':<30} | {np.mean(xgb_aucs):<25.4f}  | {np.mean(rf_aucs):.4f}")
    print(f"{'Mean Log Loss':<30} | {np.mean(xgb_losses):<25.4f}  | {np.mean(rf_losses):.4f}")
    print(f"{'Mean Precision':<30} | {np.mean(xgb_precisions)*100:<25.2f}% | {np.mean(rf_precisions)*100:.2f}%")
    print(f"{'Mean Recall':<30} | {np.mean(xgb_recalls)*100:<25.2f}% | {np.mean(rf_recalls)*100:.2f}%")
    print("-" * 85)

    # Feature Importance of full XGBoost model
    full_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
    full_xgb.fit(X, y)
    importances = full_xgb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]

    print("\nTOP 5 FEATURE IMPORTANCES (XGBOOST):")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. {feature_names[idx]:<30}: {importances[idx]:.4f}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_cross_validated_comparison()
