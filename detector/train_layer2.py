import json
import os
import pickle
import numpy as np
import xgboost as xgb

try:
    from detector.layer2_classifier import OpcodeFeatureExtractor
except ImportError:
    from layer2_classifier import OpcodeFeatureExtractor

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "manifest.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "layer2_xgb.pkl")


def train_model(manifest_path=MANIFEST_PATH, model_save_path=MODEL_PATH):
    """Trains the primary Layer 2 XGBoost Opcode Classifier on model fixtures."""
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}. Generating default fixtures first...")
        from fixtures.generate_fixtures import generate_large_fixture_dataset
        generate_large_fixture_dataset()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    extractor = OpcodeFeatureExtractor()
    X = []
    y = []

    project_root = os.path.dirname(os.path.dirname(__file__))

    for entry in manifest:
        fmt = entry.get("format", "").lower()
        if fmt not in {"pkl", "pt"}:
            continue

        rel_path = entry.get("filepath", "")
        abs_path = os.path.join(project_root, rel_path) if not os.path.isabs(rel_path) else rel_path

        if not os.path.exists(abs_path):
            continue

        feats = extractor.extract_features(abs_path)
        label = 1 if entry.get("ground_truth_label", "").upper() == "MALICIOUS" else 0

        X.append(feats)
        y.append(label)

    if not X:
        print("No valid pickle/pt fixtures found for training!")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Training Layer 2 XGBoost model on {len(X)} samples ({np.sum(y == 1)} Malicious, {np.sum(y == 0)} Benign)...")

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    clf.fit(X, y)

    with open(model_save_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Successfully saved streamlined XGBoost model artifact to {model_save_path}")
    return clf


if __name__ == "__main__":
    train_model()
