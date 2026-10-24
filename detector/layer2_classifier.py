import os
import sys

# macOS ARM64 safety net: Prevent OpenMP runtime collision between XGBoost and PyTorch
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import pickle
import pickletools
import re
import zipfile
import numpy as np

TRACKED_UNIGRAMS = [
    "GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "INST", "OBJ",
    "SETITEMS", "SETITEM", "MARK", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3",
    "EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE", "SHORT_BINUNICODE", "BINUNICODE",
    "BININT", "BININT1", "BININT2", "NONE", "NEWTRUE", "NEWFALSE", "PROTO", "STOP"
]

TRACKED_BIGRAMS = [
    ("GLOBAL", "REDUCE"),
    ("STACK_GLOBAL", "REDUCE"),
    ("MARK", "GLOBAL"),
    ("MARK", "STACK_GLOBAL"),
    ("TUPLE", "REDUCE"),
    ("TUPLE1", "REDUCE"),
    ("GLOBAL", "MARK"),
    ("EMPTY_DICT", "SETITEMS"),
    ("EMPTY_LIST", "APPENDS"),
    ("PROTO", "GLOBAL"),
    ("PROTO", "EMPTY_DICT"),
    ("BUILD", "STOP"),
    ("REDUCE", "BUILD"),
    ("REDUCE", "STOP")
]


def safe_extract_opcodes(data: bytes) -> list:
    """Safely extracts opcode names from raw pickle bytes, handling Protocol 4/5 framing."""
    opcodes = []
    try:
        for opcode, arg, pos in pickletools.genops(data):
            opcodes.append(opcode.name)
    except Exception:
        # Fallback regex/byte scan for Protocol 4/5 checkpoints with unsupported framing
        for op in pickletools.opcodes:
            if not op.code:
                continue
            pattern = re.escape(op.code.encode("latin1"))
            count = len(re.findall(pattern, data))
            if count > 0:
                opcodes.extend([op.name] * count)
    return opcodes


class OpcodeFeatureExtractor:
    """Extracts normalized opcode unigram frequencies, bigram transition frequencies,
    and structural metrics from pickle/pt files without executing untrusted code.
    """

    def extract_features(self, filepath: str) -> np.ndarray:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in {".pkl", ".pt", ".pth", ".bin"}:
            return np.zeros(len(TRACKED_UNIGRAMS) + len(TRACKED_BIGRAMS) + 3, dtype=np.float32)

        try:
            content = None
            if zipfile.is_zipfile(filepath):
                with zipfile.ZipFile(filepath, "r") as z:
                    for name in z.namelist():
                        if name.endswith(".pkl") or name.endswith("/data.pkl") or name == "data.pkl":
                            try:
                                content = z.read(name)
                                if content:
                                    break
                            except Exception:
                                pass

            if content is None:
                with open(filepath, "rb") as f:
                    content = f.read()

            opcodes = safe_extract_opcodes(content)

            if not opcodes:
                return np.zeros(len(TRACKED_UNIGRAMS) + len(TRACKED_BIGRAMS) + 3, dtype=np.float32)

            total_ops = len(opcodes)
            unigram_counts = {u: 0 for u in TRACKED_UNIGRAMS}
            for op in opcodes:
                if op in unigram_counts:
                    unigram_counts[op] += 1

            unigram_freqs = [unigram_counts[u] / total_ops for u in TRACKED_UNIGRAMS]

            bigram_counts = {f"{b[0]}->{b[1]}": 0 for b in TRACKED_BIGRAMS}
            total_bigrams = max(1, total_ops - 1)

            for i in range(len(opcodes) - 1):
                bg_key = f"{opcodes[i]}->{opcodes[i+1]}"
                if bg_key in bigram_counts:
                    bigram_counts[bg_key] += 1

            bigram_freqs = [bigram_counts[f"{b[0]}->{b[1]}"] / total_bigrams for b in TRACKED_BIGRAMS]

            # Structural callable anomaly metrics
            reduce_count = opcodes.count("REDUCE") + opcodes.count("BUILD")
            global_count = opcodes.count("GLOBAL") + opcodes.count("STACK_GLOBAL")
            eval_risk_ratio = (reduce_count + global_count) / max(1, total_ops)

            metrics = [
                float(total_ops),
                float(global_count),
                float(eval_risk_ratio)
            ]

            return np.array(unigram_freqs + bigram_freqs + metrics, dtype=np.float32)

        except Exception:
            return np.zeros(len(TRACKED_UNIGRAMS) + len(TRACKED_BIGRAMS) + 3, dtype=np.float32)

    def get_feature_names(self):
        names = [f"unigram_{u}" for u in TRACKED_UNIGRAMS]
        names += [f"bigram_{b[0]}->{b[1]}" for b in TRACKED_BIGRAMS]
        names += ["total_opcodes", "global_opcode_count", "eval_risk_ratio"]
        return names


class Layer2Classifier:
    """Layer 2: Statistical Opcode Classifier using XGBoost."""

    MODEL_JSON_PATH = os.path.join(os.path.dirname(__file__), "models", "layer2_xgb.json")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "layer2_xgb.pkl")

    def __init__(self):
        self.extractor = OpcodeFeatureExtractor()
        self.model = self._load_model()

    def _load_model(self):
        # Prefer XGBoost native JSON format (avoids OpenMP crash and pickle serialization issues)
        if os.path.exists(self.MODEL_JSON_PATH):
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(self.MODEL_JSON_PATH)
                return model
            except Exception:
                pass
        # Fallback to legacy pickle format if JSON is unavailable
        if os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def predict(self, filepath: str) -> dict:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in {".pkl", ".pt", ".pth"}:
            return {
                "layer": 2,
                "verdict": False,
                "probability": 0.0,
                "top_features": [],
                "coverage_note": f"Layer 2 operates on pickle opcodes only (.pt/.pkl); {ext} not applicable",
            }

        features = self.extractor.extract_features(filepath)
        feature_names = self.extractor.get_feature_names()

        # If zero features extracted (e.g. invalid or empty pickle stream)
        if np.all(features == 0):
            return {
                "layer": 2,
                "verdict": False,
                "probability": 0.0,
                "top_features": [],
                "coverage_note": "No pickle opcodes detected in file",
            }

        if self.model is not None:
            try:
                proba = float(self.model.predict_proba([features])[0][1])
                verdict = bool(proba >= 0.5)

                importances = getattr(self.model, "feature_importances_", np.zeros(len(features)))
                top_idx = np.argsort(importances * features)[::-1][:3]
                top_feats = []
                for idx in top_idx:
                    if importances[idx] > 0 and features[idx] > 0:
                        top_feats.append({
                            "feature": feature_names[idx],
                            "importance": round(float(importances[idx]), 4),
                            "value": round(float(features[idx]), 4)
                        })

                return {
                    "layer": 2,
                    "verdict": verdict,
                    "probability": round(proba, 4),
                    "top_features": top_feats,
                    "coverage_note": None,
                }
            except Exception:
                pass

        # Fallback structural ratio check
        eval_risk_ratio = features[-1]
        global_count = features[-2]
        proba = min(1.0, (eval_risk_ratio * 3.0) + (global_count * 0.2))
        verdict = bool(proba >= 0.5)

        return {
            "layer": 2,
            "verdict": verdict,
            "probability": round(float(proba), 4),
            "top_features": [{"feature": "eval_risk_ratio (fallback)", "importance": 1.0, "value": round(float(eval_risk_ratio), 4)}],
            "coverage_note": "Using structural risk ratio fallback",
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        l2 = Layer2Classifier()
        print(l2.predict(sys.argv[1]))
