import json
import math
import os
from pathlib import Path
from typing import Dict, Any, Optional

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "detector" / "models" / "meta_fusion_weights.json"

class FusionEngine:
    """Ensemble Meta-Fusion Engine that combines threat signals from all 4 inspection layers
    using a trained Wolpert Stacking Logistic Meta-Learner with Platt probability calibration:

    - Layer 1: Rule-based opcode AST bouncer (modelscan)
    - Layer 2: Statistical opcode classifier (SafePickle-inspired XGBoost)
    - Layer 3a: GF-transform LSB payload detector (OSLEmbeddingNet)
    - Layer 3b: Weight-distribution spectral backdoor detector
    """

    def __init__(self):
        self.meta_params = self._load_meta_weights()

    def _load_meta_weights(self) -> Dict[str, Any]:
        """Load empirical parameters trained via maximum likelihood estimation."""
        if _WEIGHTS_PATH.exists():
            try:
                with open(_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data.get("coefficients"), list) and "feature_names" in data:
                        data["coefficients"] = dict(zip(data["feature_names"], data["coefficients"]))
                    return data
            except Exception:
                pass
        # Fallback calibrated defaults
        return {
            "intercept": 0.1350,
            "coefficients": {
                "L1_Verdict": 1.1185,
                "L2_Gated_Prob": 3.5492,
                "L3a_Margin_Delta": 0.5702,
                "L3a_Conviction": 0.4600,
                "L3b_Score": 0.0000,
                "L1_L2_Synergy": 1.0881,
                "Fmt_Pickle": -2.5186,
                "Fmt_Tensor": 0.2386
            }
        }

    def fuse(
        self,
        format_ext: str,
        l1_res: Dict[str, Any],
        l2_res: Dict[str, Any],
        l3a_res: Optional[Dict[str, Any]] = None,
        l3b_res: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ext = format_ext.lower().lstrip(".")
        coef = self.meta_params.get("coefficients", {})
        intercept = float(self.meta_params.get("intercept", 0.1350))

        # Raw layer signal extraction
        s_l1 = 1.0 if l1_res.get("verdict") else 0.0
        p_l2 = float(l2_res.get("probability", 1.0 if l2_res.get("verdict") else 0.0))
        
        tau = float((l3a_res or {}).get("threshold", 0.072236))
        cal_margin = float((l3a_res or {}).get("calibrated_distance", 0.0))
        l3a_margin_delta = (cal_margin - tau) / (tau if tau > 0 else 1.0)
        l3a_v = 1.0 if (l3a_res or {}).get("verdict") else 0.0
        s_l3b = float((l3b_res or {}).get("score", 0.0))

        # Format & feature gating
        is_pickle_fmt = 1.0 if ext in ("pkl", "pt", "pth", "bin") else 0.0
        is_tensor_fmt = 1.0 if ext in ("safetensors", "pt", "pth", "h5", "hdf5") else 0.0
        
        l3a_note = str((l3a_res or {}).get("coverage_note") or "")
        has_float32 = 1.0 if ("No float32" not in l3a_note and "not support" not in l3a_note) else 0.0

        l2_gated = p_l2 * is_pickle_fmt
        l3a_gated = l3a_margin_delta * has_float32
        l1_l2_joint = s_l1 * l2_gated

        # Logit dot product z = beta_0 + sum(beta_i * x_i)
        z = (
            intercept +
            coef.get("L1_Verdict", 1.1185) * s_l1 +
            coef.get("L2_Gated_Prob", 3.5492) * l2_gated +
            coef.get("L3a_Margin_Delta", 0.5702) * l3a_gated +
            coef.get("L3a_Conviction", 0.4600) * l3a_v +
            coef.get("L3b_Score", 0.0000) * s_l3b +
            coef.get("L1_L2_Synergy", 1.0881) * l1_l2_joint +
            coef.get("Fmt_Pickle", -2.5186) * is_pickle_fmt +
            coef.get("Fmt_Tensor", 0.2386) * is_tensor_fmt
        )

        # Calibrated Bayesian posterior probability P(Malicious | x) = sigma(z)
        try:
            prob = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            prob = 0.0 if z < 0 else 1.0

        # Deterministic Safety Anchors (Ground Truth Certainty)
        if s_l1 > 0:
            prob = max(prob, 0.999)
        if l3a_v > 0:
            prob = max(prob, 0.850)

        # Base format-coverage weights (for backwards compatibility / inspection)
        if ext == "safetensors":
            w_l1, w_l2, w_l3a, w_l3b = 0.0, 0.0, 0.60, 0.40
        elif ext == "h5":
            w_l1, w_l2, w_l3a, w_l3b = 0.30, 0.0, 0.40, 0.30
        elif ext in ("pkl", "pt", "pth", "bin"):
            w_l1, w_l2, w_l3a, w_l3b = 0.20, 0.30, 0.30, 0.20
        else:
            w_l1, w_l2, w_l3a, w_l3b = 0.25, 0.25, 0.25, 0.25

        # Unified 3-Tier Classification
        if prob >= 0.70:
            verdict = "MALICIOUS"
        elif prob >= 0.40:
            verdict = "SUSPICIOUS"
        else:
            verdict = "BENIGN"

        # Forensic band preservation (honoring Layer 3a calibrated bands & legacy scores)
        supplied_band = (l3a_res or {}).get("forensic_band")
        s_l3a_legacy = float((l3a_res or {}).get("score", 0.0))
        if supplied_band in ("clean", "suspicious", "malicious"):
            forensic_band = supplied_band
        elif s_l3a_legacy >= 0.70:
            forensic_band = "malicious"
        elif s_l3a_legacy >= 0.40:
            forensic_band = "suspicious"
        elif prob >= 0.70:
            forensic_band = "malicious"
        elif prob >= 0.40:
            forensic_band = "suspicious"
        else:
            forensic_band = "clean"

        explanations = []
        if l1_res.get("verdict"):
            explanations.append("Layer 1: Rule-based dangerous callable detected")
        if l2_res.get("verdict"):
            explanations.append(f"Layer 2: Opcode structural anomaly detected (prob={p_l2:.2f})")
        if l3a_res and l3a_res.get("verdict"):
            explanations.append("Layer 3a: GF LSB steganographic payload detected")
        if l3b_res and l3b_res.get("verdict"):
            explanations.append("Layer 3b: Weight distribution anomaly detected")

        if not explanations:
            explanations.append("All applicable layers clean")

        return {
            "verdict": verdict,
            "score": round(float(prob), 4),
            "forensic_band": forensic_band,
            "weights_applied": {
                "l1": round(w_l1, 2),
                "l2": round(w_l2, 2),
                "l3a": round(w_l3a, 2),
                "l3b": round(w_l3b, 2),
            },
            "meta_coefficients": {
                "l1_coef": round(coef.get("L1_Verdict", 1.1185), 2),
                "l2_coef": round(coef.get("L2_Gated_Prob", 3.5492), 2),
                "l3a_coef": round(coef.get("L3a_Margin_Delta", 0.5702), 2),
                "l3b_coef": round(coef.get("L3b_Score", 0.0000), 2),
            },
            "explanation": " | ".join(explanations),
        }


