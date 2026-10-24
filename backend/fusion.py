
from typing import Dict, Any, Optional

class FusionEngine:
    """Ensemble Fusion Engine that combines scores from all four detection layers:

    - Layer 1: Rule-based opcode bouncer (modelscan)
    - Layer 2: Statistical opcode classifier (SafePickle-inspired)
    - Layer 3a: GF-transform LSB payload detector
    - Layer 3b: Weight-distribution backdoor detector

    Applies format-aware weight adjustment based on file format coverage.
    """

    def fuse(
        self,
        format_ext: str,
        l1_res: Dict[str, Any],
        l2_res: Dict[str, Any],
        l3a_res: Optional[Dict[str, Any]] = None,
        l3b_res: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ext = format_ext.lower().lstrip(".")

        # Base default weights
        w_l1, w_l2, w_l3a, w_l3b = 0.20, 0.30, 0.30, 0.20

        # Adjust weights based on format coverage
        if ext == "safetensors":
            w_l1, w_l2, w_l3a, w_l3b = 0.0, 0.0, 0.60, 0.40
        elif ext == "h5":
            w_l1, w_l2, w_l3a, w_l3b = 0.30, 0.0, 0.40, 0.30
        elif ext in ("pkl", "pt", "pth"):
            w_l1, w_l2, w_l3a, w_l3b = 0.20, 0.30, 0.30, 0.20

        # Extract normalized scores (0.0 to 1.0)
        s_l1 = 1.0 if l1_res.get("verdict") else 0.0
        s_l2 = float(l2_res.get("probability", 1.0 if l2_res.get("verdict") else 0.0))

        # Layer 3a score
        if l3a_res and "score" in l3a_res:
            s_l3a = float(l3a_res["score"])
        elif l3a_res and "verdict" in l3a_res:
            s_l3a = 1.0 if l3a_res["verdict"] else 0.0
        else:
            s_l3a = 0.0

        # Layer 3b score
        if l3b_res and "score" in l3b_res:
            s_l3b = float(l3b_res["score"])
        elif l3b_res and "verdict" in l3b_res:
            s_l3b = 1.0 if l3b_res["verdict"] else 0.0
        else:
            s_l3b = 0.0

        # Compute fused score
        total_weight = w_l1 + w_l2 + w_l3a + w_l3b
        if total_weight > 0:
            fused_score = (w_l1 * s_l1 + w_l2 * s_l2 + w_l3a * s_l3a + w_l3b * s_l3b) / total_weight
        else:
            fused_score = 0.0

        # Override: If any layer detects a confirmed threat, or fused score >= 0.5, mark as MALICIOUS
        is_malicious = (
            fused_score >= 0.5
            or bool(l1_res.get("verdict", False))
            or bool(l2_res.get("verdict", False))
            or (l3a_res is not None and bool(l3a_res.get("verdict", False)))
            or (l3b_res is not None and bool(l3b_res.get("verdict", False)))
        )

        explanations = []
        if l1_res.get("verdict"):
            explanations.append("Layer 1: Rule-based dangerous callable detected")
        if l2_res.get("verdict"):
            explanations.append(f"Layer 2: Opcode structural anomaly detected (prob={s_l2:.2f})")
        if l3a_res and l3a_res.get("verdict"):
            explanations.append("Layer 3a: GF LSB payload pattern detected")
        if l3b_res and l3b_res.get("verdict"):
            explanations.append("Layer 3b: Weight distribution anomaly detected")

        if not explanations:
            explanations.append("All applicable layers clean")

        return {
            "verdict": "MALICIOUS" if is_malicious else "BENIGN",
            "score": round(float(fused_score), 4),
            "weights_applied": {
                "l1": round(w_l1, 2),
                "l2": round(w_l2, 2),
                "l3a": round(w_l3a, 2),
                "l3b": round(w_l3b, 2),
            },
            "explanation": " | ".join(explanations),
        }
