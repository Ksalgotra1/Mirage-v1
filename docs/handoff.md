# Handoff Package for Person B (GF Track, Disarmament & Dashboard)

Welcome Person B! All backend infrastructure, Opcode Track (Layers 1 & 2), virtual environment, fixture manifest, and FastAPI orchestration server are complete and verified.

---

## 🛠️ What Person A Built & Handed Off

1. **Virtual Environment**: Pre-configured in [`.venv/`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/.venv) with `torch`, `torchvision`, `xgboost`, `scikit-learn`, `fastapi`, `uvicorn`, and `httpx`.
2. **FastAPI Backend**: Located in [`backend/main.py`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/backend/main.py) running on `http://localhost:8000`.
3. **Fixture Manifest**: Append-only schema in [`fixtures/manifest.json`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/fixtures/manifest.json).
4. **Opcode Detection**: Layer 1 (`modelscan`) and Layer 2 (`XGBoost`) fully trained and integrated into the fusion engine.

---

## 📋 Person B Task Checklist

### Task 1: Layer 3a — GF LSB Detector (`detector/layer3a_gf.py`)
Create [`detector/layer3a_gf.py`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/detector/layer3a_gf.py) implementing `Layer3aGFDetector`:

```python
class Layer3aGFDetector:
    def predict(self, filepath: str) -> dict:
        # 1. Extract raw model weights
        # 2. weights_to_gf_image(weights) -> uint8[S, S]
        # 3. Feature extraction (e.g. frozen ResNet18 backbone)
        # 4. Cosine distance to clean centroid
        return {
            "layer": "3a",
            "verdict": True,          # True if distance >= threshold
            "distance": 0.28,         # Surfaced continuous score for dashboard gauge
            "threshold": 0.15,
            "score": 0.85,            # Normalized 0.0 - 1.0 score for fusion
            "extracted_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "coverage_note": None
        }
```

---

### Task 2: Layer 3b — Weight Distribution Detector (`detector/layer3b_wd.py`)
Create [`detector/layer3b_wd.py`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/detector/layer3b_wd.py) implementing `Layer3bWDDetector`:

```python
class Layer3bWDDetector:
    def predict(self, filepath: str) -> dict:
        # 1. Per-layer statistical metrics (mean, variance, kurtosis, outlier channels)
        # 2. N-standard-deviation deviation from reference model distribution
        return {
            "layer": "3b",
            "verdict": False,
            "deviation": 0.42,        # Standard deviation distance from clean reference
            "threshold": 3.0,
            "score": 0.14,            # Normalized 0.0 - 1.0 score for fusion
            "coverage_note": None
        }
```

> **Integration Note**: Once `layer3a_gf.py` and `layer3b_wd.py` exist in `detector/`, [`backend/main.py`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/backend/main.py) will automatically import and invoke them in sequence!

---

### Task 3: LSB Fixture Generation & Manifest Appending
Add your LSB stego and weight backdoor fixtures in `fixtures/generate_fixtures.py` (or a sub-script) and append them to [`fixtures/manifest.json`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/fixtures/manifest.json):

```json
{
  "filepath": "fixtures/stego_resnet18_6pct.pt",
  "format": "pt",
  "ground_truth_label": "MALICIOUS",
  "attack_type": "lsb_stego",
  "embedding_rate": 0.06,
  "perturbation_magnitude": 0.0
}
```

---

### Task 4: Verified Disarmament
Implement `x_lsb_extract` to pull payload bytes back out, zero/re-randomize low-order bits, and run a forward pass on a benchmark tensor input before and after to verify `logit_diff < 0.001`.

---

### Task 5: React Dashboard (`frontend/`)
Connect your React UI to Person A's FastAPI backend:

* `POST http://localhost:8000/scan` (Send multipart file upload)
  * Response contains `layer1`, `layer2`, `layer3a`, `layer3b`, `fused`, and `disarmament` objects.
* `GET http://localhost:8000/fixtures/eval` (Fetch evaluation metrics & format coverage table)
* `GET http://localhost:8000/health` (API health check)

---

## ⚡ Quick Start Commands for Person B

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run backend API server
.venv/bin/python backend/main.py

# 3. Test API scanning endpoint
.venv/bin/python backend/test_api.py

# 4. Re-run evaluation harness
.venv/bin/python eval/run_eval.py
```
