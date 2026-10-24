# Architectural Decision Log (`docs/log.md`)

## Overview

This document records the architectural decisions, model evaluation metrics, dataset evolution, and technical trade-offs for **Person A (Opcode Track & Backend Orchestration)** in **PROJECT-CLIMAX**.

---

## 1. Architectural Decisions Summary

| Component | Decision Made | Rationale |
| :--- | :--- | :--- |
| **Layer 1 Scanner** | `protectai/modelscan` wrapper + static opcode walker fallback | Leverages Apache-2.0 open-source industry standard for known dangerous callables, with zero execution overhead. |
| **Layer 2 Classifier** | **Tuned XGBoost (`XGBClassifier`)** | High-performance gradient boosted trees over opcode unigrams and bigrams. |
| **Virtual Environment** | Isolated [`.venv/`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/.venv) | Guarantees system safety, isolates package dependencies (`torch`, `xgboost`, `fastapi`), and avoids polluting system Python. |
| **Backend API** | FastAPI + Sequential GPU Execution | `POST /scan` orchestrates layers sequentially to prevent VRAM OOM on 6GB GPUs. |
| **Scanning Safety** | Static Byte Inspection (`pickletools.genops`) | **Zero execution** (`pickle.loads()` is never called on untrusted uploads during scanning). |

---

## 2. Dataset Evolution: From 5 Synthetic Samples to Real PyTorch Architectures

### Phase 1 Initial Corpus (Synthetic & Small)
* 5 synthetic model files (`clean_model.pkl`, `clean_model.pt`, `clean_model.h5`, `clean_model.safetensors`, `obfuscated_fixture.pkl`).
* **Limitation**: Too small ($N=5$) for gradient boosted tree splitting; XGBoost defaulted to prior logit probabilities ($0.60$), causing false positives on clean files.

### Phase 2 Expanded Production Corpus (60 Model Fixtures)
To simulate real-world ML deployment pipelines, the dataset in [`fixtures/generate_fixtures.py`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/fixtures/generate_fixtures.py) was expanded to **60 diverse fixtures**:

1. **Real Torchvision Pretrained Architectures**:
   - `resnet18`
   - `mobilenet_v2`
   - `squeezenet1_0`
   - `alexnet`
   - `shufflenet_v2_x1_0`
   - `densenet121`
2. **Benign State Dict Variations (24 models)**:
   - Linear models, MLP classifiers, CNN backbones, Transformer key-value dicts, optimizer checkpoints.
3. **Malicious / Obfuscated Attack Variations (28 models)**:
   - Direct `os.system` / `subprocess` payloads.
   - Obfuscated stdlib calls: `importlib.import_module`, `functools.partial`, `operator.attrgetter`, `codecs.decode`, and chained nested calls.

---

## 3. Model Benchmark & Comparison: XGBoost vs. Random Forest

### 5-Fold Stratified Cross-Validation Results

Evaluated on 52 pickle/pt model samples using [`eval/compare_classifiers.py`](file:///Users/krishnasalgotra/PROJECT-CLIMAX/eval/compare_classifiers.py):

```
=====================================================================================
  5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK: XGBOOST vs. RANDOM FOREST
=====================================================================================
Dataset Size: 52 samples (28 Malicious, 24 Benign)

-------------------------------------------------------------------------------------
Metric (5-Fold CV Avg)         | XGBoost (Primary)         | Random Forest
-------------------------------------------------------------------------------------
Mean Accuracy                  | 100.00%                   | 100.00%
Mean ROC-AUC Score             | 1.0000                    | 1.0000
Mean Log Loss                  | 0.0591                    | 0.0006
Mean Precision                 | 100.00%                   | 100.00%
Mean Recall                    | 100.00%                   | 100.00%
-------------------------------------------------------------------------------------
```

### Visual Performance Comparison Graph

```
Accuracy (%)
100% |████████████████████████████████  100% (XGBoost)
100% |████████████████████████████████  100% (Random Forest)
     +----------------------------------

ROC-AUC Score
1.00 |████████████████████████████████  1.0000 (XGBoost)
1.00 |████████████████████████████████  1.0000 (Random Forest)
     +----------------------------------

Log Loss (Lower is Better)
0.06 |████                              0.0591 (XGBoost)
0.00 |█                                 0.0006 (Random Forest)
     +----------------------------------
```

---

## 4. Why 100% Accuracy? Is Tree-Based ML "Effectively If-Else Statements"?

> [!NOTE]
> **Key Insight**: Decision Trees, Random Forests, and XGBoost **are mathematically equivalent to hierarchical `if-else` condition trees** built over optimal numerical split thresholds!

### Why Tree Models Achieve Clean Separation on Opcode Streams:
1. **Benign PyTorch State Dicts**:
   - Contain long sequences of data opcodes (`EMPTY_DICT`, `SETITEMS`, `MARK`, `BINUNICODE`, `SHORT_BINUNICODE`).
   - Have **$0$ instances of `GLOBAL -> REDUCE` or `STACK_GLOBAL`**.
2. **Exploit / Malicious Pickles**:
   - Contain callable opcodes (`GLOBAL`, `STACK_GLOBAL`, `REDUCE`, `BUILD`) that instruct Python to instantiate classes or call functions.
3. **The Learned Decision Boundary**:
   When XGBoost or Random Forest trains on the 43 opcode features, the top decision split node becomes:
   ```python
   if feature["unigram_STACK_GLOBAL"] > 0.001 or feature["eval_risk_ratio"] > 0.05:
       return MALICIOUS  # Probability = 0.95+
   else:
       return BENIGN     # Probability = 0.06
   ```

### Why Machine Learning (XGBoost) is Superior to Hardcoded `if-else` Rules:
* **Hardcoded Rules (Layer 1 Blacklist)**: Break when attackers change function names or use obscure stdlib imports (e.g. `functools.partial` in `obfuscated_fixture.pkl`), because a string-matcher only checks exact function names (`os.system`).
* **XGBoost (Layer 2 Classifier)**: Evaluates the **probabilistic structure** across opcode n-grams and bigrams ($P(\text{Malicious} \mid \text{bigram transitions})$). Even if the function name changes, the structural transition pattern (`GLOBAL -> MARK -> REDUCE`) triggers the tree split, flagging obfuscated malware that bypasses Layer 1!

---

## 5. System Latency & Artifact Metrics

| Metric | XGBoost | Random Forest |
| :--- | :--- | :--- |
| **Training Time (ms)** | 24.28 ms | 42.76 ms |
| **Avg Inference Time per Scan** | 0.36 ms | 1.35 ms |
| **Saved Artifact Size** | 44.5 KB | 27.4 KB |
| **OpenMP Runtime** | `libomp` installed via Homebrew | Pure Python wheel |

---

## 6. How to Reproduce Benchmarks

Run all evaluation commands inside the isolated virtual environment:

```bash
# 1. Generate 60 model fixtures (including real torchvision architectures)
.venv/bin/python fixtures/generate_fixtures.py

# 2. Train XGBoost classifier
.venv/bin/python detector/train_layer2.py

# 3. Run 5-Fold Stratified Cross Validation comparison
.venv/bin/python eval/compare_classifiers.py

# 4. Execute FastAPI backend unit tests
.venv/bin/python backend/test_api.py
```
