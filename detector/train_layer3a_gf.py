"""Train a Layer 3a v4 ModelXRay-style SCZ episode with Multi-Experiment Benchmarking & Exhaustive Sample Diagnostics.

Features:
  - 3 Multi-Experiment Benchmarks (Dev-Mixed, Dev-ZeroShot-Small, Dev-ZeroShot-Large)
  - Smooth Softplus Triplet Loss & Cosine Annealing Learning Rate
  - Generates 5 Publication-Quality Presentation Charts (including Forensic Band Distributions)
  - Exports exhaustive per-sample diagnostics to diagnostic_sample_scores.csv
  - Exports complete threshold calibration and band distinctions to calibration JSON
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

# Ensure UTF-8 output handling on Windows consoles
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.models.gf_contrastive_net import OSLEmbeddingNet
from fixtures.layer3a_manifest import load_manifest, manifest_sha256

# ── 22-Model Zoo Partitioning ──────────────────────────────────────────────────
# Regime A: 10 Homogeneous ResNets for SCZ Support Training (40 samples: 20 clean + 20 stego)
HOMO_TRAIN_BASE_IDS = frozenset([
    "resnet18-v1", "resnet34-v1", "resnet50-v1", "resnet50-v2",
    "resnet101-v1", "resnet101-v2", "resnet152-v1", "resnet152-v2",
    "resnext50-v1", "wide-resnet50-v1",
])
TRAIN_BASE_IDS = HOMO_TRAIN_BASE_IDS

# Regime B: Cross-Family Invariant Support (5 Distinct Families: 2 ResNet, 2 DenseNet, 2 MobileNet, 2 SqueezeNet, 2 Lightweight/Inception)
CROSS_FAMILY_TRAIN_BASE_IDS = frozenset([
    "resnet18-v1", "resnet50-v1",
    "densenet121", "densenet169",
    "mobilenet-v2", "mobilenet-v3-small",
    "squeezenet10", "squeezenet11",
    "googlenet", "shufflenet-v2",
])

# 3 Held-Out ResNets (Homogeneous Dev)
HOMO_HELD_OUT_IDS = frozenset([
    "resnext50-v2", "resnext101-v1", "wide-resnet101-v1",
])

# Regime A Exp 1: Dev-Mixed (3 Homo ResNets + 5 Unseen Diverse = 8 Base Models / 24 Artifacts)
MIXED_DEV_IDS = frozenset([
    "resnext50-v2", "resnext101-v1", "wide-resnet101-v1",
    "squeezenet10", "mobilenet-v3-small", "mobilenet-v2", "densenet121", "googlenet",
])

# Regime A Exp 2: Dev-ZeroShot-Small (4 Distinct Unseen Architectures / 12 Artifacts)
ZEROSHOT_SMALL_IDS = frozenset([
    "squeezenet11", "shufflenet-v2", "efficientnet-b0", "densenet169",
])

# Regime A Exp 3: Dev-ZeroShot-Large (All 9 Diverse Unseen Architectures / 27 Artifacts)
ZEROSHOT_LARGE_IDS = frozenset([
    "squeezenet10", "squeezenet11", "mobilenet-v3-small", "shufflenet-v2",
    "mobilenet-v2", "efficientnet-b0", "densenet121", "densenet169", "googlenet",
])

# Regime B Held-Out Zero-Shot Test Sets (10 Unseen ResNets + 1 Unseen EfficientNet)
CROSS_HELD_OUT_RESNET_IDS = frozenset([
    "resnet34-v1", "resnet101-v1", "resnet101-v2", "resnet152-v1", "resnet152-v2",
    "resnext50-v1", "resnext50-v2", "resnext101-v1", "wide-resnet50-v1", "wide-resnet101-v1",
])
CROSS_HELD_OUT_DIVERSE_IDS = frozenset([
    "efficientnet-b0",
])


def load_gf_image(filepath: str | Path) -> torch.Tensor:
    """Load one raw uint8 GF image as ``[1, H, W]`` without normalization."""
    arr = np.load(filepath)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected [1, H, H] square image, got {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.uint8)))


def items_for_records(records: list[dict], x_bits: int) -> list[dict]:
    items = []
    for record in records:
        attack = record["attack"]
        if record["label"] == "stego" and int(attack.get("x_bits", -1)) != x_bits:
            continue
        items.append({
            "image": load_gf_image(record["filepath"]),
            "label": int(record["label"] == "stego"),
            "artifact_id": record["artifact_id"],
            "base_checkpoint_id": record["base_checkpoint_id"],
            "architecture": record.get("architecture", record["architecture_family"]),
            "architecture_family": record["architecture_family"],
        })
    return items


def select_scz_support(items: list[dict], architecture: str, per_class: int = 0, designated_bases: frozenset | None = None) -> list[dict]:
    """Select a symmetrically paired support set with ALL stego variants included."""
    all_archs = sorted({item.get("architecture") for item in items if item.get("architecture")})
    all_families = sorted({item.get("architecture_family") for item in items if item.get("architecture_family")})

    if architecture in ("cross_family", "famous_small", "all", "cross_arch"):
        matching = items
        match_type = "cross-family"
    elif architecture in all_archs:
        matching = [item for item in items if item.get("architecture") == architecture]
        match_type = "exact topology"
    elif architecture in all_families:
        matching = [item for item in items if item.get("architecture_family") == architecture]
        match_type = "architecture family"
    else:
        matching = items
        match_type = "fallback-all"

    # Filter to designated training base IDs if present
    target_bases = designated_bases if designated_bases is not None else TRAIN_BASE_IDS
    matching_bases = [item for item in matching if item.get("base_checkpoint_id") in target_bases]
    if len(matching_bases) >= 10:
        matching = matching_bases

    by_base: dict[str, dict[int, list[dict]]] = {}
    for item in matching:
        base_id = item.get("base_checkpoint_id", item.get("architecture", item.get("artifact_id", "default")))
        by_base.setdefault(base_id, {}).setdefault(item["label"], []).append(item)

    valid_bases = sorted([b for b, labels in by_base.items() if 0 in labels and 1 in labels])
    target_count = len(valid_bases) if per_class <= 0 else min(per_class, len(valid_bases))

    if len(valid_bases) < target_count or target_count == 0:
        req = max(1, per_class)
        selected: list[dict] = []
        for label in (0, 1):
            candidates = sorted((x for x in matching if x["label"] == label), key=lambda x: x["artifact_id"])
            selected.extend(candidates[:req])
        return selected

    selected: list[dict] = []
    chosen_bases = valid_bases[:target_count]
    for b in chosen_bases:
        clean_items = sorted(by_base[b][0], key=lambda x: x["artifact_id"])
        stego_items = sorted(by_base[b][1], key=lambda x: x["artifact_id"])

        # Include 2 clean samples (if 1 clean exists, augment/pair to match 2 stego)
        if len(clean_items) >= len(stego_items):
            selected.extend(clean_items[:len(stego_items)])
        else:
            # Replicate/balance clean items so N_clean == N_stego per base model
            for i in range(len(stego_items)):
                base_clean = clean_items[i % len(clean_items)]
                clean_replica = dict(base_clean)
                if i > 0:
                    clean_replica["artifact_id"] = f"{base_clean['artifact_id']}_bal{i}"
                selected.append(clean_replica)

        # Include all stego samples
        for stego_item in stego_items:
            selected.append(stego_item)

    n_clean = len([s for s in selected if s['label'] == 0])
    n_stego = len([s for s in selected if s['label'] == 1])
    print(f"[SCZ Support Pairs] Symmetrically paired {len(chosen_bases)} base models ({n_clean} clean + {n_stego} stego): {chosen_bases}")
    return selected


def make_triplet_indices(items: list[dict] | list[int], cross_family: bool = False) -> list[tuple[int, int, int]]:
    """Build deterministic class-balanced triplets.

    Integer labels remain supported for the lightweight regression contract;
    corpus training passes the richer item dictionaries.
    """
    labels = [int(item) if isinstance(item, (int, np.integer)) else item["label"] for item in items]
    families = ["unknown" if isinstance(item, (int, np.integer)) else item.get("architecture_family", "unknown") for item in items]
    triplets: list[tuple[int, int, int]] = []

    for label in (0, 1):
        positives = [i for i, value in enumerate(labels) if value == label]
        negatives = [i for i, value in enumerate(labels) if value != label]
        for anchor, positive in combinations(positives, 2):
            is_cross = (families[anchor] != families[positive])
            triplets.append((anchor, positive, random.choice(negatives) if len(negatives) > 50 else negatives[0]))
            # Full cross-product for rich signal
            for negative in negatives:
                triplets.append((anchor, positive, negative))
                if cross_family and is_cross:
                    # Boost cross-family invariant pairs
                    triplets.append((anchor, positive, negative))

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in triplets:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Smooth Softplus Triplet Loss — avoids early gradient collapse to zero."""
    d_ap = (anchor - positive).square().sum(dim=-1)
    d_an = (anchor - negative).square().sum(dim=-1)
    return torch.nn.functional.softplus(d_ap - d_an + margin).mean()


def embeddings(model: OSLEmbeddingNet, items: list[dict], device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        batch = torch.stack([item["image"] for item in items]).to(device)
        return model(batch).cpu().numpy()


def find_calibrated_threshold(
    benign_scores: np.ndarray,
    malicious_scores: np.ndarray,
) -> float:
    """Prior-Balanced optimal support decision threshold search with margin center tie-breaking."""
    if len(benign_scores) == 0 or len(malicious_scores) == 0:
        return 0.0
    all_scores = sorted(set(list(benign_scores) + list(malicious_scores)))
    candidates = [all_scores[0] - 0.1]
    for a, b in zip(all_scores[:-1], all_scores[1:]):
        candidates.append(0.5 * (a + b))
    candidates.append(all_scores[-1] + 0.1)

    best_t, best_score = candidates[0], -1.0
    for t in candidates:
        tnr = float((benign_scores <= t).mean())
        tpr = float((malicious_scores > t).mean())
        ba = 0.5 * (tnr + tpr)

        # Margin gap centering: prefer placing threshold in the center of the clean/stego gap
        clean_below = benign_scores[benign_scores <= t]
        stego_above = malicious_scores[malicious_scores > t]
        clean_margin = float(t - np.max(clean_below)) if len(clean_below) > 0 else 0.0
        stego_margin = float(np.min(stego_above) - t) if len(stego_above) > 0 else 0.0
        margin_score = min(clean_margin, stego_margin)

        score = ba + 0.001 * margin_score
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)


def classify_support(
    query_embeddings: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    apply_cl2: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Paper CL2-transformed nearest-centroid and 1-NN classification with calibrated threshold."""
    if len(query_embeddings) == 0:
        return np.array([]), np.array([]), 0.0

    if apply_cl2:
        mean_train = support_embeddings.mean(axis=0, keepdims=True)
        sup_c = support_embeddings - mean_train
        sup_proc = sup_c / np.linalg.norm(sup_c, axis=1, keepdims=True).clip(1e-7)

        q_c = query_embeddings - mean_train
        q_proc = q_c / np.linalg.norm(q_c, axis=1, keepdims=True).clip(1e-7)

        c0 = sup_proc[support_labels == 0].mean(axis=0)
        c0 = c0 / max(np.linalg.norm(c0), 1e-7)
        c1 = sup_proc[support_labels == 1].mean(axis=0)
        c1 = c1 / max(np.linalg.norm(c1), 1e-7)

        s_sup = np.dot(sup_proc, c1) - np.dot(sup_proc, c0)
        tau = find_calibrated_threshold(s_sup[support_labels == 0], s_sup[support_labels == 1])

        s_q = np.dot(q_proc, c1) - np.dot(q_proc, c0)
        centroid_pred = (s_q > tau).astype(int)

        nn_indices = np.argmin(((q_proc[:, None] - sup_proc[None]) ** 2).sum(axis=-1), axis=1)
        return centroid_pred, support_labels[nn_indices].astype(int), tau
    else:
        c0 = support_embeddings[support_labels == 0].mean(axis=0)
        c1 = support_embeddings[support_labels == 1].mean(axis=0)

        d0_sup = np.linalg.norm(support_embeddings - c0, axis=1)
        d1_sup = np.linalg.norm(support_embeddings - c1, axis=1)
        s_sup = d0_sup - d1_sup
        tau = find_calibrated_threshold(s_sup[support_labels == 0], s_sup[support_labels == 1])

        d0_q = np.linalg.norm(query_embeddings - c0, axis=1)
        d1_q = np.linalg.norm(query_embeddings - c1, axis=1)
        s_q = d0_q - d1_q
        centroid_pred = (s_q > tau).astype(int)

        nn_indices = np.argmin(((query_embeddings[:, None] - support_embeddings[None]) ** 2).sum(axis=-1), axis=1)
        return centroid_pred, support_labels[nn_indices].astype(int), tau


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"ba": 0.0, "tpr": 0.0, "tnr": 0.0, "precision": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
    ba = (tpr + tnr) / 2.0
    return {
        "ba": float(ba),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "precision": float(precision),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return compute_classification_metrics(y_true, y_pred)["ba"]


def evaluate_split(
    model: OSLEmbeddingNet,
    dev_items: list[dict],
    support_images: torch.Tensor,
    support_labels: np.ndarray,
    device: str,
) -> dict[str, Any]:
    """Evaluate one dev subset returning Raw, CL2, Precision, Recall, and F1 metrics."""
    if not dev_items:
        return {"centroid_ba": 0.0, "1nn_ba": 0.0, "best_ba": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    with torch.no_grad():
        sup_emb = model(support_images.to(device)).cpu().numpy()
        dev_imgs = torch.stack([item["image"] for item in dev_items]).to(device)
        dev_emb = model(dev_imgs).cpu().numpy()

    dev_lbls = np.array([item["label"] for item in dev_items])
    c_pred_r, nn_pred_r, _ = classify_support(dev_emb, sup_emb, support_labels, apply_cl2=False)
    m_c_r = compute_classification_metrics(dev_lbls, c_pred_r)
    m_nn_r = compute_classification_metrics(dev_lbls, nn_pred_r)

    c_pred_c, nn_pred_c, _ = classify_support(dev_emb, sup_emb, support_labels, apply_cl2=True)
    m_c_c = compute_classification_metrics(dev_lbls, c_pred_c)
    m_nn_c = compute_classification_metrics(dev_lbls, nn_pred_c)

    # Pick the superior classifier mode (Centroid vs 1-NN)
    best_m = max([m_c_r, m_nn_r, m_c_c, m_nn_c], key=lambda x: x["ba"])

    return {
        "raw_centroid_ba": m_c_r["ba"],
        "raw_1nn_ba": m_nn_r["ba"],
        "cl2_centroid_ba": m_c_c["ba"],
        "cl2_1nn_ba": m_nn_c["ba"],
        "centroid_ba": max(m_c_r["ba"], m_c_c["ba"]),
        "1nn_ba": max(m_nn_r["ba"], m_nn_c["ba"]),
        "best_ba": best_m["ba"],
        "precision": best_m["precision"],
        "recall": best_m["tpr"],
        "specificity": best_m["tnr"],
        "f1": best_m["f1"],
        "tp": best_m["tp"],
        "fp": best_m["fp"],
        "tn": best_m["tn"],
        "fn": best_m["fn"],
    }


def train_episode(
    model: OSLEmbeddingNet,
    support: list[dict],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    margin: float,
    device: str,
    cross_family: bool = False,
    dev_mixed_items: list[dict] | None = None,
    dev_zs_small_items: list[dict] | None = None,
    dev_zs_large_items: list[dict] | None = None,
) -> list[dict]:
    support_images = torch.stack([item["image"] for item in support])
    model.adapt_normalization(support_images)
    model.to(device)
    labels = np.array([item["label"] for item in support])
    triplets = make_triplet_indices(support, cross_family=cross_family)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history: list[dict] = []
    rng = random.Random(42)
    _best_score: float = -1.0
    _best_state: dict | None = None
    _best_epoch: int = 0

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            rng.shuffle(triplets)
            losses = []
            for start in range(0, len(triplets), batch_size):
                current = triplets[start:start + batch_size]
                indices = torch.tensor(current, device=device)
                batch = support_images.to(device)
                optimizer.zero_grad()
                all_embeddings = model(batch)
                loss = triplet_loss(
                    all_embeddings[indices[:, 0]],
                    all_embeddings[indices[:, 1]],
                    all_embeddings[indices[:, 2]],
                    margin,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite triplet loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            scheduler.step()
            value = float(np.mean(losses))
            recent_losses = [row["triplet_loss"] for row in history[-5:]] + [value] if history else [value]
            trailing_mean = float(np.mean(recent_losses[-5:]))

            # Evaluate across the 3 Experiments
            model.eval()
            m_mixed = evaluate_split(model, dev_mixed_items or [], support_images, labels, device)
            m_zs_small = evaluate_split(model, dev_zs_small_items or [], support_images, labels, device)
            m_zs_large = evaluate_split(model, dev_zs_large_items or [], support_images, labels, device)

            current_lr = float(scheduler.get_last_lr()[0])
            history.append({
                "epoch": epoch,
                "triplet_loss": value,
                "trailing5": trailing_mean,
                "lr": current_lr,
                "dev_mixed_ba": m_mixed["best_ba"],
                "dev_mixed_centroid_ba": m_mixed["centroid_ba"],
                "dev_mixed_1nn_ba": m_mixed["1nn_ba"],
                "dev_zs_small_ba": m_zs_small["best_ba"],
                "dev_zs_small_centroid_ba": m_zs_small["centroid_ba"],
                "dev_zs_small_1nn_ba": m_zs_small["1nn_ba"],
                "dev_zs_large_ba": m_zs_large["best_ba"],
                "dev_zs_large_centroid_ba": m_zs_large["centroid_ba"],
                "dev_zs_large_1nn_ba": m_zs_large["1nn_ba"],
                # Legacy compatibility key
                "dev_ba": m_mixed["best_ba"],
            })

            # Save checkpoint based on primary mixed & large performance
            combined_score = 0.6 * m_mixed["best_ba"] + 0.4 * m_zs_large["best_ba"]
            if combined_score > _best_score:
                _best_score = combined_score
                _best_epoch = epoch
                _best_state = copy.deepcopy(model.state_dict())

            if epoch % 10 == 0 or epoch == 1:
                best_marker = " *** BEST ***" if combined_score >= _best_score else ""
                print(
                    f"epoch {epoch:03d}/{epochs}: loss={value:.6f}  "
                    f"Dev-Mixed={m_mixed['best_ba']:.4f}  "
                    f"Dev-ZS-Small={m_zs_small['best_ba']:.4f}  "
                    f"Dev-ZS-Large={m_zs_large['best_ba']:.4f}{best_marker}",
                    flush=True,
                )

            # Early stopping check: if loss hit minimum plateau and score hasn't improved for 25 epochs
            if epoch - _best_epoch >= 25 and value < 0.05:
                print(f"[EARLY STOPPING] Generalization peaked at epoch {_best_epoch} (Score: {_best_score:.4f}). Stopping early to preserve peak checkpoint.", flush=True)
                break
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Caught Ctrl+C! Gracefully restoring best model from epoch {_best_epoch}...", flush=True)
    finally:
        if _best_state is not None:
            model.load_state_dict(_best_state)
            print(f"[BEST CKPT] Restored epoch {_best_epoch} model with best combined generalization score {_best_score:.4f}\n")

    model.eval()
    return history


def compute_sample_diagnostics(
    model: OSLEmbeddingNet,
    all_items: list[dict],
    support: list[dict],
    c0: np.ndarray,
    c1: np.ndarray,
    calibrated_tau: float,
    device: str,
) -> list[dict]:
    """Compute exhaustive per-sample forensic metrics, continuous scores, and band distinctions."""
    support_images = torch.stack([item["image"] for item in support]).to(device)
    support_lbl = np.array([item["label"] for item in support])

    with torch.no_grad():
        sup_emb = model(support_images).cpu().numpy()

    mean_train = sup_emb.mean(axis=0, keepdims=True)
    sup_c = sup_emb - mean_train
    sup_proc = sup_c / np.linalg.norm(sup_c, axis=1, keepdims=True).clip(1e-7)

    support_artifact_ids = {s["artifact_id"] for s in support}

    records = []
    for item in all_items:
        img_t = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img_t).cpu().numpy()

        q_c = emb - mean_train
        q_proc = (q_c / np.linalg.norm(q_c, axis=1, keepdims=True).clip(1e-7)).squeeze(0)

        cos_c0 = float(np.dot(q_proc, c0))
        cos_c1 = float(np.dot(q_proc, c1))
        delta = float(cos_c1 - cos_c0)

        # Continuous normalized probability score (0.0 to 1.0)
        norm_score = float(1.0 / (1.0 + np.exp(-4.0 * (delta - calibrated_tau))))

        # Forensic decision band mapping
        if norm_score < 0.40:
            band = "CLEAN"
        elif norm_score < 0.70:
            band = "SUSPICIOUS"
        else:
            band = "MALICIOUS"

        pred = 1 if delta > calibrated_tau else 0
        true_lbl = int(item["label"])

        # 1-NN nearest neighbor calculation
        dists = np.sum((sup_proc - q_proc[None, :]) ** 2, axis=1)
        nn_idx = int(np.argmin(dists))
        nn_lbl = int(support_lbl[nn_idx])
        nn_dist = float(np.sqrt(dists[nn_idx]))
        nn_base_id = support[nn_idx].get("base_checkpoint_id", "ref")

        if item["artifact_id"] in support_artifact_ids:
            split_name = "train_support"
        elif item.get("base_checkpoint_id") in MIXED_DEV_IDS:
            split_name = "dev_mixed"
        elif item.get("base_checkpoint_id") in ZEROSHOT_SMALL_IDS:
            split_name = "dev_zs_small"
        else:
            split_name = "dev_other"

        records.append({
            "artifact_id": item["artifact_id"],
            "base_checkpoint_id": item.get("base_checkpoint_id", "unknown"),
            "architecture_family": item.get("architecture_family", "unknown"),
            "split": split_name,
            "true_label": true_lbl,
            "pred_label": pred,
            "status": "PASS" if pred == true_lbl else "FAIL",
            "cos_sim_clean": round(cos_c0, 4),
            "cos_sim_stego": round(cos_c1, 4),
            "delta_margin": round(delta, 4),
            "normalized_score": round(norm_score, 4),
            "forensic_band": band,
            "nn_neighbor_base": nn_base_id,
            "nn_neighbor_label": nn_lbl,
            "nn_distance": round(nn_dist, 4),
        })

    return records


def generate_training_charts(
    history: list[dict],
    output_dir: Path,
    final_mixed_ba: float,
    final_zs_small_ba: float,
    final_zs_large_ba: float,
    diagnostic_records: list[dict] | None = None,
) -> None:
    """Generate 5 publication-quality presentation charts for the 3 experiments and band diagnosis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("[CHARTS] matplotlib not installed, skipping chart generation.")
        return

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    losses = [h["triplet_loss"] for h in history]
    trailing = [h.get("trailing5", h["triplet_loss"]) for h in history]
    lrs = [h.get("lr", 1e-4) for h in history]

    ba_mixed = [h.get("dev_mixed_ba", 0.0) for h in history]
    ba_zs_small = [h.get("dev_zs_small_ba", 0.0) for h in history]
    ba_zs_large = [h.get("dev_zs_large_ba", 0.0) for h in history]

    # Shared palette & styling
    COLORS = {
        "loss": "#FF6B6B",
        "trailing": "#4ECDC4",
        "mixed": "#45B7D1",
        "zs_small": "#FFA502",
        "zs_large": "#9B59B6",
        "gate": "#2ECC71",
        "clean_band": "#2ECC71",
        "susp_band": "#F1C40F",
        "mal_band": "#E74C3C",
        "bg": "#1A1A2E",
        "grid": "#16213E",
        "text": "#E0E0E0",
        "accent": "#0F3460",
    }
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["accent"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 12,
    })

    # ── CHART 1: Triplet Loss & LR Schedule ───────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, losses, color=COLORS["loss"], alpha=0.35, linewidth=1, label="Epoch Loss")
    ax1.plot(epochs, trailing, color=COLORS["trailing"], linewidth=2.5, label="Trailing-5 Loss")
    ax1.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Smooth Triplet Loss (Softplus)", fontsize=14, fontweight="bold", color=COLORS["trailing"])
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, color="#F1C40F", linestyle=":", linewidth=1.8, label="Cosine LR Schedule")
    ax2.set_ylabel("Learning Rate", fontsize=12, color="#F1C40F")

    lns = ax1.get_lines() + ax2.get_lines()
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right", framealpha=0.8, facecolor=COLORS["bg"])
    ax1.set_title("Layer 3a — Triplet Loss Convergence (Smooth Softplus + Cosine Annealing)", fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart1_triplet_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {charts_dir / 'chart1_triplet_loss.png'}")

    # ── CHART 2: 3-Experiment Generalization Trajectories ─────────────────────
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(epochs, ba_mixed, color=COLORS["mixed"], linewidth=2.5, marker="o", markersize=3, label="Exp 1: Dev-Mixed (40% Homo / 60% Unseen)")
    ax.plot(epochs, ba_zs_small, color=COLORS["zs_small"], linewidth=2.5, marker="s", markersize=3, label="Exp 2: Dev-ZeroShot-Small (4 Diverse Models)")
    ax.plot(epochs, ba_zs_large, color=COLORS["zs_large"], linewidth=2.5, marker="^", markersize=3, label="Exp 3: Dev-ZeroShot-Large (Full 9-Model Zoo)")

    ax.axhline(y=0.70, color=COLORS["gate"], linestyle="--", linewidth=2, alpha=0.9, label="Enterprise Gate Threshold (0.70)")
    ax.axhline(y=0.50, color=COLORS["loss"], linestyle=":", linewidth=1.5, alpha=0.5, label="Random Guess Baseline (0.50)")

    ax.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax.set_ylabel("Balanced Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Layer 3a — Multi-Experiment Generalization Trajectories (100 Epochs)", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0.35, 1.05)
    ax.legend(loc="lower right", framealpha=0.85, facecolor=COLORS["bg"], fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(charts_dir / "chart2_multi_experiment_dev_ba.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {charts_dir / 'chart2_multi_experiment_dev_ba.png'}")

    # ── CHART 3: Final Experiment Benchmark Bar Comparison ───────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    exps = [
        "Exp 1: Dev-Mixed\n(40% Homo / 60% Unseen)",
        "Exp 2: Dev-ZeroShot-Small\n(4 Diverse Families)",
        "Exp 3: Dev-ZeroShot-Large\n(Full 9-Model Zoo)",
    ]
    scores = [final_mixed_ba * 100, final_zs_small_ba * 100, final_zs_large_ba * 100]
    bar_colors = [COLORS["mixed"], COLORS["zs_small"], COLORS["zs_large"]]
    bars = ax.bar(exps, scores, color=bar_colors, width=0.52, edgecolor="white", linewidth=0.5)

    ax.axhline(y=70.0, color=COLORS["gate"], linestyle="--", linewidth=2, alpha=0.85, label="Gate (70.0%)")
    ax.axhline(y=50.0, color=COLORS["loss"], linestyle=":", linewidth=1.5, alpha=0.5, label="Random (50.0%)")

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        status_tag = "PASS" if score >= 70.0 else "STUDY"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2.0,
            f"{score:.1f}%\n[{status_tag}]",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=COLORS["text"]
        )

    ax.set_ylabel("Final Balanced Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Layer 3a — Benchmark Results Across Experimental Regimes", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.8, facecolor=COLORS["bg"])
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(charts_dir / "chart3_experiment_comparison_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {charts_dir / 'chart3_experiment_comparison_bar.png'}")

    # ── CHART 4: Loss vs Multi-Experiment Generalization ──────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Epoch", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Smooth Triplet Loss", color=COLORS["loss"], fontsize=14, fontweight="bold")
    ln1 = ax1.plot(epochs, trailing, color=COLORS["loss"], linewidth=2.5, label="Trailing Loss")
    ax1.tick_params(axis="y", labelcolor=COLORS["loss"])
    ax1.set_ylim(0, max(trailing) * 1.25)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Balanced Accuracy", color=COLORS["mixed"], fontsize=14, fontweight="bold")
    ln2 = ax2.plot(epochs, ba_mixed, color=COLORS["mixed"], linewidth=2.2, label="Dev-Mixed BA")
    ln3 = ax2.plot(epochs, ba_zs_large, color=COLORS["zs_large"], linewidth=2.2, linestyle="--", label="Dev-ZeroShot-Large BA")
    ax2.axhline(y=0.70, color=COLORS["gate"], linestyle=":", linewidth=1.8, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor=COLORS["mixed"])
    ax2.set_ylim(0.35, 1.05)

    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right", framealpha=0.8, facecolor=COLORS["bg"])
    ax1.set_title("Layer 3a — Training Loss vs Generalization Regimes", fontsize=16, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(charts_dir / "chart4_loss_vs_generalization.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {charts_dir / 'chart4_loss_vs_generalization.png'}")

    # ── CHART 5: Per-Sample Continuous Score & Forensic Band Diagnosis ────────
    if diagnostic_records:
        fig, ax = plt.subplots(figsize=(13, 7))

        # Shaded Forensic Decision Bands
        ax.axhspan(0.00, 0.40, color=COLORS["clean_band"], alpha=0.12, label="Clean Band [0.00 – 0.40]")
        ax.axhspan(0.40, 0.70, color=COLORS["susp_band"], alpha=0.15, label="Suspicious / Gemini Triage Band [0.40 – 0.70]")
        ax.axhspan(0.70, 1.00, color=COLORS["mal_band"], alpha=0.15, label="Malicious Stego Band [0.70 – 1.00]")

        clean_records = [r for r in diagnostic_records if r["true_label"] == 0]
        stego_records = [r for r in diagnostic_records if r["true_label"] == 1]

        rng = np.random.default_rng(42)
        # Plot Clean Samples (X around 1.0)
        x_clean = 1.0 + rng.uniform(-0.12, 0.12, size=len(clean_records))
        y_clean = [r["normalized_score"] for r in clean_records]
        ax.scatter(x_clean, y_clean, color="#2ECC71", edgecolors="white", s=80, alpha=0.85, label=f"True Clean Samples (N={len(clean_records)})", zorder=4)

        # Plot Stego Samples (X around 2.0)
        x_stego = 2.0 + rng.uniform(-0.12, 0.12, size=len(stego_records))
        y_stego = [r["normalized_score"] for r in stego_records]
        ax.scatter(x_stego, y_stego, color="#E74C3C", edgecolors="white", s=80, alpha=0.85, marker="^", label=f"True Stego Samples (N={len(stego_records)})", zorder=4)

        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels(["True Clean Models", "True Stego Models"], fontsize=13, fontweight="bold")
        ax.set_ylabel("Continuous Normalized Stego Score (0.0 to 1.0)", fontsize=13, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.6, 2.4)
        ax.set_title("Layer 3a — Sample Score Distribution & Forensic Band Separation", fontsize=16, fontweight="bold", pad=15)
        ax.axhline(0.40, color="#F1C40F", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axhline(0.70, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.legend(loc="upper left", framealpha=0.9, facecolor=COLORS["bg"], fontsize=10)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(charts_dir / "chart5_score_distribution_bands.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[CHART] Saved: {charts_dir / 'chart5_score_distribution_bands.png'}")

    print(f"\n[CHARTS] All presentation and diagnostic charts saved to {charts_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference-architecture", default="resnet", type=str)
    parser.add_argument("--regime", choices=["homogeneous", "cross_family"], default="homogeneous", help="Training regime")
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--x-bits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    all_items = items_for_records(records, args.x_bits)

    if args.regime == "cross_family":
        print(f"\n[REGIME] Configuring Cross-Family Invariant Benchmark (5 Distinct Architectural Families)")
        support_candidates = [it for it in all_items if it.get("base_checkpoint_id") in CROSS_FAMILY_TRAIN_BASE_IDS]
        support = select_scz_support(support_candidates, "cross_family", per_class=args.per_class, designated_bases=CROSS_FAMILY_TRAIN_BASE_IDS)
        dev_mixed_items = [it for it in all_items if it.get("base_checkpoint_id") in CROSS_HELD_OUT_RESNET_IDS]
        dev_zs_small_items = [it for it in all_items if it.get("base_checkpoint_id") in CROSS_HELD_OUT_DIVERSE_IDS]
        support_artifact_ids = {s["artifact_id"] for s in support}
        dev_zs_large_items = [it for it in all_items if it["artifact_id"] not in support_artifact_ids]
    else:
        print(f"\n[REGIME] Configuring Homogeneous Baseline Benchmark (10 ResNet-Family Checkpoints)")
        support_candidates = [it for it in all_items if it.get("base_checkpoint_id") in HOMO_TRAIN_BASE_IDS]
        if len(support_candidates) < 10:
            support_candidates = [it for it in all_items if it.get("architecture_family") == "resnet"]
        support = select_scz_support(support_candidates, args.reference_architecture, per_class=args.per_class, designated_bases=HOMO_TRAIN_BASE_IDS)
        dev_mixed_items = [it for it in all_items if it.get("base_checkpoint_id") in MIXED_DEV_IDS]
        dev_zs_small_items = [it for it in all_items if it.get("base_checkpoint_id") in ZEROSHOT_SMALL_IDS]
        dev_zs_large_items = [it for it in all_items if it.get("base_checkpoint_id") in ZEROSHOT_LARGE_IDS]

    print(
        f"[EXPERIMENTS SETUP] Regime: {args.regime.upper()}\n"
        f"  Support Set:            {len(support)} samples ({len([s for s in support if s['label']==0])} clean + {len([s for s in support if s['label']==1])} stego)\n"
        f"  Exp 1 (Dev-Mixed):       {len(dev_mixed_items)} samples (Held-Out Architectural Subset)\n"
        f"  Exp 2 (Dev-ZS-Small):    {len(dev_zs_small_items)} samples (Held-Out Diverse Models)\n"
        f"  Exp 3 (Dev-ZS-Large):    {len(dev_zs_large_items)} samples (Full Zero-Shot Held-Out Zoo)"
    )

    model = OSLEmbeddingNet(emb_dim=args.emb_dim, dropout=0.2)
    history = train_episode(
        model,
        support,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        device=args.device,
        cross_family=(args.regime == "cross_family"),
        dev_mixed_items=dev_mixed_items,
        dev_zs_small_items=dev_zs_small_items,
        dev_zs_large_items=dev_zs_large_items,
    )

    support_emb = embeddings(model, support, args.device)
    support_lbl = np.array([s["label"] for s in support])
    insample_c_pred, _, _ = classify_support(support_emb, support_emb, support_lbl, apply_cl2=False)
    insample_ba = balanced_accuracy(support_lbl, insample_c_pred)
    print(f"\n[SANITY] In-Sample Support Centroid BA = {insample_ba:.4f}")

    # Compute final metrics on all 3 experimental splits
    m_mixed = evaluate_split(model, dev_mixed_items, torch.stack([item["image"] for item in support]), support_lbl, args.device)
    m_zs_small = evaluate_split(model, dev_zs_small_items, torch.stack([item["image"] for item in support]), support_lbl, args.device)
    m_zs_large = evaluate_split(model, dev_zs_large_items, torch.stack([item["image"] for item in support]), support_lbl, args.device)

    final_mixed_ba = float(m_mixed["best_ba"])
    final_zs_small_ba = float(m_zs_small["best_ba"])
    final_zs_large_ba = float(m_zs_large["best_ba"])

    # Compute calibrated tau on support set
    mean_train = support_emb.mean(axis=0, keepdims=True)
    sup_c = support_emb - mean_train
    sup_proc = sup_c / np.linalg.norm(sup_c, axis=1, keepdims=True).clip(1e-7)
    c0 = sup_proc[support_lbl == 0].mean(axis=0)
    c0 = c0 / max(np.linalg.norm(c0), 1e-7)
    c1 = sup_proc[support_lbl == 1].mean(axis=0)
    c1 = c1 / max(np.linalg.norm(c1), 1e-7)
    s_sup = np.dot(sup_proc, c1) - np.dot(sup_proc, c0)
    calibrated_tau = find_calibrated_threshold(s_sup[support_lbl == 0], s_sup[support_lbl == 1])

    # ── EXHAUSTIVE SAMPLE-LEVEL DIAGNOSTICS & BAND LOGGING ─────────────────────
    diag_records = compute_sample_diagnostics(
        model, all_items, support, c0, c1, calibrated_tau, args.device
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Tabular CSV with all sample metrics
    csv_path = args.output_dir / "diagnostic_sample_scores.csv"
    if diag_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag_records[0].keys()))
            writer.writeheader()
            writer.writerows(diag_records)
        print(f"[DIAGNOSTICS] Saved complete sample breakdown ({len(diag_records)} samples) to {csv_path}")

    # 2. Save Model Weights & Prototypes
    torch.save(model.state_dict(), args.output_dir / "layer3a_gf_fourpart_embedding_net.pt")

    np.savez(
        args.output_dir / "layer3a_gf_fourpart_prototypes.npz",
        clean_prototype=c0,
        stego_prototype=c1,
        proto_clean=c0,
        proto_stego=c1,
        mean_train=mean_train.squeeze(0),
        support_embeddings=support_emb,
        support_labels=support_lbl,
        threshold=np.array(calibrated_tau),
    )

    # 3. Save Comprehensive Calibration JSON
    calib = {
        "format_version": "layer3a_gf_fourpart_v1",
        "status": "completed",
        "threshold": float(calibrated_tau),
        "insample_balanced_accuracy": float(insample_ba),
        "experiment_1_dev_mixed_ba": final_mixed_ba,
        "experiment_2_dev_zeroshot_small_ba": final_zs_small_ba,
        "experiment_3_dev_zeroshot_large_ba": final_zs_large_ba,
        "forensic_bands": {
            "clean_band_max": 0.40,
            "suspicious_band_min": 0.40,
            "suspicious_band_max": 0.70,
            "malicious_band_min": 0.70,
        },
        # Legacy compatibility keys
        "dev_centroid_balanced_accuracy": final_mixed_ba,
        "dev_1nn_balanced_accuracy": float(m_mixed["1nn_ba"]),
        "reference_architecture": args.reference_architecture,
        "x_bits": args.x_bits,
        "manifest_sha256": manifest_sha256(records),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_diagnostics_summary": {
            "total_samples": len(diag_records),
            "clean_samples": len([r for r in diag_records if r["true_label"] == 0]),
            "stego_samples": len([r for r in diag_records if r["true_label"] == 1]),
            "clean_in_clean_band": len([r for r in diag_records if r["true_label"] == 0 and r["forensic_band"] == "CLEAN"]),
            "stego_in_malicious_band": len([r for r in diag_records if r["true_label"] == 1 and r["forensic_band"] == "MALICIOUS"]),
            "samples_in_suspicious_band": len([r for r in diag_records if r["forensic_band"] == "SUSPICIOUS"]),
        },
        "sample_diagnostics": diag_records,
        "history": history,
    }
    (args.output_dir / "layer3a_gf_fourpart_calibration.json").write_text(json.dumps(calib, indent=2))

    # 4. Generate All 5 Charts (including Forensic Band Scatter Plot)
    generate_training_charts(history, args.output_dir, final_mixed_ba, final_zs_small_ba, final_zs_large_ba, diag_records)

    # ── DETAILED CONSOLE OUTPUT (ASCII SAFE) ───────────────────────────────────
    print("\n" + "=" * 78)
    print("  LAYER 3A — MULTI-EXPERIMENT BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"  Exp 1: Dev-Mixed (40% Homo / 60% Unseen)  : {final_mixed_ba*100:.2f}% Balanced Accuracy  [{'PASS' if final_mixed_ba>=0.70 else 'STUDY'}]")
    print(f"  Exp 2: Dev-ZeroShot-Small (4 Diverse)     : {final_zs_small_ba*100:.2f}% Balanced Accuracy  [{'PASS' if final_zs_small_ba>=0.70 else 'STUDY'}]")
    print(f"  Exp 3: Dev-ZeroShot-Large (Full 9-Model)   : {final_zs_large_ba*100:.2f}% Balanced Accuracy  [Non-blocking Study]")
    print("-" * 78)

    # Print sample predictions for Dev-Mixed using ASCII safe marks
    print("\nDetailed Dev-Mixed Breakdown:")
    dev_imgs = torch.stack([item["image"] for item in dev_mixed_items]).to(args.device)
    with torch.no_grad():
        d_emb = model(dev_imgs).cpu().numpy()
    d_preds, _, _ = classify_support(d_emb, support_emb, support_lbl, apply_cl2=True)
    dev_lbls = np.array([item["label"] for item in dev_mixed_items])

    for i, item in enumerate(dev_mixed_items):
        base_id = item.get("base_checkpoint_id", item.get("architecture", "unknown"))
        art_id = item.get("artifact_id", f"item_{i}")
        label = dev_lbls[i]
        pred = d_preds[i]
        mark = "[PASS]" if pred == label else "[FAIL]"
        print(f"  {mark} {base_id:25s} {art_id:40s} True={label} Pred={pred}")

    print("\n[SUCCESS] Training pipeline completed! Model files, diagnostic CSV, and 5 presentation charts saved in output/.\n")


if __name__ == "__main__":
    main()
