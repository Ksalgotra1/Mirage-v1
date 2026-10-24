"""
eval/diagnose_checkpoints.py
===============================
Diagnostic evaluation of Layer 3a training checkpoints against the dev split.

Answers three questions:
  1. Which epoch (30 / 60 / 90 / 120) was closest to 0.70 dev balanced accuracy?
  2. Does the embedding space separate by label or by architecture family?
  3. Per-architecture confusion matrix for SqueezeNet, VGG11, VGG13.

Modes
-----
  --manifest-only   Print what dev artifacts are required and their expected paths.
                    Use this first to confirm filenames before downloading.

  --checkpoint-dir  Directory with epoch_030.pt .. epoch_120.pt
  --manifest        Path to manifest_local.json (or manifest.json)
  --corpus-root     Directory that is the parent of the artifact files listed in the
                    manifest. Only needed if filepaths in the manifest are relative.
  --x-bits          Which x_bits stego label to evaluate (default: 8)
  --output-dir      Where to write plots and JSON report (default: eval/diag_output/)

Usage
-----
  # Step 1 — see what dev files are needed (no corpus required):
  python -m eval.diagnose_checkpoints --manifest /path/to/manifest_local.json --manifest-only

  # Step 2 — full diagnostic (needs manifest + dev artifacts):
  python -m eval.diagnose_checkpoints \\
      --manifest /path/to/manifest_local.json \\
      --checkpoint-dir temporary/ \\
      --corpus-root /path/to/corpus_x8_retry1/ \\
      --output-dir eval/diag_output/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.models.gf_contrastive_net import EmbeddingNet
# NOTE: diagnose_checkpoints.py targets the *pre-v4* per-tensor-window trainer
# format (epoch_030.pt … epoch_120.pt from the old metric-loss pipeline).
# The v4 SCZ trainer (detector/train_layer3a_gf.py) produces a different bundle
# (layer3a_gf_fourpart_embedding_net.pt + _prototypes.npz + _calibration.json).
# _load_state_dict and select_gf_windows were removed from detector/layer3a_gf.py
# in the v5 rewrite. If you need to diagnose v4 SCZ bundles, read the calibration
# JSON directly — it already contains the full training history and dev BA.
from fixtures.generate_patch_dataset import tensor_to_gf_fourpart_patch
from fixtures.layer3a_manifest import load_manifest


# ---------------------------------------------------------------------------
# Manifest inspection helpers
# ---------------------------------------------------------------------------

def describe_dev_artifacts(records: list[dict]) -> None:
    """Print a table of the dev artifacts and their expected paths."""
    dev = [r for r in records if r["split"] == "dev"]
    print(f"\n{'='*70}")
    print(f"DEV SPLIT — {len(dev)} artifacts needed for diagnostic")
    print(f"{'='*70}")
    families: dict[str, list] = defaultdict(list)
    for r in dev:
        families[r["architecture_family"]].append(r)
    for fam, recs in sorted(families.items()):
        print(f"\n  Architecture family: {fam}")
        for r in sorted(recs, key=lambda x: (x["base_checkpoint_id"], x["label"])):
            label_tag = f"[{r['label'].upper()}]"
            xbits_tag = f" x_bits={r['attack'].get('x_bits', '-')}" if r["label"] == "stego" else ""
            fp = r["filepath"]
            exists = Path(fp).exists()
            status = "✅ found locally" if exists else "❌ not found locally"
            print(f"    {label_tag:<8} {r['artifact_id']}")
            print(f"             filepath: {fp}")
            print(f"             {status}{xbits_tag}")
    print()


def resolve_artifact_paths(records: list[dict], corpus_root: Path | None) -> list[dict]:
    """
    Resolve artifact filepaths.
    - If corpus_root is given, always resolve by filename only (basename join).
      This handles the case where the manifest contains absolute Colab/Drive paths
      that don't exist locally — we just take the filename and look in corpus_root.
    - Otherwise use the filepath as-is.
    """
    resolved = []
    for r in records:
        r2 = dict(r)
        p = Path(r["filepath"])
        if corpus_root is not None:
            # Always use basename — strips any Colab/Drive prefix
            p = corpus_root / p.name
        r2["_resolved_path"] = str(p)
        resolved.append(r2)
    return resolved


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_dev_items(records: list[dict], x_bits: int) -> list[dict]:
    """Load dev artifacts and extract GF-Fourpart patches."""
    import torch
    from fixtures.generate_patch_dataset import qualifying_tensors

    dev = [r for r in records if r["split"] == "dev"]
    items = []
    for idx, record in enumerate(dev, 1):
        path = Path(record["_resolved_path"])
        if record["label"] == "stego" and int(record["attack"].get("x_bits", -1)) != x_bits:
            print(f"  [{idx}/{len(dev)}] SKIP (x_bits mismatch) {record['artifact_id']}")
            continue
        if not path.exists():
            print(f"  [{idx}/{len(dev)}] MISSING {path} — skipping")
            continue
        print(f"  [{idx}/{len(dev)}] Loading {record['artifact_id']}  ({path.stat().st_size / 1e6:.1f} MB)")
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                print("    ERROR: not a state dict — skipping")
                continue
        except Exception as exc:
            print(f"    ERROR loading: {exc} — skipping")
            continue
        windows = qualifying_tensors(state)  # [(name, tensor), ...]
        print(f"    → {len(windows)} qualifying tensors")
        for tensor_name, tensor in windows:
            max_offsets = max(1, tensor.numel() // (224 * 224))
            for offset in range(min(2, max_offsets)):
                items.append({
                    "patch": tensor_to_gf_fourpart_patch(tensor, offset),
                    "label": int(record["label"] == "stego"),
                    "base_checkpoint_id": record["base_checkpoint_id"],
                    "architecture_family": record["architecture_family"],
                    "artifact_id": record["artifact_id"],
                    "tensor_name": tensor_name,
                    "offset": offset,
                })
    return items



# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> tuple[EmbeddingNet, list[dict], int]:
    """Load a training checkpoint. Returns (model, history, epoch)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Not a valid Layer 3a checkpoint: {path}")
    model = EmbeddingNet(emb_dim=64)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("history", []), int(ckpt["epoch"])


def compute_fixed_prototypes(
    model: EmbeddingNet, dev_items: list[dict], n_sources: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build prototypes from the first n_sources clean and stego dev items
    (same logic as the trainer: fixed reference approach).
    This is a diagnostic approximation — production uses train-split references.
    """
    source_ids = sorted({it["base_checkpoint_id"] for it in dev_items})[:n_sources]
    chosen = [it for it in dev_items if it["base_checkpoint_id"] in source_ids]
    patches = torch.stack([it["patch"] for it in chosen])
    labels = np.array([it["label"] for it in chosen])
    with torch.no_grad():
        emb = model(patches).cpu().numpy()
    proto_clean = emb[labels == 0].mean(axis=0)
    proto_stego = emb[labels == 1].mean(axis=0)
    return proto_clean, proto_stego


def score_checkpoint(
    model: EmbeddingNet,
    dev_items: list[dict],
    top_fraction: float = 0.20,
    n_proto_sources: int = 1,  # use 1 source for prototypes, evaluate on remainder
) -> dict:
    """
    Score all dev groups and compute balanced accuracy.
    Returns a rich dict of per-architecture results.
    """
    # Build prototypes from first source; evaluate all (approximation for diagnostic)
    proto_clean, proto_stego = compute_fixed_prototypes(model, dev_items, n_proto_sources)
    gap = float(np.linalg.norm(proto_clean - proto_stego))

    # Group by (base_checkpoint_id, label)
    grouped: dict[tuple, list] = defaultdict(list)
    for it in dev_items:
        grouped[(it["base_checkpoint_id"], it["label"], it["architecture_family"])].append(it)

    source_scores = []
    with torch.no_grad():
        for (src_id, label, arch_fam), rows in sorted(grouped.items()):
            patches = torch.stack([r["patch"] for r in rows])
            emb = model(patches).cpu().numpy()
            margins = (
                np.linalg.norm(emb - proto_clean[None, :], axis=1) -
                np.linalg.norm(emb - proto_stego[None, :], axis=1)
            )
            top_count = max(1, int(np.ceil(len(margins) * top_fraction)))
            score = float(np.mean(np.partition(margins, -top_count)[-top_count:]))
            source_scores.append({
                "base_checkpoint_id": src_id,
                "architecture_family": arch_fam,
                "label": label,
                "score": score,
                "n_patches": len(rows),
            })

    # Find best threshold
    values = sorted({s["score"] for s in source_scores})
    candidates = [values[0] - 1e-6, values[-1] + 1e-6] + [
        (a + b) / 2 for a, b in zip(values, values[1:])
    ]
    best_bal_acc, best_thresh = -1.0, 0.0
    for thresh in candidates:
        y = np.array([s["label"] for s in source_scores])
        p = np.array([s["score"] >= thresh for s in source_scores])
        tpr = p[y == 1].mean() if (y == 1).any() else 0.0
        tnr = (~p[y == 0]).mean() if (y == 0).any() else 0.0
        bal = float((tpr + tnr) / 2)
        if bal > best_bal_acc:
            best_bal_acc, best_thresh = bal, thresh

    # Per-architecture breakdown
    arch_results: dict[str, dict] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for s in source_scores:
        pred = s["score"] >= best_thresh
        gt = s["label"]
        arch = s["architecture_family"]
        if gt == 1 and pred:
            arch_results[arch]["tp"] += 1
        elif gt == 0 and not pred:
            arch_results[arch]["tn"] += 1
        elif gt == 0 and pred:
            arch_results[arch]["fp"] += 1
        else:
            arch_results[arch]["fn"] += 1

    return {
        "balanced_accuracy": best_bal_acc,
        "threshold": best_thresh,
        "prototype_gap": gap,
        "source_scores": source_scores,
        "arch_results": dict(arch_results),
    }


# ---------------------------------------------------------------------------
# PCA embedding visualisation
# ---------------------------------------------------------------------------

def plot_embeddings(model: EmbeddingNet, dev_items: list[dict], epoch: int, output_dir: Path) -> None:
    """2D PCA of all dev patch embeddings, colored by label and architecture."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("  [PLOT] matplotlib or sklearn not available — skipping PCA plot")
        return

    patches = torch.stack([it["patch"] for it in dev_items])
    labels = np.array([it["label"] for it in dev_items])
    archs = [it["architecture_family"] for it in dev_items]

    with torch.no_grad():
        emb = model(patches).cpu().numpy()

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left: colored by label
    for lbl, marker, color, name in [(0, "o", "#4CAF50", "clean"), (1, "^", "#F44336", "stego")]:
        mask = labels == lbl
        axes[0].scatter(coords[mask, 0], coords[mask, 1], c=color, marker=marker,
                        alpha=0.7, s=40, label=name, edgecolors="white", linewidths=0.3)
    axes[0].set_title(f"Epoch {epoch:03d} — embeddings by label")
    axes[0].legend(framealpha=0.8)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Right: colored by architecture family
    unique_archs = sorted(set(archs))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(unique_archs), 1)))
    for i, arch in enumerate(unique_archs):
        mask = np.array([a == arch for a in archs])
        clean_m = mask & (labels == 0)
        stego_m = mask & (labels == 1)
        axes[1].scatter(coords[clean_m, 0], coords[clean_m, 1], color=palette[i], marker="o",
                        alpha=0.7, s=40, edgecolors="white", linewidths=0.3, label=f"{arch} (clean)")
        axes[1].scatter(coords[stego_m, 0], coords[stego_m, 1], color=palette[i], marker="^",
                        alpha=0.7, s=50, edgecolors="white", linewidths=0.3, label=f"{arch} (stego)")
    axes[1].set_title(f"Epoch {epoch:03d} — embeddings by architecture")
    axes[1].legend(fontsize=7, framealpha=0.8, ncol=2)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    plt.suptitle("Layer 3a Diagnostic: Dev Embedding Space", fontweight="bold")
    fig.tight_layout()
    out = output_dir / f"embedding_pca_epoch{epoch:03d}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Saved {out}")


def plot_loss_curves(all_histories: dict[int, list[dict]], output_dir: Path) -> None:
    """Overlay loss curves for all 4 checkpoints."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    colors = ["#5B5CE2", "#E25B5B", "#5BE2A0", "#E2A05B"]
    for (epoch, history), color in zip(sorted(all_histories.items()), colors):
        if history:
            epochs = [row["epoch"] for row in history]
            losses = [row["metric_loss"] for row in history]
            ax.plot(epochs, losses, color=color, linewidth=2, label=f"up to epoch {epoch}")
            ax.axvline(epoch, color=color, linestyle="--", alpha=0.4, linewidth=1)

    ax.axhline(0.53, color="#999", linestyle=":", linewidth=1, label="plateau (~0.53)")
    ax.set_title("Layer 3a GF-Fourpart — training loss (all checkpoints)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pairwise metric loss")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = output_dir / "loss_curves_all_checkpoints.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Saved {out}")


def print_summary_table(results: dict[int, dict]) -> None:
    """Print a ranked summary table of all checkpoint results."""
    print(f"\n{'='*70}")
    print(f"{'CHECKPOINT DIAGNOSTIC SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Epoch':>8}  {'Dev Bal-Acc':>12}  {'Proto Gap':>10}  {'Threshold':>12}  {'Gate':>6}")
    print(f"{'-'*8:>8}  {'-'*12:>12}  {'-'*10:>10}  {'-'*12:>12}  {'-'*6:>6}")
    ranked = sorted(results.items(), key=lambda x: x[1]["balanced_accuracy"], reverse=True)
    for epoch, r in ranked:
        bal = r["balanced_accuracy"]
        gate_str = "✅ PASS" if bal >= 0.70 else ("⚠️ CLOSE" if bal >= 0.65 else "❌ FAIL")
        print(f"  {epoch:>6}    {bal:>10.4f}    {r['prototype_gap']:>9.4f}    {r['threshold']:>11.6f}    {gate_str}")

    best_epoch, best_r = ranked[0]
    print(f"\n  Best epoch: {best_epoch}  →  dev balanced accuracy = {best_r['balanced_accuracy']:.4f}")
    if best_r["balanced_accuracy"] >= 0.70:
        print(f"  ✅ Epoch {best_epoch} PASSES the 0.70 gate — bundle can be exported from this checkpoint.")
    else:
        print(f"  ❌ No checkpoint passes 0.70. Gap to gate: {0.70 - best_r['balanced_accuracy']:.4f}")
        print("     → Retraining required with improved architecture/data.")

    print(f"\n{'='*70}")
    print("PER-ARCHITECTURE BREAKDOWN (best epoch)")
    print(f"{'='*70}")
    for arch, counts in best_r.get("arch_results", {}).items():
        tp = counts.get("tp", 0)
        tn = counts.get("tn", 0)
        fp = counts.get("fp", 0)
        fn = counts.get("fn", 0)
        total = tp + tn + fp + fn
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        print(f"\n  {arch}")
        print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}  (n={total})")
        print(f"    TPR (sensitivity): {tpr:.3f}  |  TNR (specificity): {tnr:.3f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Path to manifest_local.json (or manifest.json)")
    parser.add_argument("--corpus-root", type=Path, default=None,
                        help="Root directory of the corpus (parent of artifact files). "
                             "Only needed if manifest filepaths are relative.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("temporary"),
                        help="Directory containing epoch_030.pt .. epoch_120.pt (default: temporary/)")
    parser.add_argument("--x-bits", type=int, default=8,
                        help="x_bits value to evaluate (default: 8)")
    parser.add_argument("--top-fraction", type=float, default=0.20,
                        help="Top-fraction for score aggregation (default: 0.20)")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/diag_output"),
                        help="Directory for output plots and JSON (default: eval/diag_output/)")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only print required dev artifact paths, then exit.")
    args = parser.parse_args()

    print("\nLayer 3a Checkpoint Diagnostic")
    print(f"  manifest:       {args.manifest}")
    print(f"  checkpoint-dir: {args.checkpoint_dir}")
    print(f"  x_bits:         {args.x_bits}")
    print(f"  output-dir:     {args.output_dir}")

    print("\n[1/5] Loading manifest...")
    records = load_manifest(args.manifest)
    records = resolve_artifact_paths(records, args.corpus_root)
    describe_dev_artifacts(records)

    if args.manifest_only:
        print("--manifest-only mode: exiting without loading checkpoints or corpus files.")
        return

    print(f"[2/5] Extracting dev patches (x_bits={args.x_bits})...")
    dev_items = extract_dev_items(records, args.x_bits)
    if not dev_items:
        print("ERROR: No dev items extracted. Check corpus paths and manifest.")
        sys.exit(1)
    print(f"  → {len(dev_items)} dev patch items across "
          f"{len({it['base_checkpoint_id'] for it in dev_items})} source checkpoints")

    print(f"\n[3/5] Scanning checkpoints in {args.checkpoint_dir}...")
    checkpoint_paths = sorted(args.checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoint_paths:
        print(f"ERROR: No epoch_*.pt files found in {args.checkpoint_dir}")
        sys.exit(1)
    print(f"  Found: {[p.name for p in checkpoint_paths]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[int, dict] = {}
    all_histories: dict[int, list[dict]] = {}

    for ckpt_path in checkpoint_paths:
        print(f"\n  Evaluating {ckpt_path.name}...")
        model, history, epoch = load_checkpoint(ckpt_path)
        all_histories[epoch] = history
        result = score_checkpoint(model, dev_items, top_fraction=args.top_fraction)
        all_results[epoch] = result
        print(f"    dev_balanced_accuracy = {result['balanced_accuracy']:.4f}  "
              f"(gap={result['prototype_gap']:.4f}, thresh={result['threshold']:.6f})")
        print(f"[4/5] Generating embedding PCA for epoch {epoch}...")
        plot_embeddings(model, dev_items, epoch, args.output_dir)

    print("\n[5/5] Writing summary...")
    plot_loss_curves(all_histories, args.output_dir)
    print_summary_table(all_results)

    # Save JSON report
    report_path = args.output_dir / "checkpoint_diagnostic_report.json"
    report = {
        "x_bits": args.x_bits,
        "top_fraction": args.top_fraction,
        "manifest": str(args.manifest),
        "checkpoints_evaluated": len(all_results),
        "results_by_epoch": {
            str(epoch): {
                "balanced_accuracy": r["balanced_accuracy"],
                "threshold": r["threshold"],
                "prototype_gap": r["prototype_gap"],
                "arch_results": r["arch_results"],
                "source_scores": r["source_scores"],
            }
            for epoch, r in all_results.items()
        }
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  JSON report: {report_path}")
    print(f"  Plots:       {args.output_dir}/")


if __name__ == "__main__":
    main()
