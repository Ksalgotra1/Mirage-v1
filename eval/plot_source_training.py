"""Create presentation charts from Layer 3a source-training logs.

Usage: python eval/plot_source_training.py --log path/to/training.log --output-dir eval/charts
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+):\s+train_loss=([0-9.]+)\s+test_accuracy=([0-9.]+)")
SEED_RE = re.compile(r"seed(\d+)")
START_RE = re.compile(r"starting checkpoint .*seed(\d+)", re.IGNORECASE)


def parse_log(path: Path) -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {}
    current = "seed101"
    previous_epoch = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start_match = START_RE.search(line)
        if start_match:
            current = f"seed{start_match.group(1)}"
        match = EPOCH_RE.search(line)
        if match:
            epoch = int(match.group(1))
            # Pasted multi-run logs may omit the explicit seed header. Detect
            # the epoch counter resetting and assign the next seed label.
            if previous_epoch is not None and epoch < previous_epoch:
                current_seed = re.search(r"seed(\d+)", current)
                current = f"seed{int(current_seed.group(1)) + 1}" if current_seed else f"run{len(series)+1}"
            series.setdefault(current, []).append({
                "epoch": float(epoch),
                "loss": float(match.group(3)),
                "accuracy": float(match.group(4)),
            })
            previous_epoch = epoch
        seed_match = SEED_RE.search(line)
        if seed_match:
            current = f"seed{seed_match.group(1)}"
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/charts"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    series = parse_log(args.log)
    if not series:
        raise SystemExit("No epoch summary lines found in log")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    colors = ["#9B8AFB", "#38D9A9", "#FFB86B", "#7DCFFF", "#FF7AB6", "#C3E88D"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#0A0A0B")
    fig.suptitle("PROJECT-CLIMAX · Source Model Training", fontsize=19, fontweight="bold", color="#F4F1FF")
    for index, (seed, values) in enumerate(sorted(series.items())):
        x = [row["epoch"] for row in values]
        axes[0].plot(x, [row["loss"] for row in values], marker="o", linewidth=2.5, markersize=4, color=colors[index % len(colors)], label=seed)
        axes[1].plot(x, [100 * row["accuracy"] for row in values], marker="o", linewidth=2.5, markersize=4, color=colors[index % len(colors)], label=seed)
    axes[0].set_title("Optimization trajectory", color="#F4F1FF", pad=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].set_title("Held-out classification accuracy", color="#F4F1FF", pad=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "source_training_curves.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = []
    for seed, values in sorted(series.items()):
        best = max(values, key=lambda row: row["accuracy"])
        summary.append((seed, 100 * best["accuracy"], int(best["epoch"])))
    summary.sort(key=lambda item: item[1], reverse=True)
    fig, axis = plt.subplots(figsize=(11, 6), facecolor="#0A0A0B")
    bars = axis.bar([item[0] for item in summary], [item[1] for item in summary], color=colors[:len(summary)], width=0.62)
    axis.set_title("Best held-out accuracy by source checkpoint", fontsize=18, fontweight="bold", color="#F4F1FF", pad=16)
    axis.set_ylabel("Best accuracy (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.16)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, (_, score, epoch) in zip(bars, summary):
        axis.text(bar.get_x() + bar.get_width() / 2, score + 2, f"{score:.1f}%\nE{epoch}", ha="center", va="bottom", color="#F4F1FF", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.output_dir / "source_checkpoint_quality.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {args.output_dir / 'source_training_curves.png'}")
    print(f"Wrote {args.output_dir / 'source_checkpoint_quality.png'}")


if __name__ == "__main__":
    main()
