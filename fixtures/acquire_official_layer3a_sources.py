"""Acquire the locked official TorchVision Layer 3a source zoo reproducibly.

Each catalog row names an architecture and a non-moving weight enum. The script uses
TorchVision's hash-checked weight download, exports only the state dictionary, and
writes an audited project source registry. It never creates attacks; split assignment
comes from the catalog before any derivative exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
import torchvision.models as models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_catalog(rows: list[dict[str, Any]]) -> None:
    required = {"id", "architecture", "weights", "architecture_family", "split"}
    ids = set()
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"catalog row missing {sorted(missing)}")
        if row["id"] in ids:
            raise ValueError(f"duplicate catalog id: {row['id']}")
        ids.add(row["id"])
        enum = models.get_model_weights(row["architecture"])
        if row["weights"] not in enum.__members__ or row["weights"] == "DEFAULT":
            raise ValueError(f"{row['id']} uses an unavailable or moving weight enum")


def acquire(rows: list[dict[str, Any]], output_dir: Path, registry_path: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry: list[dict[str, Any]] = []
    for row in rows:
        enum = models.get_model_weights(row["architecture"])
        weights = enum[row["weights"]]
        destination = output_dir / f"{row['id']}.pt"
        # TorchVision checks the official filename hash when fetching. Export the
        # resolved state dictionary to our own non-executable project artifact.
        state_dict = weights.get_state_dict(progress=True, check_hash=True)
        torch.save(state_dict, destination)
        registry.append({
            "base_checkpoint_id": row["id"],
            "filepath": destination.resolve().as_posix(),
            "split": row["split"],
            "architecture_family": row["architecture_family"],
            "architecture": row["architecture"],
            "format": "pt",
            "source_sha256": sha256(destination),
            "provenance": {
                "origin": "official TorchVision pretrained weight",
                "license": "TorchVision model weights; see upstream weight metadata",
                "source_url": weights.url,
                "torchvision_version": torchvision.__version__,
                "weight_enum": f"{enum.__name__}.{row['weights']}",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "training": {
                "status": "official_pretrained",
                "training_dataset": "ImageNet-1K",
                "training_run_id": f"official:{enum.__name__}.{row['weights']}",
            },
        })
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("fixtures/official_layer3a_source_catalog.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/layer3a_official_sources"))
    parser.add_argument("--registry", type=Path, default=Path("fixtures/layer3a_official_sources.json"))
    args = parser.parse_args()
    rows = json.loads(args.catalog.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("catalog must be a JSON list")
    validate_catalog(rows)
    registry = acquire(rows, args.output_dir, args.registry)
    print(f"Acquired {len(registry)} audited official source checkpoints")
    print(f"Registry: {args.registry}")


if __name__ == "__main__":
    main()
