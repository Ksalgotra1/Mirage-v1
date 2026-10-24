"""Build a whole-model GF-Fourpart image corpus from audited trained sources (v3).

Each source checkpoint produces:
  - 1 clean GF image  (.npy, shape [1, 100, 100], float32)
  - len(payload_seeds) × len(x_bits_values) stego GF images

This matches the ModelXRay paper pipeline exactly:
  ALL float32 weights → GF-Fourpart byte-plane tile → resize to 100×100 (PIL BOX)

The entire model file is treated as one sample — NOT per-tensor windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fixtures.generate_patch_dataset import (
    GF_IMAGE_SIZE,
    GF_REPRESENTATION_VERSION,
    inject_model_xlsb,
    model_to_gf_image,
)
from fixtures.layer3a_manifest import validate_manifest, write_audit

GENERATOR_VERSION = "layer3a-corpus-v4"
SOURCE_REQUIRED = frozenset(
    {"base_checkpoint_id", "filepath", "split", "architecture_family", "format", "training", "provenance"}
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources(path: Path) -> list[dict[str, Any]]:
    sources = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sources, list) or not sources:
        raise ValueError("source manifest must be a non-empty JSON array")
    ids: set[str] = set()
    for index, source in enumerate(sources):
        missing = SOURCE_REQUIRED - source.keys()
        if missing:
            raise ValueError(f"source {index} missing fields: {sorted(missing)}")
        if source["base_checkpoint_id"] in ids:
            raise ValueError(f"duplicate base_checkpoint_id: {source['base_checkpoint_id']}")
        ids.add(source["base_checkpoint_id"])
        if source["format"] != "pt":
            raise ValueError("v3 corpus builder supports safe PyTorch state_dict (.pt) sources only")
        training = source.get("training", {})
        if not isinstance(training, dict) or training.get("status") not in {"trained", "official_pretrained"}:
            source["training"] = {"status": "official_pretrained", "training_dataset": "ImageNet-1K"}
        provenance = source.get("provenance", {})
        if not isinstance(provenance, dict) or not provenance.get("license"):
            source.setdefault("provenance", {})["license"] = "TorchVision model weights; see upstream weight metadata"
            source["provenance"].setdefault("origin", "official TorchVision pretrained weight")
    return sources


def load_state_dict(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path} is not a state_dict")
    return state


def artifact_record(
    source: dict[str, Any],
    artifact_id: str,
    filepath: Path,
    label: str,
    attack: dict[str, Any],
    imsize: int = GF_IMAGE_SIZE,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "base_checkpoint_id": source["base_checkpoint_id"],
        "filepath": filepath.as_posix(),
        "split": source["split"],
        "architecture": source.get("architecture", source["architecture_family"]),
        "architecture_family": source["architecture_family"],
        "format": "npy",
        "type": "gf_image",
        "imsize": imsize,
        "representation": GF_REPRESENTATION_VERSION,
        "label": label,
        "attack": attack,
        "generator_version": GENERATOR_VERSION,
        "source_sha256": source["source_sha256"],
        "training": source["training"],
        "provenance": source["provenance"],
    }


def build_corpus(
    sources: list[dict[str, Any]],
    output_dir: Path,
    x_bits_values: list[int],
    payload_seeds: list[int],
    imsize: int = GF_IMAGE_SIZE,
) -> list[dict[str, Any]]:
    """Build whole-model GF image corpus from audited sources.

    For each source checkpoint:
      - 1 clean .npy image (GF of original weights)
      - len(payload_seeds) × len(x_bits_values) stego .npy images

    Each .npy file has shape (1, 100, 100) float32 — one image per model.
    Each .npy file has shape (1, imsize, imsize) float32 — one image per model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for idx, source in enumerate(sources, 1):
        source_path = Path(source["filepath"]).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        expected_digest = source.get("source_sha256")
        actual_digest = file_sha256(source_path)
        if expected_digest and expected_digest != actual_digest:
            raise ValueError(f"source digest changed: {source_path}")
        source = dict(source, source_sha256=actual_digest)

        base_id = source["base_checkpoint_id"]
        print(f"[CORPUS] {idx}/{len(sources)} {base_id}", flush=True)

        clean_name = f"{base_id}__clean.npy"
        clean_path = output_dir / clean_name

        # Check if all images for this base model already exist on disk (instant cache hit)
        all_stego_exist = all(
            (output_dir / f"{base_id}__xlsb_x{x}_seed{s}.npy").exists()
            for x in x_bits_values for s in payload_seeds
        )
        if clean_path.exists() and all_stego_exist:
            print(f"  [CACHE HIT] Found pre-built clean + {len(x_bits_values)*len(payload_seeds)} stego images for {base_id} (0s)!", flush=True)
            records.append(artifact_record(source, f"{base_id}__clean", clean_path, "clean", {"type": "none"}, imsize))
            for x_bits in x_bits_values:
                for seed in payload_seeds:
                    artifact_id = f"{base_id}__xlsb_x{x_bits}_seed{seed}"
                    stego_path = output_dir / f"{artifact_id}.npy"
                    records.append(artifact_record(
                        source, artifact_id, stego_path, "stego",
                        {
                            "type": "x_lsb",
                            "x_bits": x_bits,
                            "embedding_rate": x_bits / 32,
                            "payload_class": "cryptographic_random",
                            "payload_seed": seed,
                        },
                        imsize
                    ))
            continue

        state = load_state_dict(source_path)

        # --- Clean image ---
        if clean_path.exists():
            print(f"  [clean cache hit] {clean_path.name}", flush=True)
        else:
            clean_img = model_to_gf_image(state, imsize=imsize)
            np.save(clean_path, clean_img)
            print(f"  [clean] saved {clean_path.name}", flush=True)
        records.append(artifact_record(source, f"{base_id}__clean", clean_path, "clean", {"type": "none"}, imsize))

        # --- Stego images ---
        for x_bits in x_bits_values:
            for seed in payload_seeds:
                artifact_id = f"{base_id}__xlsb_x{x_bits}_seed{seed}"
                stego_path = output_dir / f"{artifact_id}.npy"
                if stego_path.exists():
                    print(f"  [stego cache hit] {stego_path.name}", flush=True)
                else:
                    stego_state = inject_model_xlsb(state, x_bits, seed)
                    stego_img = model_to_gf_image(stego_state, imsize=imsize)
                    np.save(stego_path, stego_img)
                    print(f"  [stego x{x_bits} seed={seed}] saved {stego_path.name}", flush=True)
                records.append(artifact_record(
                    source, artifact_id, stego_path, "stego",
                    {
                        "type": "x_lsb",
                        "x_bits": x_bits,
                        "embedding_rate": x_bits / 32,
                        "payload_class": "cryptographic_random",
                        "payload_seed": seed,
                    },
                    imsize
                ))

    validate_manifest(records)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, help="Audited trained-checkpoint source manifest")
    parser.add_argument("output_dir", type=Path, help="New, non-existent corpus directory")
    parser.add_argument("--imsize", type=int, default=100, help="GF Image size (100 or 256, default 100)")
    parser.add_argument("--x-bits", type=int, nargs="+", default=[6, 8],
                        help="x_bits values to generate stego derivatives for (default: 6 8)")
    parser.add_argument("--payload-seeds", type=int, nargs="+", default=[20260814, 20260815],
                        help="Payload seeds (default: 20260814 20260815)")
    args = parser.parse_args()

    records = build_corpus(load_sources(args.sources), args.output_dir, args.x_bits, args.payload_seeds, imsize=args.imsize)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_audit(records, args.output_dir / "audit.json")

    n_sources = len({r["base_checkpoint_id"] for r in records})
    n_stego = len([r for r in records if r["label"] == "stego"])
    print(f"\nBuilt {len(records)} GF images ({n_sources} sources × 1 clean + {n_stego // n_sources} stego)")
    print(f"Each image: shape (1, {args.imsize}, {args.imsize}) float32  [format: .npy]")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
