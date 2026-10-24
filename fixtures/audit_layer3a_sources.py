"""Verify a Layer 3a source-checkpoint registry before corpus generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    records = json.loads(args.registry.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("registry must be a non-empty JSON list")

    identifiers = set()
    for record in records:
        identifier = record["base_checkpoint_id"]
        if identifier in identifiers:
            raise ValueError(f"duplicate base_checkpoint_id: {identifier}")
        identifiers.add(identifier)
        path = Path(record["filepath"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = record.get("source_sha256")
        actual = sha256(path)
        if expected and actual != expected:
            raise ValueError(f"SHA256 mismatch for {identifier}: expected {expected}, got {actual}")
        print(f"OK {identifier}: {path.name} sha256={actual}")

    print(f"Validated {len(records)} source checkpoints")


if __name__ == "__main__":
    main()
