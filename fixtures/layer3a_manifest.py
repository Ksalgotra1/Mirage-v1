"""Dataset contract and split auditing for defensible Layer 3a experiments.

The unit of independence is a *base checkpoint*, not a tensor or patch. Every clean
checkpoint and every attacked derivative shares one ``base_checkpoint_id`` and one
split. This module uses only the standard library so manifests stay inspectable.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

MANIFEST_VERSION = 1
VALID_SPLITS = frozenset({"train", "reference", "dev", "test", "cross_arch_test"})
REQUIRED_FIELDS = frozenset(
    {
        "artifact_id", "base_checkpoint_id", "filepath", "split",
        "architecture_family", "format", "label", "attack", "generator_version",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_sha256(records: Iterable[dict[str, Any]]) -> str:
    """Return a stable digest independent of record ordering."""
    rows = sorted(canonical_json(record) for record in records)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Layer 3a manifest must be a JSON array")
    validate_manifest(raw)
    return raw


def validate_manifest(records: Iterable[dict[str, Any]]) -> None:
    """Enforce group-level splitting and reproducible attack metadata."""
    records = list(records)
    if not records:
        raise ValueError("Layer 3a manifest is empty")

    seen_artifacts: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_archs: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record {index} is missing required fields: {sorted(missing)}")
        artifact_id = record["artifact_id"]
        if artifact_id in seen_artifacts:
            raise ValueError(f"duplicate artifact_id: {artifact_id}")
        seen_artifacts.add(artifact_id)
        if record["split"] not in VALID_SPLITS:
            raise ValueError(f"record {artifact_id} has invalid split")
        if record["label"] not in {"clean", "stego"}:
            raise ValueError(f"record {artifact_id} has invalid label")
        if not isinstance(record["attack"], dict):
            raise ValueError(f"record {artifact_id} attack must be an object")
        attack_type = record["attack"].get("type")
        if record["label"] == "clean" and attack_type != "none":
            raise ValueError(f"clean record {artifact_id} must have attack.type='none'")
        if record["label"] == "stego":
            required_attack = {"type", "x_bits", "payload_class", "payload_seed"}
            missing_attack = required_attack - record["attack"].keys()
            if missing_attack:
                raise ValueError(f"stego record {artifact_id} missing attack fields: {sorted(missing_attack)}")
        group_splits[record["base_checkpoint_id"]].add(record["split"])
        group_archs[record["base_checkpoint_id"]].add(record["architecture_family"])

    mixed_splits = {key: value for key, value in group_splits.items() if len(value) != 1}
    if mixed_splits:
        raise ValueError(f"base checkpoints span splits (leakage): {mixed_splits}")
    mixed_archs = {key: value for key, value in group_archs.items() if len(value) != 1}
    if mixed_archs:
        raise ValueError(f"base checkpoints span architecture families: {mixed_archs}")


def audit_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Create a serializable summary for every training/evaluation report."""
    records = list(records)
    validate_manifest(records)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    arch_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    attack_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split = record["split"]
        groups_by_split[split].add(record["base_checkpoint_id"])
        arch_by_split[split][record["architecture_family"]] += 1
        attack_by_split[split][record["attack"]["type"]] += 1

    intersections = {}
    names = sorted(groups_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            intersections[f"{left}__{right}"] = sorted(groups_by_split[left] & groups_by_split[right])
    return {
        "manifest_version": MANIFEST_VERSION,
        "manifest_sha256": manifest_sha256(records),
        "artifact_count": len(records),
        "base_checkpoint_count": len({r["base_checkpoint_id"] for r in records}),
        "base_checkpoints_by_split": {key: sorted(value) for key, value in sorted(groups_by_split.items())},
        "architecture_artifacts_by_split": {key: dict(value) for key, value in sorted(arch_by_split.items())},
        "attack_artifacts_by_split": {key: dict(value) for key, value in sorted(attack_by_split.items())},
        "split_intersections": intersections,
    }


def write_audit(records: Iterable[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    """Write the immutable split-audit summary alongside a generated corpus."""
    summary = audit_summary(records)
    Path(path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
