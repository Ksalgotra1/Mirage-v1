# Layer 3a defensible corpus

`fixtures/build_layer3a_corpus.py` builds the Layer 3a corpus only from audited,
trained `.pt` state dictionaries. The input source manifest is intentionally separate
from the output artifact manifest.

Every source checkpoint needs an immutable `base_checkpoint_id`, a split assigned
before any attack is created, documented training metadata, and a license/provenance
record. A clean artifact plus X-LSB derivatives at X=2, 4, 6, 8, and 16 are generated
inside the same checkpoint group. The default payloads are independently derived,
high-entropy bit streams; the safe ASCII sentinel is reserved for the disarmament demo
and is not part of primary training or test results.

The corpus builder writes `manifest.json` and `audit.json` under a new output directory.
The audit contains the manifest hash and proves all parent-checkpoint split intersections
are empty.

Example:

```bash
python3 fixtures/build_layer3a_corpus.py \
  fixtures/layer3a_official_sources.json \
  data/layer3a_corpus_official_v1
python3 eval/audit_layer3a_manifest.py data/layer3a_corpus_official_v1/manifest.json
```

The locked primary source strategy is documented in
[`LAYER3A_CORPUS_PROTOCOL.md`](LAYER3A_CORPUS_PROTOCOL.md): use a provenance-audited
official pretrained model zoo unchanged, split by parent checkpoint and architecture
family, and reserve a whole family for final cross-family testing. Do not add current
`weights=None` fixtures to this corpus.

## Generating source checkpoints

`colab_train_layer3a_sources.ipynb` is the source of truth for the optional controlled
CIFAR-10 fine-tuning ablation. It exports reproducible state dictionaries with logged
seeds and held-out test accuracy, but it is not the primary corpus-generation path.
The primary path is audited acquisition of unchanged official pretrained checkpoints,
as defined in the locked protocol.

## Primary v1 execution

```bash
# 1. Fetch the locked, official 24-checkpoint source zoo and write its registry.
python3 fixtures/acquire_official_layer3a_sources.py

# 2. Create a fresh paired clean/stego corpus. This never overwrites an existing run.
python3 fixtures/build_layer3a_corpus.py \
  fixtures/layer3a_official_sources.json \
  data/layer3a_corpus_official_v1 \
  --x-bits 8

# 3. Train on train sources and calibrate only on the isolated dev family.
# The command refuses to mark the bundle usable unless it clears its development gate.
python3 detector/train_layer3a_gf.py \
  --manifest data/layer3a_corpus_official_v1/manifest.json
```

Open `colab_train_layer3a_sources.ipynb` in Google Colab, configure its architecture,
split, and seed group, then run it. The notebook downloads CIFAR-10 on first use, writes
audited final state dictionaries, and appends source hashes and training metadata to its
Drive-backed registry. It saves resumable snapshots every 20 epochs and a best-validation
checkpoint whenever validation accuracy improves.
