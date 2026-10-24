# Layer 3a source-corpus protocol (locked v1)

## Decision

The primary Layer 3a dataset is a **versioned zoo of official pretrained CNN
checkpoints used unchanged**. Locally fine-tuned CIFAR-10 checkpoints are retained
only as a clearly labeled controlled robustness ablation; they are not the primary
source corpus and cannot substitute for architecture-diverse evaluation.

## Threat model and unit of independence

The deployment target is an unknown downloaded model checkpoint. A **source
checkpoint**, not a tensor, patch, or payload derivative, is the independent unit.
For each benign source checkpoint, the corpus creates one clean artifact and matched
stego derivatives using the documented X-LSB procedure. All artifacts, tensors, and
patches from that source checkpoint must remain in one split.

The scanner does not claim coverage for formats or model classes excluded from this
protocol. v1 covers official TorchVision float32 image-classification CNN state
dictionaries and their locally generated derivatives. Transformers, quantized models,
and third-party pickled checkpoints are separate future strata, not silently mixed
into v1.

## Source acquisition and provenance

Only official TorchVision named weight enums are accepted in the primary corpus.
For every source, record:

- `base_checkpoint_id`: immutable ID containing family, architecture, and weight enum.
- TorchVision package version, architecture factory, exact weight enum, official
  download URL, and retrieval date.
- Original file SHA-256 and the SHA-256 of the normalized exported state dictionary.
- License/provenance field pointing to the official source.
- Model family, architecture, parameter count, tensor count, tensor names/shapes,
  dtype, and serialization format.

Download integrity must be checked by TorchVision's hash validation where supported,
then audited again with the project registry tool. Source files are never modified in
place; attacks are written as new derivative files.

## Locked source-zoo composition

The first release target is at least **24 independent official checkpoints** across
at least eight CNN families. Suggested strata are below; specific weight enums are
recorded in the registry rather than assumed from a moving `DEFAULT` alias.

| Family | Minimum source checkpoints | Role |
| --- | ---: | --- |
| ResNet / ResNeXt | 4 | train |
| DenseNet | 3 | train |
| MobileNet | 3 | train |
| EfficientNet | 3 | train |
| RegNet | 3 | train |
| ShuffleNet / MNASNet | 3 | train |
| SqueezeNet / VGG | 3 | development-only family |
| ConvNeXt | 2 | final cross-family test-only family |

This table is a minimum source-acquisition target, not a license to tune decisions on
the final test family. If an official checkpoint cannot be loaded or audited, it is
excluded with a registry reason and replaced by another official checkpoint from the
same predeclared family.

## Splits and model selection

1. Assign the split **before** generating any stego derivative.
2. Train/reference sources are drawn only from the six train families.
3. Development uses its isolated family for detector choice, threshold selection,
   calibration, and stopping decisions.
4. The ConvNeXt family is frozen as the final cross-family test. It is never used for
   fitting, reference prototypes, threshold selection, or iterative debugging.
5. In addition to the frozen final test, report leave-one-family-out results across
   each eligible family. This exposes architecture dependence instead of hiding it in
   a pooled score.
6. Never randomly split patches from one checkpoint across train/dev/test. Matched
   clean/stego pairs, all payload rates, and all tensor patches stay together.

## Attack construction and labels

- Clean and stego artifacts are paired from the same source checkpoint.
- Primary results use high-entropy payload bits and the exact recorded attack seed.
  The ASCII sentinel exists only for the UI/disarmament demonstration and is excluded
  from primary training and evaluation.
- Payload settings are predeclared and evaluated per setting; pooled metrics must not
  conceal a weak payload regime. The current implementation's X=8 setting is the
  initial primary operating point. X=6 and X=16 are reported as boundary/robustness
  settings once detector selection is frozen.
- The source model's functional quality is not re-used as a detection label. For an
  official model, the pristine official checkpoint is the clean reference and the
  local derivative is the stego counterpart.

## Fine-tuned models: permitted but secondary

The existing ImageNet-initialized, CIFAR-10 fine-tuned ResNet-18 seed runs are kept
as an ablation stratum. They answer a narrow question: whether Layer 3a is robust to
ordinary SGD/augmentation/seed variation within a common architecture. They may be
added to a separate reported experiment but must not be mixed into the primary score
without a stratum label, and must not influence final-test architecture selection.

## Required reports and gates

Before a detector is called usable, publish for every locked split and payload setting:

- number of independent source checkpoints, families, tensors, and patches;
- TPR, FPR, FNR, balanced accuracy, ROC-AUC where scores exist, and confidence
  intervals computed by resampling **source checkpoints**, not patches;
- per-family results and pooled results;
- calibration/threshold chosen only on development data;
- failure/coverage report for unsupported tensors, formats, and health-gated results;
- manifest and source-registry SHA-256 hashes sufficient to reproduce the exact run.

Any model that emits NaNs, collapses to chance on the development family, depends on
per-query support resampling, or cannot be evaluated with a fixed reference set is
rejected from deployment.

## Current execution order

1. Preserve the completed CIFAR seed-201 run and optionally finish the already-started
   seed-202 run as the ablation pair. Do not schedule more CIFAR seed training.
2. Acquire and audit the official source zoo according to the table.
3. Freeze the source registry and split assignment.
4. Generate a new, versioned clean/stego corpus; do not overwrite the old baseline
   corpus.
5. Train candidate detectors using train sources only; select architecture and
   threshold on development.
6. Freeze all choices, run the final ConvNeXt cross-family test once, and publish
   both per-family and pooled results.
