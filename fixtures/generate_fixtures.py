import json
import os
import pickle
import struct
import sys
import numpy as np

# Use PyTorch & Torchvision installed in .venv
try:
    import torch
    import torchvision.models as models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.json")
FIXTURES_DIR = os.path.dirname(__file__)


# Sample payload classes for generating diverse pickle attack opcodes
class DirectSystemPayload:
    def __reduce__(self):
        import os
        return (os.system, ("echo 'SENTINEL_MARKER_DIRECT'",))


class SubprocessPayload:
    def __reduce__(self):
        import subprocess
        return (subprocess.Popen, (["echo", "SENTINEL_SUBPROCESS"],))


class ImportlibPayload:
    def __reduce__(self):
        import importlib
        return (importlib.import_module, ("os",))


class FunctoolsPartialPayload:
    def __reduce__(self):
        import functools
        return (functools.partial, (print, "SENTINEL_PARTIAL_PAYLOAD"))


class OperatorAttrPayload:
    def __reduce__(self):
        import operator
        return (operator.attrgetter, ("system",))


class CodecsDecodePayload:
    def __reduce__(self):
        import codecs
        return (codecs.decode, (b"ZWNobyAnU0VOVElORUwn", "base64"))


class ChainedNestedPayload:
    def __reduce__(self):
        import functools, operator
        return (functools.reduce, (operator.add, [1, 2, 3]))


def create_dummy_h5(filepath, size_kb=10):
    with open(filepath, "wb") as f:
        f.write(b"\x89HDF\r\n\x1a\n" + os.urandom(size_kb * 1024))


def create_dummy_safetensors(filepath, num_tensors=5):
    header = {}
    offset = 0
    for i in range(num_tensors):
        length = 100 * 4
        header[f"weight_layer_{i}"] = {
            "dtype": "F32",
            "shape": [10, 10],
            "data_offsets": [offset, offset + length],
        }
        offset += length

    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(b"\x00" * offset)


def generate_large_fixture_dataset():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    manifest = []
    sample_idx = 0

    print("Generating comprehensive dataset including real PyTorch / torchvision architectures...")

    # 1. BENIGN SAMPLES (Real Torchvision Architectures & State Dicts)
    if HAS_TORCH:
        tv_architectures = [
            ("resnet18", models.resnet18),
            ("mobilenet_v2", models.mobilenet_v2),
            ("squeezenet1_0", models.squeezenet1_0),
            ("alexnet", models.alexnet),
            ("shufflenet_v2_x1_0", models.shufflenet_v2_x1_0),
            ("densenet121", models.densenet121),
        ]

        for arch_name, model_fn in tv_architectures:
            sample_idx += 1
            filename = f"benign_{arch_name}_real.pt"
            filepath = os.path.join(FIXTURES_DIR, filename)

            try:
                m = model_fn(weights=None)  # Real PyTorch architecture state dict
                torch.save(m.state_dict(), filepath)

                manifest.append({
                    "filepath": os.path.relpath(filepath, os.path.dirname(FIXTURES_DIR)),
                    "format": "pt",
                    "ground_truth_label": "BENIGN",
                    "attack_type": "none",
                    "embedding_rate": 0.0,
                    "perturbation_magnitude": 0.0,
                })
            except Exception as e:
                print(f"Error generating {arch_name}: {e}")

    # Additional Benign State Dicts (Linear, CNN, MLP, Transformer Key-Values)
    arch_names = ["linear_model", "mlp_classifier", "cnn_backbone", "resnet_custom", "transformer_encoder", "bert_embeddings"]
    for arch in arch_names:
        for v in range(3):
            sample_idx += 1
            filename = f"benign_{arch}_v{v}.pkl"
            filepath = os.path.join(FIXTURES_DIR, filename)

            state_dict = {}
            num_layers = (v + 1) * 3
            for l in range(num_layers):
                w_key = f"layer_{l}.weight"
                b_key = f"layer_{l}.bias"
                state_dict[w_key] = np.random.randn(10, 10).tolist()
                state_dict[b_key] = np.zeros(10).tolist()

            state_dict["meta"] = {"arch": arch, "version": v, "epoch": 10 * v}

            with open(filepath, "wb") as f:
                pickle.dump(state_dict, f)

            manifest.append({
                "filepath": os.path.relpath(filepath, os.path.dirname(FIXTURES_DIR)),
                "format": "pkl",
                "ground_truth_label": "BENIGN",
                "attack_type": "none",
                "embedding_rate": 0.0,
                "perturbation_magnitude": 0.0,
            })

    # Benign H5 and Safetensors
    for i in range(4):
        h5_path = os.path.join(FIXTURES_DIR, f"benign_keras_model_{i}.h5")
        create_dummy_h5(h5_path, size_kb=15 * (i + 1))
        manifest.append({
            "filepath": os.path.relpath(h5_path, os.path.dirname(FIXTURES_DIR)),
            "format": "h5",
            "ground_truth_label": "BENIGN",
            "attack_type": "none",
            "embedding_rate": 0.0,
            "perturbation_magnitude": 0.0,
        })

        st_path = os.path.join(FIXTURES_DIR, f"benign_hf_model_{i}.safetensors")
        create_dummy_safetensors(st_path, num_tensors=3 + i)
        manifest.append({
            "filepath": os.path.relpath(st_path, os.path.dirname(FIXTURES_DIR)),
            "format": "safetensors",
            "ground_truth_label": "BENIGN",
            "attack_type": "none",
            "embedding_rate": 0.0,
            "perturbation_magnitude": 0.0,
        })

    # 2. MALICIOUS SAMPLES (Pickle-exec / Obfuscated Attack Variations)
    payload_classes = [
        ("pickle_exec_direct_system", DirectSystemPayload),
        ("pickle_exec_direct_subprocess", SubprocessPayload),
        ("pickle_exec_importlib", ImportlibPayload),
        ("pickle_exec_partial", FunctoolsPartialPayload),
        ("pickle_exec_attrgetter", OperatorAttrPayload),
        ("pickle_exec_codecs", CodecsDecodePayload),
        ("pickle_exec_chained", ChainedNestedPayload),
    ]

    for attack_name, payload_cls in payload_classes:
        for v in range(4):
            sample_idx += 1
            ext = "pt" if v % 2 == 0 else "pkl"
            filename = f"malicious_{attack_name}_var{v}.{ext}"
            filepath = os.path.join(FIXTURES_DIR, filename)

            model_wrapper = {
                "weights": [[0.5, 0.5], [0.5, 0.5]],
                "bias": [0.1, 0.1],
                "exploit_hook": payload_cls(),
                "iteration": v,
            }

            with open(filepath, "wb") as f:
                pickle.dump(model_wrapper, f)

            manifest.append({
                "filepath": os.path.relpath(filepath, os.path.dirname(FIXTURES_DIR)),
                "format": ext,
                "ground_truth_label": "MALICIOUS",
                "attack_type": attack_name,
                "embedding_rate": 0.0,
                "perturbation_magnitude": 0.0,
            })

    # Save manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset generation complete: {len(manifest)} total fixtures saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    generate_large_fixture_dataset()
