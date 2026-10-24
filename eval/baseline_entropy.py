import math
import os
from collections import Counter

class ByteEntropyBaseline:
    """Byte-entropy threshold baseline detector (B5-equivalent).

    Computes Shannon entropy of byte distribution in file and applies threshold.
    """

    def __init__(self, threshold: float = 7.8):
        self.threshold = threshold

    def compute_entropy(self, filepath: str) -> float:
        if not os.path.exists(filepath):
            return 0.0

        with open(filepath, "rb") as f:
            data = f.read()

        if not data:
            return 0.0

        length = len(data)
        counts = Counter(data)
        entropy = 0.0

        for count in counts.values():
            p = count / length
            entropy -= p * math.log2(p)

        return round(entropy, 4)

    def predict(self, filepath: str) -> dict:
        entropy = self.compute_entropy(filepath)
        verdict = entropy >= self.threshold
        return {
            "baseline": "entropy_b5",
            "entropy": entropy,
            "threshold": self.threshold,
            "verdict": verdict
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        be = ByteEntropyBaseline()
        print(be.predict(sys.argv[1]))
