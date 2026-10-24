import os
import sys

class Layer1ModelScan:
    """Layer 1: Rule-based opcode bouncer backed by protectai/modelscan (Apache-2.0).

    Scans model pickle opcodes and Keras/H5 models for known unsafe callables (e.g., os.system, subprocess).
    """

    SUPPORTED_EXTENSIONS = {".pt", ".pkl", ".h5", ".pth", ".bin"}

    def scan(self, filepath: str) -> dict:
        ext = os.path.splitext(filepath)[1].lower()

        # Format coverage boundary check
        if ext == ".safetensors":
            return {
                "layer": 1,
                "verdict": False,
                "issues": [],
                "coverage_note": "modelscan does not support .safetensors — covered downstream by Layer 3",
            }

        if ext not in self.SUPPORTED_EXTENSIONS:
            return {
                "layer": 1,
                "verdict": False,
                "issues": [],
                "coverage_note": f"Unsupported extension {ext} for Layer 1 scanner",
            }

        try:
            from modelscan.modelscan import ModelScan
            scanner = ModelScan()
            results = scanner.scan(filepath)

            issues_list = []
            verdict = False

            # Modelscan returns dictionary or object with issues
            # We parse issues from modelscan output
            issues = results.get("issues", []) if isinstance(results, dict) else getattr(results, "issues", [])

            for issue in issues:
                issue_dict = {
                    "severity": str(getattr(issue, "severity", "UNKNOWN")),
                    "description": str(getattr(issue, "description", "")),
                    "operator": str(getattr(issue, "operator", "")),
                    "module": str(getattr(issue, "module", "")),
                }
                issues_list.append(issue_dict)

                # If any CRITICAL or HIGH issue found, verdict is True (Malicious)
                sev = issue_dict["severity"].upper()
                if "CRITICAL" in sev or "HIGH" in sev or "MEDIUM" in sev:
                    verdict = True

            return {
                "layer": 1,
                "verdict": verdict,
                "issues": issues_list,
                "coverage_note": None,
            }

        except ImportError:
            # Fallback direct pickle opcode checker if modelscan is not yet installed
            return self._fallback_pickle_checker(filepath)
        except Exception as e:
            return {
                "layer": 1,
                "verdict": False,
                "issues": [],
                "coverage_note": f"ModelScan execution error: {str(e)}",
            }

    def _fallback_pickle_checker(self, filepath: str) -> dict:
        """Fallback direct opcode walker if modelscan import fails."""
        import pickletools
        dangerous_callables = {"os.system", "subprocess", "builtins.eval", "builtins.exec", "posix.system"}
        issues = []
        verdict = False

        ext = os.path.splitext(filepath)[1].lower()
        if ext in {".pkl", ".pt", ".pth"}:
            try:
                with open(filepath, "rb") as f:
                    content = f.read()

                # Walk opcodes
                current_module = ""
                for opcode, arg, pos in pickletools.genops(content):
                    if opcode.name in ("GLOBAL", "STACK_GLOBAL"):
                        target = f"{arg}"
                        for danger in dangerous_callables:
                            if danger in target:
                                verdict = True
                                issues.append({
                                    "severity": "CRITICAL",
                                    "description": f"Dangerous callable detected at position {pos}",
                                    "operator": target,
                                    "module": "pickle",
                                })
            except Exception as e:
                pass

        return {
            "layer": 1,
            "verdict": verdict,
            "issues": issues,
            "coverage_note": "Fallback pickle opcode scanner used (modelscan not loaded)",
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        l1 = Layer1ModelScan()
        print(l1.scan(sys.argv[1]))
