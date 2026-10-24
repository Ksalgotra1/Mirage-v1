import os
import pickletools
import sys
import zipfile

class Layer1ModelScan:
    """Layer 1: Rule-based opcode bouncer.

    Scans model pickle opcodes for known unsafe callables (e.g., os.system, subprocess, eval, exec).
    """

    SUPPORTED_EXTENSIONS = {".pt", ".pkl", ".h5", ".pth", ".bin"}
    DANGEROUS_CALLABLES = {
        "os.system", "nt.system", "posix.system", "subprocess.Popen", "subprocess.call",
        "subprocess.run", "subprocess.check_output", "builtins.eval", "builtins.exec",
        "builtins.__import__", "importlib.import_module", "posix.popen", "nt.popen"
    }

    def scan(self, filepath: str) -> dict:
        ext = os.path.splitext(filepath)[1].lower()

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
            issues = results.get("issues", []) if isinstance(results, dict) else getattr(results, "issues", [])

            for issue in issues:
                issue_dict = {
                    "severity": str(getattr(issue, "severity", "UNKNOWN")),
                    "description": str(getattr(issue, "description", "")),
                    "operator": str(getattr(issue, "operator", "")),
                    "module": str(getattr(issue, "module", "")),
                }
                issues_list.append(issue_dict)

                sev = issue_dict["severity"].upper()
                if "CRITICAL" in sev or "HIGH" in sev or "MEDIUM" in sev:
                    verdict = True

            if verdict or len(issues_list) > 0:
                return {
                    "layer": 1,
                    "verdict": verdict,
                    "issues": issues_list,
                    "coverage_note": None,
                }
        except Exception:
            pass

        # Robust stack-aware opcode scanner
        return self._fallback_pickle_checker(filepath)

    def _fallback_pickle_checker(self, filepath: str) -> dict:
        """Robust stack-aware direct opcode walker for Pickle and PyTorch zipfiles."""
        issues = []
        verdict = False

        ext = os.path.splitext(filepath)[1].lower()
        if ext in {".pkl", ".pt", ".pth", ".bin"}:
            try:
                content = None
                if zipfile.is_zipfile(filepath):
                    with zipfile.ZipFile(filepath, "r") as z:
                        for name in z.namelist():
                            if name.endswith(".pkl") or name.endswith("/data.pkl") or name == "data.pkl":
                                try:
                                    content = z.read(name)
                                    if content:
                                        break
                                except Exception:
                                    pass

                if content is None:
                    with open(filepath, "rb") as f:
                        content = f.read()

                stack = []
                for opcode, arg, pos in pickletools.genops(content):
                    op_name = opcode.name

                    if op_name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "STRING"):
                        stack.append(str(arg))

                    elif op_name in ("GLOBAL", "STACK_GLOBAL"):
                        target = ""
                        if op_name == "STACK_GLOBAL" and len(stack) >= 2:
                            func = stack.pop()
                            mod = stack.pop()
                            target = f"{mod}.{func}"
                        elif arg is not None:
                            target = str(arg).replace(" ", ".")

                        for danger in self.DANGEROUS_CALLABLES:
                            if danger in target or target in danger:
                                verdict = True
                                issues.append({
                                    "severity": "CRITICAL",
                                    "description": f"Dangerous callable detected at position {pos}: {target}",
                                    "operator": target,
                                    "module": "pickle",
                                })
            except Exception:
                pass

        return {
            "layer": 1,
            "verdict": verdict,
            "issues": issues,
            "coverage_note": "Fallback pickle opcode scanner used" if not verdict else None,
        }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        l1 = Layer1ModelScan()
        print(l1.scan(sys.argv[1]))
