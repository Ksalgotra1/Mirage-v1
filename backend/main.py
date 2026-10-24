import collections
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Add parent dir to sys.path to import detector and eval modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detector.layer1_modelscan import Layer1ModelScan
from detector.layer2_classifier import Layer2Classifier
from backend.fusion import FusionEngine
from backend.summarizer import summarize_findings
from backend.session_store import session_store
from backend.scan_jobs import ScanJob, scan_job_store
from detector.disarmament_engine import (
    disarmament_engine,
    DisarmamentVerificationError,
    DisarmamentNotPossibleError,
)

# ---------------------------------------------------------------------------
# Rate limiter — 10 /disarm requests per IP per hour
# ---------------------------------------------------------------------------
_DISARM_RATE_LIMIT = 10
_DISARM_RATE_WINDOW = 3600  # seconds
_rate_counters: dict[str, collections.deque] = {}


def _check_rate_limit(ip: str) -> None:
    """Raise HTTP 429 if the IP has exceeded the disarm rate limit."""
    now = time.time()
    window_start = now - _DISARM_RATE_WINDOW
    dq = _rate_counters.setdefault(ip, collections.deque())
    # Drop timestamps outside the current window
    while dq and dq[0] < window_start:
        dq.popleft()
    if len(dq) >= _DISARM_RATE_LIMIT:
        retry_after = int(_DISARM_RATE_WINDOW - (now - dq[0]))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {_DISARM_RATE_LIMIT} disarm requests per hour. "
                   f"Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
    dq.append(now)

try:
    from detector.forensic_signals import ForensicSignalEngine
    _forensic_engine = ForensicSignalEngine()
except Exception:
    _forensic_engine = None

app = FastAPI(
    title="Ensemble ML-Model Malware/Backdoor Detector API",
    description="Multi-layered detector for model files (.pt, .pkl, .h5, .safetensors)",
    version="1.0.0",
)

# Enable CORS for dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

l1_scanner = Layer1ModelScan()
l2_classifier = Layer2Classifier()
fusion_engine = FusionEngine()
_ASYNC_STAGE_MIN_SECONDS = 1.5


def _run_forensic_analysis(filepath: str, forensic_band: str) -> dict | None:
    """Run the band-appropriate byte investigation, or skip clean findings."""
    if forensic_band == "clean":
        return None
    if forensic_band not in ("suspicious", "malicious"):
        return {"error": f"Unknown forensic band: {forensic_band}", "band": forensic_band}
    if _forensic_engine is None:
        return {"error": "Forensic signal engine unavailable", "band": forensic_band}
    try:
        return _forensic_engine.analyze(filepath, band=forensic_band)
    except Exception as exc:
        return {"error": f"Forensic engine error: {exc}", "band": forensic_band}


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ensemble-model-detector"}


@app.post("/scan")
def scan_model(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_model"
    ext = os.path.splitext(filename)[1].lower()

    if not ext:
        raise HTTPException(status_code=400, detail="Uploaded file missing extension")

    # Write uploaded file to temporary location for scanning
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        tmp_path = tmp.name

    try:
        return JSONResponse(content=_execute_scan(tmp_path, filename))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _execute_scan(filepath: str, filename: str, job: ScanJob | None = None) -> dict:
    """Run the existing pipeline and publish stage events when a job is supplied."""
    ext = os.path.splitext(filename)[1].lower()

    started_at: dict[str, float] = {}

    def emit(stage: str, status: str, message: str, metadata: dict | None = None) -> None:
        if job:
            job.emit(stage, status, message, metadata)

    def start(stage: str, message: str, metadata: dict | None = None) -> None:
        started_at[stage] = time.monotonic()
        emit(stage, "started", message, metadata)

    def complete(stage: str, message: str, metadata: dict | None = None) -> None:
        if job and stage in started_at:
            remaining = _ASYNC_STAGE_MIN_SECONDS - (time.monotonic() - started_at[stage])
            if remaining > 0:
                time.sleep(remaining)
        emit(stage, "completed", message, metadata)

    start("layer1", "Tracing serialized opcode execution paths")
    l1_failed = False
    try:
        l1_res = l1_scanner.scan(filepath)
    except Exception:
        l1_failed = True
        l1_res = {"layer": 1, "verdict": False, "issues": [], "coverage_note": "Layer 1 execution failure"}
    complete("layer1", "Opcode inspection complete", {"flagged": bool(l1_res.get("verdict")), "outcome": "failed" if l1_failed else "flagged" if l1_res.get("verdict") else "clear"})

    start("layer2", "Profiling statistical opcode grammar")
    l2_failed = False
    try:
        l2_res = l2_classifier.predict(filepath)
    except Exception:
        l2_failed = True
        l2_res = {"layer": 2, "verdict": False, "probability": 0.0, "top_features": [], "coverage_note": "Layer 2 execution failure"}
    complete("layer2", "Statistical grammar profile complete", {"flagged": bool(l2_res.get("verdict")), "outcome": "failed" if l2_failed else "flagged" if l2_res.get("verdict") else "clear"})

    start("layer3a", "Fingerprinting tensor byte planes")
    l3a_failed = False
    try:
        l3a_res = _run_layer3a_detector(filepath, ext)
    except Exception:
        l3a_failed = True
        l3a_res = {"layer": "3a", "verdict": False, "distance": 0.0, "threshold": 0.05, "score": 0.0, "forensic_band": "clean", "coverage_note": "Layer 3a execution failure"}
    forensic_band = l3a_res.get("forensic_band", "clean")
    complete("layer3a", "Steganographic signal classified", {
        "flagged": bool(l3a_res.get("verdict")), "score": l3a_res.get("score", 0.0), "forensic_band": forensic_band,
        "outcome": "failed" if l3a_failed else "flagged" if l3a_res.get("verdict") else "elevated" if forensic_band == "suspicious" else "clear",
    })

    if forensic_band == "clean":
        forensic_signals = None
        emit("forensics", "skipped", "No byte forensics required for the clean band", {"forensic_band": forensic_band, "outcome": "skipped"})
    else:
        start("forensics", "Running full byte forensics", {"forensic_band": forensic_band, "analysis_depth": "full"})
        forensic_signals = _run_forensic_analysis(filepath, forensic_band)
        complete("forensics", "Full forensic byte analysis complete", {"forensic_band": forensic_band, "analysis_depth": "full", "outcome": "flagged" if forensic_band == "malicious" else "elevated"})

    start("layer3b", "Inspecting spectral weight geometry")
    l3b_failed = False
    try:
        l3b_res = _run_layer3b_detector(filepath, ext)
    except Exception:
        l3b_failed = True
        l3b_res = {"layer": "3b", "verdict": False, "score": 0.0, "flagged_layers": [], "coverage_note": "Layer 3b execution failure"}
    complete("layer3b", "Spectral weight inspection complete", {"flagged": bool(l3b_res.get("verdict")), "outcome": "failed" if l3b_failed else "flagged" if l3b_res.get("verdict") else "clear"})

    start("fusion", "Correlating threat signals across defense layers")
    fused_res = fusion_engine.fuse(ext, l1_res, l2_res, l3a_res, l3b_res)
    fused_outcome = "flagged" if fused_res.get("verdict") == "MALICIOUS" else "elevated" if fused_res.get("verdict") == "SUSPICIOUS" else "clear"
    complete("fusion", "Threat signal correlation complete", {"verdict": fused_res.get("verdict"), "score": fused_res.get("score"), "outcome": fused_outcome})

    disarmament_info = None
    any_layer_flagged = any(result.get("verdict") for result in (l1_res, l2_res, l3a_res, l3b_res))
    scan_result_dict = {"layer1": l1_res, "layer2": l2_res, "layer3a": l3a_res, "layer3b": l3b_res, "fused": fused_res}
    if fused_res.get("verdict") == "MALICIOUS" or any_layer_flagged:
        disarm_scan_id = session_store.save(filepath, scan_result_dict)
        disarmament_info = {"status": "READY_FOR_DISARMAMENT", "scan_id": disarm_scan_id, "expires_in_seconds": 3600, "endpoint": "POST /disarm"}

    response = {"filename": filename, "format": ext.lstrip("."), "layer1": l1_res, "layer2": l2_res, "layer3a": l3a_res, "layer3b": l3b_res, "fused": fused_res, "steganography_forensics": forensic_signals, "disarmament": disarmament_info}
    start("summary", "Assembling incident evidence and remediation guidance")
    try:
        response["summary"] = summarize_findings(response)
    except Exception:
        response["summary"] = {"summary": "Summary unavailable.", "provider": "error"}
    complete("summary", "Incident dossier assembled", {"outcome": fused_outcome})
    return response


def _run_scan_job(job: ScanJob) -> None:
    try:
        job.emit("upload", "started", "Securing uploaded model for inspection")
        time.sleep(_ASYNC_STAGE_MIN_SECONDS)
        job.emit("upload", "completed", "Model received; initializing inspection", {"outcome": "clear"})
        result = _execute_scan(job.filepath, job.filename, job)
        job.emit("output", "started", "Preparing verified scan output")
        time.sleep(_ASYNC_STAGE_MIN_SECONDS)
        output_outcome = "flagged" if result.get("fused", {}).get("verdict") == "MALICIOUS" else "elevated" if result.get("fused", {}).get("verdict") == "SUSPICIOUS" else "clear"
        job.emit("output", "completed", "Scan output is ready", {"result": result, "verdict": result.get("fused", {}).get("verdict"), "score": result.get("fused", {}).get("score"), "outcome": output_outcome})
        job.finish(result)
    except Exception:
        job.emit("output", "failed", "The scan could not be completed")
        job.finish({"detail": "The scan could not be completed"})
    finally:
        if os.path.exists(job.filepath):
            os.remove(job.filepath)


@app.post("/scans", status_code=202)
def create_scan_job(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_model"
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        raise HTTPException(status_code=400, detail="Uploaded file missing extension")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        filepath = tmp.name
    job = scan_job_store.create(filename, filepath)
    threading.Thread(target=_run_scan_job, args=(job,), daemon=True).start()
    return {"scan_id": job.scan_id, "stream_url": f"/scans/{job.scan_id}/events"}


@app.get("/scans/{scan_id}/events")
def scan_events(scan_id: str, request: Request, after: int | None = None):
    job = scan_job_store.get(scan_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Scan job not found or expired")
    try:
        last_seen = int(request.headers.get("last-event-id", after or 0))
    except ValueError:
        last_seen = after or 0

    def event_stream():
        nonlocal last_seen
        while True:
            with job.condition:
                pending = [event for event in job.events if event["id"] > last_seen]
                terminal = job.terminal
                if not pending and not terminal:
                    job.condition.wait(timeout=15)
                    pending = [event for event in job.events if event["id"] > last_seen]
                    terminal = job.terminal
            if not pending:
                if terminal:
                    return
                yield ": heartbeat\n\n"
                continue
            for event in pending:
                last_seen = event["id"]
                event_name = "complete" if event["stage"] == "output" and event["status"] == "completed" else "failed" if event["status"] == "failed" else "scan_update"
                yield f"id: {event['id']}\nevent: {event_name}\ndata: {json.dumps(event)}\n\n"
            if terminal:
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.post("/disarm")
def disarm_model(
    request: Request,
    scan_id: str = Form(...),
    confirm: bool = Form(False),
):
    """
    Disarm a previously scanned malicious model file.

    Requires:
      - scan_id: returned by POST /scan when verdict is MALICIOUS
      - confirm: must be set to true to proceed

    Returns:
      Streaming download of the cleaned file, with forensic evidence in headers.

    Rate limit: 10 requests per IP per hour.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    # Purge stale sessions on every call (cheap, no background thread needed)
    session_store.purge_expired()

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Disarmament requires explicit confirmation. Set confirm=true to proceed.",
        )

    session = session_store.get(scan_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"scan_id '{scan_id}' not found or expired. Sessions last 1 hour.",
        )

    try:
        report = disarmament_engine.disarm(session.filepath, session.scan_result)
    except DisarmamentNotPossibleError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"File requires manual review — automated disarmament failed: {exc}",
        )
    except DisarmamentVerificationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Disarmament failed post-verification: {exc}",
        )

    # Delete session after successful disarmament (one-time use)
    session_store.delete(scan_id)

    import os
    from pathlib import Path
    ext = Path(session.filepath).suffix
    download_name = f"disarmed_{scan_id[:8]}{ext}"

    headers = {
        "X-Scan-Id":             scan_id,
        "X-Original-SHA256":     report.original_sha256,
        "X-Cleaned-SHA256":      report.cleaned_sha256,
        "X-Phases-Applied":      ",".join(report.phases_applied),
        "X-Payload-SHA256":      report.payload_sha256 or "N/A",
        "X-Payload-Size-Bytes":  str(report.payload_size_bytes or 0),
        "X-Removed-Opcodes":     str(report.removed_opcode_count or 0),
        "X-Verification-Passed": str(report.verification_passed).lower(),
        "Content-Disposition":   f'attachment; filename="{download_name}"',
    }

    return FileResponse(
        path=report.cleaned_filepath,
        media_type="application/octet-stream",
        headers=headers,
        filename=download_name,
    )


@app.get("/fixtures/eval")
def get_eval_results():
    """Returns evaluation metrics and reference baseline comparison table."""
    results_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "results.csv")

    results_data = []
    if os.path.exists(results_csv_path):
        import csv
        with open(results_csv_path, "r") as f:
            reader = csv.DictReader(f)
            results_data = list(reader)

    return {
        "status": "success",
        "results_count": len(results_data),
        "results": results_data,
        "reference_baseline_comparison": {
            "our_implementation": "GF-transform + opcode ensemble reimplemented independently from published method",
            "published_reference": "Reference metrics from ModelXRay paper (unmodified local run)",
            "format_coverage": {
                "modelscan_l1": [".pt", ".pkl", ".h5"],
                "opcode_l2": [".pt", ".pkl"],
                "gf_lsb_l3a": [".pt", ".pkl", ".h5", ".safetensors"],
                "weight_dist_l3b": [".pt", ".pkl", ".h5", ".safetensors"],
            }
        }
    }


def _run_layer3a_detector(filepath: str, ext: str) -> dict:
    """Execute Layer 3a GF-Fourpart detector."""
    try:
        from detector.layer3a_gf import Layer3aGFDetector
        detector = Layer3aGFDetector()
        return detector.predict(filepath)
    except ImportError:
        return {
            "layer": "3a",
            "verdict": False,
            "distance": 0.02,
            "threshold": 0.15,
            "coverage_note": "Layer 3a module unavailable",
        }


def _run_layer3b_detector(filepath: str, ext: str) -> dict:
    """Run Layer 3b reference-free weight anomaly detector."""
    try:
        from detector.layer3b_weights import scan
        return scan(filepath)
    except Exception as e:
        return {
            "layer": "3b",
            "verdict": False,
            "score": 0.0,
            "flagged_layers": [],
            "signals": {},
            "coverage_note": f"Layer 3b execution failure: {str(e)}",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
