/**
 * PROJECT-CLIMAX API Client
 * Interfaces with the FastAPI defense engine backend (Port 8000).
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function getHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        return { status: "offline", error: err.message };
    }
}

export async function scanModelFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/scan`, {
        method: "POST",
        body: formData,
    });

    if (!res.ok) {
        let errDetail = "Scan request failed";
        try {
            const errJson = await res.json();
            errDetail = errJson.detail || errDetail;
        } catch {
            errDetail = await res.text();
        }
        throw new Error(errDetail);
    }

    return await res.json();
}

export async function createScanJob(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/scans`, { method: "POST", body: formData });
    if (!res.ok) {
        let detail = "Unable to start scan";
        try {
            detail = (await res.json()).detail || detail;
        } catch {
            detail = await res.text();
        }
        throw new Error(detail);
    }

    const job = await res.json();
    return { ...job, streamUrl: new URL(job.stream_url, API_BASE).toString() };
}

export async function disarmModelSession(scanId) {
    const formData = new FormData();
    formData.append("scan_id", scanId);
    formData.append("confirm", "true");

    const res = await fetch(`${API_BASE}/disarm`, {
        method: "POST",
        body: formData,
    });

    if (res.status === 429) {
        const retryAfter = res.headers.get("Retry-After") || "3600";
        throw new Error(`Rate limit exceeded (10 req/hr). Try again in ${Math.ceil(Number(retryAfter) / 60)} minutes.`);
    }

    if (!res.ok) {
        let errDetail = "Disarmament failed";
        try {
            const errJson = await res.json();
            errDetail = errJson.detail || errDetail;
        } catch {
            errDetail = await res.text();
        }
        throw new Error(errDetail);
    }

    const blob = await res.blob();
    const contentDisposition = res.headers.get("Content-Disposition") || "";
    let filename = "disarmed_model.pt";
    const filenameMatch = contentDisposition.match(/filename=["']?([^"';]+)["']?/);
    if (filenameMatch && filenameMatch[1]) {
        filename = filenameMatch[1];
    }

    const headers = {
        payloadSha256: res.headers.get("X-Payload-SHA256"),
        payloadSizeBytes: res.headers.get("X-Payload-Size-Bytes"),
        originalSha256: res.headers.get("X-Original-SHA256"),
        cleanedSha256: res.headers.get("X-Cleaned-SHA256"),
        phasesApplied: res.headers.get("X-Phases-Applied"),
        verificationPassed: res.headers.get("X-Verification-Passed"),
    };

    return { blob, filename, headers };
}

export async function getBenchmarkMatrix() {
    try {
        const res = await fetch(`${API_BASE}/fixtures/eval`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.warn("Live benchmark endpoint unavailable, using offline cache:", err);
        return null;
    }
}
