const STORAGE_KEY = 'mirage.disarm-handoff.v1';

function storageAvailable() {
    return typeof window !== 'undefined' && Boolean(window.sessionStorage);
}

function findingSources(scanResult) {
    const sources = [];
    if (scanResult.layer1?.verdict) sources.push('L1 opcode');
    if (scanResult.layer2?.verdict) sources.push('L2 grammar');
    if (scanResult.layer3a?.verdict || ['suspicious', 'malicious'].includes(scanResult.layer3a?.forensic_band)) sources.push('L3a tensor');
    if (scanResult.layer3b?.verdict) sources.push('L3b spectral');
    return sources;
}

export function saveDisarmHandoff(scanResult) {
    const disarmament = scanResult?.disarmament;
    if (!storageAvailable() || !disarmament?.scan_id) return null;
    const handoff = {
        scanId: disarmament.scan_id,
        filename: scanResult.filename || 'scanned model',
        format: scanResult.format || 'model',
        verdict: scanResult.fused?.verdict || 'REVIEW REQUIRED',
        riskScore: scanResult.fused?.score ?? null,
        sources: findingSources(scanResult),
        status: disarmament.status,
        expiresAt: Date.now() + (disarmament.expires_in_seconds || 3600) * 1000,
    };
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(handoff));
    return handoff;
}

export function getDisarmHandoff() {
    if (!storageAvailable()) return null;
    try {
        const raw = window.sessionStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        const handoff = JSON.parse(raw);
        if (!handoff?.scanId || !handoff?.expiresAt || handoff.expiresAt <= Date.now()) {
            clearDisarmHandoff();
            return null;
        }
        return handoff;
    } catch {
        clearDisarmHandoff();
        return null;
    }
}

export function clearDisarmHandoff() {
    if (storageAvailable()) window.sessionStorage.removeItem(STORAGE_KEY);
}
