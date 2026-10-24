const STORAGE_KEY = 'mirage.scan-history.v1';
const MAX_ENTRIES = 50;

function storageAvailable() {
    return typeof window !== 'undefined' && Boolean(window.localStorage);
}

export function saveScanToHistory(scanResult) {
    if (!storageAvailable() || !scanResult) return;
    const entry = {
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
        timestamp: Date.now(),
        filename: scanResult.filename || 'unknown',
        format: scanResult.format || '—',
        verdict: scanResult.fused?.verdict || 'UNKNOWN',
        riskScore: scanResult.fused?.score ?? null,
        layers: {
            l1: Boolean(scanResult.layer1?.verdict),
            l2: Boolean(scanResult.layer2?.verdict),
            l3a: Boolean(scanResult.layer3a?.verdict),
            l3b: Boolean(scanResult.layer3b?.verdict),
        },
    };
    try {
        const history = getScanHistory();
        history.unshift(entry);
        if (history.length > MAX_ENTRIES) history.length = MAX_ENTRIES;
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch { /* quota or parse error — silently skip */ }
}

export function getScanHistory() {
    if (!storageAvailable()) return [];
    try {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

export function clearScanHistory() {
    if (storageAvailable()) window.localStorage.removeItem(STORAGE_KEY);
}
