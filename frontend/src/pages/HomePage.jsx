import React, { useEffect, useRef, useState } from 'react';
import { ShieldAlert, ShieldCheck, AlertTriangle, Upload, FileCode, XCircle, ArrowRight, RefreshCw, Terminal, Cpu, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { createScanJob } from '@/lib/api.js';
import { clearDisarmHandoff, getDisarmHandoff, saveDisarmHandoff } from '@/lib/disarmHandoff.js';
import { saveScanToHistory } from '@/lib/scanHistory.js';
import ForensicEvidencePanel from '@/components/ForensicEvidencePanel.jsx';
import LiveScanPanel from '@/components/LiveScanPanel.jsx';
import Reveal from '@/components/Reveal.jsx';

function formatLayerIssue(issue) {
    if (typeof issue === "string") return issue;
    if (!issue || typeof issue !== "object") return String(issue ?? "Unknown issue");

    return [
        issue.severity && `[${issue.severity}]`,
        issue.description,
        issue.operator && `Operator: ${issue.operator}`,
        issue.module && `Module: ${issue.module}`,
    ].filter(Boolean).join(" ") || "Unspecified scanner finding";
}

export default function HomePage() {
    const navigate = useNavigate();
    const [file, setFile] = useState(null);
    const [scanning, setScanning] = useState(false);
    const [scanResult, setScanResult] = useState(null);
    const [scanError, setScanError] = useState(null);
const [scanEvents, setScanEvents] = useState([]);
    const [streamError, setStreamError] = useState(null);
    const eventSourceRef = useRef(null);
    const resultRef = useRef(null);

    useEffect(() => () => {
        eventSourceRef.current?.close();
    }, []);

    useEffect(() => {
        if (scanResult) resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, [scanResult]);

    useEffect(() => {
        if (scanResult) {
            saveScanToHistory(scanResult);
            if (scanResult.disarmament) saveDisarmHandoff(scanResult);
        }
    }, [scanResult]);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setScanResult(null);
            setScanError(null);
            clearDisarmHandoff();
            setScanEvents([]);
            setStreamError(null);
        }
    };

    const handleScan = async (selectedFile = file) => {
        if (!selectedFile) return;
        setScanning(true);
        setScanError(null);
        setScanResult(null);
        clearDisarmHandoff();
        setScanEvents([]);
        setStreamError(null);
        eventSourceRef.current?.close();

        try {
            const job = await createScanJob(selectedFile);
            const stream = new EventSource(job.streamUrl);
            eventSourceRef.current = stream;

            const receiveEvent = (event) => {
                try {
                    const payload = JSON.parse(event.data);
                    setScanEvents((previous) => [...previous, payload].slice(-100));
                    setStreamError(null);
                    return payload;
                } catch {
                    return null;
                }
            };
            stream.addEventListener('scan_update', receiveEvent);
            stream.addEventListener('complete', (event) => {
                const payload = receiveEvent(event);
                const result = payload?.metadata?.result;
                if (result) {
                    setScanResult(result);
                }
                setScanning(false);
                stream.close();
            });
            stream.addEventListener('failed', (event) => {
                const payload = receiveEvent(event);
                setScanError(payload?.message || 'The scan could not be completed.');
                setScanning(false);
                stream.close();
            });
            stream.onerror = () => {
                if (eventSourceRef.current === stream && stream.readyState !== EventSource.CLOSED) {
                    setStreamError('Connection interrupted — reconnecting to scan telemetry…');
                }
            };
        } catch (err) {
            setScanError(err.message || "Failed to scan model file.");
            setScanning(false);
        }
    };

    const continueToDisarm = () => {
        if (!scanResult?.disarmament) return;
        if (!getDisarmHandoff()) {
            setScanError('Unable to secure this remediation case in the current browser tab. Please allow session storage and try again.');
            return;
        }
        navigate('/disarm');
    };

    const isMalicious = scanResult?.fused?.verdict === "MALICIOUS";
    const isSuspicious = scanResult?.fused?.verdict === "SUSPICIOUS";
    const isClean = scanResult?.fused?.verdict === "BENIGN" || scanResult?.fused?.verdict === "CLEAN";
    const findingSources = scanResult ? [
        ['L1', Boolean(scanResult.layer1?.verdict)], ['L2', Boolean(scanResult.layer2?.verdict)],
        ['L3A', Boolean(scanResult.layer3a?.verdict), scanResult.layer3a?.forensic_band], ['L3B', Boolean(scanResult.layer3b?.verdict)],
    ] : [];

    return (
        <div className="space-y-12">
            {/* Hero Section */}
            <Reveal className="border-2 border-ink bg-[#F2ECDC] p-8 md:p-12">
                <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                    <div className="space-y-3 max-w-3xl">
                        <div className="inline-flex items-center gap-2 rounded-full border border-ink bg-[#EFC23A] px-3.5 py-1 font-mono text-[11px] font-extrabold uppercase tracking-[0.16em]">
                            <span className="h-2 w-2 rounded-full bg-[#C4402E] animate-ping" />
                            DEPLOYMENT GATEKEEPER · 4-LAYER SCAN
                        </div>
                        <h1 className="font-serif text-3xl sm:text-5xl font-black tracking-tight text-[#17130E] leading-[1.08]">
                            Detect. Interrogate. <span className="italic font-medium text-[#C4402E]">Disarm.</span>
                        </h1>
                        <p className="font-sans text-sm sm:text-base text-[#17130E]/80 leading-relaxed max-w-2xl">
                            Multi-layer neural model defense engine. Detects malicious opcode execution, 43-D statistical grammar anomalies, 512×512 Galois Field steganography, and reference-free spectral backdoors.
                        </p>
                    </div>

                    <div className="flex flex-col gap-2.5 border-2 border-dashed border-ink/40 bg-[#E8DFC6] p-5 font-mono text-xs">
                        <span className="font-bold uppercase tracking-wider text-[#17130E]/60 text-[10px]">ACTIVE DEFENSE SPECS</span>
                        <div className="flex justify-between gap-6 border-b border-ink/20 pb-1">
                            <span>L1 OPCODE LATENCY</span>
                            <b className="text-[#2F4FE0]">&lt; 4.0ms</b>
                        </div>
                        <div className="flex justify-between gap-6 border-b border-ink/20 pb-1">
                            <span>L2 XGBOOST TREES</span>
                            <b>100 ENSEMBLE</b>
                        </div>
                        <div className="flex justify-between gap-6 border-b border-ink/20 pb-1">
                            <span>L3A SCZ MARGIN τ</span>
                            <b className="text-[#C4402E]">0.072236</b>
                        </div>
                        <div className="flex justify-between gap-6">
                            <span>DISARM SURGERY</span>
                            <b className="text-[#17130E]">AUTO-VERIFY</b>
                        </div>
                    </div>
                </div>
            </Reveal>

            {/* Scanner Upload Area */}
            <Reveal delay={0.08} className="border-2 border-ink bg-[#F2ECDC] p-6 sm:p-10 shadow-[6px_6px_0px_#17130E]">
                <div className="flex items-center justify-between border-b-2 border-ink pb-4 mb-6">
                    <div>
                        <h2 className="font-mono text-sm font-extrabold uppercase tracking-[0.2em] text-[#17130E]">
                            1. MODEL UPLOAD & DEEP INSPECTION
                        </h2>
                        <p className="font-mono text-[11px] text-[#17130E]/60">Supports PyTorch (.pt, .pth, .bin), SafeTensors (.safetensors), Pickles (.pkl), HDF5 (.h5)</p>
                    </div>
                    <span className="font-mono text-xs font-bold text-[#2F4FE0]">PORT 8000 ARMED</span>
                </div>

                <div className="grid gap-6 md:grid-cols-12">
                    <div className="md:col-span-8">
                        <label
                            htmlFor="file-upload"
                            className="group flex flex-col items-center justify-center border-2 border-dashed border-ink bg-[#E8DFC6] p-8 sm:p-12 text-center cursor-pointer transition hover:bg-[#dfd4b8]"
                        >
                            <input
                                id="file-upload"
                                type="file"
                                accept=".pt,.pth,.safetensors,.pkl,.h5,.hdf5,.bin"
                                onChange={handleFileChange}
                                className="hidden"
                            />
                            <div className="flex h-14 w-14 items-center justify-center rounded-full border-2 border-ink bg-[#F2ECDC] transition group-hover:scale-110">
                                <Upload className="h-6 w-6 text-[#17130E]" />
                            </div>
                            <p className="mt-4 font-serif text-lg font-bold text-[#17130E]">
                                {file ? file.name : "Drop model weights checkpoint here"}
                            </p>
                            <p className="font-mono text-xs text-[#17130E]/60 mt-1">
                                {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · Click to choose different file` : "or click to browse local filesystem"}
                            </p>
                        </label>
                    </div>

                    <div className="md:col-span-4 flex flex-col justify-between gap-4 border-2 border-ink bg-[#E8DFC6] p-6">
                        <div>
                            <span className="font-mono text-[10.5px] font-bold uppercase tracking-wider text-[#17130E]/60 block mb-2">
                                SCAN CONTROLS
                            </span>
                            <p className="text-xs text-[#17130E]/80 leading-relaxed font-mono">
                                Executes all 4 inspection layers sequentially with complete host process isolation and zero unsafe unpickling.
                            </p>
                        </div>

                        <button
                            onClick={() => handleScan()}
                            disabled={!file || scanning}
                            className="w-full inline-flex items-center justify-center gap-2 rounded-full border-2 border-ink bg-[#EFC23A] py-3.5 font-mono text-xs font-bold uppercase tracking-[0.18em] text-[#17130E] transition-all hover:bg-[#f6cd4b] hover:shadow-[3px_3px_0px_#17130E] disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {scanning ? (
                                <>
                                    <RefreshCw className="h-4 w-4 animate-spin text-[#17130E]" />
                                    INSPECTING WEIGHTS...
                                </>
                            ) : (
                                <>
                                    <ShieldAlert className="h-4 w-4 text-[#C4402E]" />
                                    RUN FULL 4-LAYER SCAN
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {scanError && (
                    <div className="mt-6 border-2 border-[#C4402E] bg-[#F7DFDA] p-4 text-[#C4402E] font-mono text-xs flex items-center gap-3">
                        <XCircle className="h-5 w-5 flex-shrink-0" />
                        <span><b>Scan Error:</b> {scanError}</span>
                    </div>
                )}
            </Reveal>

            {(scanning || scanEvents.length > 0) && (
                <LiveScanPanel
                    events={scanEvents}
                    scanning={scanning}
                    connectionError={streamError}
                />
            )}

            {/* Scan Results Board */}
            {scanResult && (
                <Reveal ref={resultRef} delay={0.08} className="space-y-8">
                    {/* Fused Verdict Header */}
                    <div className={`border-2 border-ink p-8 shadow-[6px_6px_0px_#17130E] ${
                        isMalicious ? "bg-[#F7DFDA]" : isSuspicious ? "bg-[#EFC23A]" : "bg-[#E4EAFB]"
                    }`}>
                        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between border-b-2 border-ink pb-6">
                            <div>
                                <span className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#17130E]/70 block mb-1">
                                    CASE FILE VERDICT · {scanResult.filename}
                                </span>
                                <h3 className="font-serif text-3xl sm:text-4xl font-black tracking-tight text-[#17130E] flex items-center gap-3">
                                    {isMalicious && <ShieldAlert className="h-8 w-8 text-[#C4402E]" />}
                                    {isSuspicious && <AlertTriangle className="h-8 w-8 text-[#17130E]" />}
                                    {isClean && <ShieldCheck className="h-8 w-8 text-[#2F4FE0]" />}
                                    VERDICT: {scanResult.fused?.verdict}
                                </h3>
                            </div>

                            <div className="flex items-center gap-6">
                                <div className="text-right font-mono">
                                    <span className="text-[10px] text-[#17130E]/70 uppercase tracking-widest block font-bold">FUSED RISK SCORE</span>
                                    <b className="text-2xl font-bold">{scanResult.fused?.score?.toFixed(3) || "0.000"}</b>
                                </div>
                                <div className="text-right font-mono">
                                    <span className="text-[10px] text-[#17130E]/70 uppercase tracking-widest block font-bold">FORENSIC BAND</span>
                                    <b className="text-base font-bold uppercase">{scanResult.fused?.forensic_band || "CLEAN"}</b>
                                </div>
                            </div>
                        </div>

                        {scanResult.fused?.explanation && (
                            <p className="mt-4 font-mono text-xs text-[#17130E]/85">
                                <b>Synthesis:</b> {scanResult.fused.explanation}
                            </p>
                        )}
                        <div className="mt-5 flex flex-wrap gap-2 border-t border-ink/20 pt-4 font-mono text-[10px] font-bold uppercase">
                            {findingSources.map(([layer, flagged, band]) => <span key={layer} className={`border px-2.5 py-1 ${flagged ? 'border-[#C4402E] bg-[#F7DFDA] text-[#C4402E]' : band === 'suspicious' ? 'border-ink bg-[#17130E] text-[#EFC23A]' : 'border-[#2F4FE0] bg-[#E4EAFB] text-[#2F4FE0]'}`}>{layer} · {flagged ? 'Flagged' : band === 'suspicious' ? 'Elevated' : 'Clear'}</span>)}
                        </div>
                    </div>

                    {/* 4 Layer Details Grid */}
                    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                        {/* Layer 1 */}
                        <div className="border-2 border-ink bg-[#F2ECDC] p-5 space-y-3">
                            <div className="flex items-center justify-between border-b border-ink/20 pb-2">
                                <span className="font-mono text-xs font-bold">L1: OPCODE SCAN</span>
                                <span className={`font-mono text-[10px] font-bold px-2 py-0.5 border border-ink rounded-full ${
                                    scanResult.layer1?.verdict ? "bg-[#C4402E] text-white" : "bg-[#2F4FE0] text-white"
                                }`}>
                                    {scanResult.layer1?.verdict ? "FLAGGED" : "CLEARED"}
                                </span>
                            </div>
                            <p className="text-xs font-mono text-[#17130E]/70">
                                {scanResult.layer1?.coverage_note || "Zero-deserialization byte parser"}
                            </p>
                            {scanResult.layer1?.issues?.length > 0 ? (
                                <ul className="space-y-1 font-mono text-[11px] text-[#C4402E]">
                                    {scanResult.layer1.issues.map((iss, i) => (
                                        <li key={i}>• {formatLayerIssue(iss)}</li>
                                    ))}
                                </ul>
                            ) : (
                                <span className="font-mono text-[11px] text-[#2F4FE0] block">✓ No dangerous globals</span>
                            )}
                        </div>

                        {/* Layer 2 */}
                        <div className="border-2 border-ink bg-[#F2ECDC] p-5 space-y-3">
                            <div className="flex items-center justify-between border-b border-ink/20 pb-2">
                                <span className="font-mono text-xs font-bold">L2: XGBOOST N-GRAM</span>
                                <span className={`font-mono text-[10px] font-bold px-2 py-0.5 border border-ink rounded-full ${
                                    scanResult.layer2?.verdict ? "bg-[#C4402E] text-white" : "bg-[#2F4FE0] text-white"
                                }`}>
                                    {scanResult.layer2?.verdict ? "FLAGGED" : "CLEARED"}
                                </span>
                            </div>
                            <div className="font-mono text-xs">
                                <span>P(MALICIOUS): </span>
                                <b>{(scanResult.layer2?.probability * 100 || 0).toFixed(1)}%</b>
                            </div>
                            {scanResult.layer2?.top_features?.length > 0 && (
                                <div className="space-y-1 font-mono text-[10px] text-[#17130E]/80">
                                    <span className="text-[#17130E]/60 uppercase">Top Risk Feature:</span>
                                    <div>• {scanResult.layer2.top_features[0]?.feature} (+{scanResult.layer2.top_features[0]?.value?.toFixed(3)})</div>
                                </div>
                            )}
                        </div>

                        {/* Layer 3a */}
                        <div className="border-2 border-ink bg-[#F2ECDC] p-5 space-y-3">
                            <div className="flex items-center justify-between border-b border-ink/20 pb-2">
                                <span className="font-mono text-xs font-bold">L3A: STEGANALYSIS</span>
                                <span className={`font-mono text-[10px] font-bold px-2 py-0.5 border border-ink rounded-full ${
                                    scanResult.layer3a?.verdict ? "bg-[#C4402E] text-white" : scanResult.layer3a?.forensic_band === "suspicious" ? "bg-[#EFC23A] text-[#17130E]" : "bg-[#2F4FE0] text-white"
                                }`}>
                                    {scanResult.layer3a?.verdict ? "FLAGGED" : scanResult.layer3a?.forensic_band === "suspicious" ? "ELEVATED" : "CLEARED"}
                                </span>
                            </div>
                            <div className="font-mono text-xs space-y-1">
                                <div>Margin Score: <b>{scanResult.layer3a?.score?.toFixed(3) || "0.000"}</b></div>
                                <div className="text-[10px] text-[#17130E]/60">τ = 0.072236 calibrated</div>
                            </div>
                            <span className="font-mono text-[10.5px] text-[#17130E]/80 block">
                                Band: <b className="uppercase">{scanResult.layer3a?.forensic_band || "clean"}</b>
                            </span>
                        </div>

                        {/* Layer 3b */}
                        <div className="border-2 border-ink bg-[#F2ECDC] p-5 space-y-3">
                            <div className="flex items-center justify-between border-b border-ink/20 pb-2">
                                <span className="font-mono text-xs font-bold">L3B: SPECTRAL INVAR</span>
                                <span className={`font-mono text-[10px] font-bold px-2 py-0.5 border border-ink rounded-full ${
                                    scanResult.layer3b?.verdict ? "bg-[#C4402E] text-white" : "bg-[#2F4FE0] text-white"
                                }`}>
                                    {scanResult.layer3b?.verdict ? "FLAGGED" : "CLEARED"}
                                </span>
                            </div>
                            <div className="font-mono text-xs">
                                <span>Anomaly Score: <b>{scanResult.layer3b?.score?.toFixed(3) || "0.000"}</b></span>
                            </div>
                            <p className="font-mono text-[10px] text-[#17130E]/70">
                                {scanResult.layer3b?.flagged_layers?.length > 0
                                    ? `Flagged: ${scanResult.layer3b.flagged_layers.join(", ")}`
                                    : "No spectral collapse observed"}
                            </p>
                        </div>
                    </div>

                    <ForensicEvidencePanel forensics={scanResult.steganography_forensics} />

                    {/* Gemini Threat Assessment */}
                    {scanResult.summary && (() => {
                        const raw = scanResult.summary.summary || JSON.stringify(scanResult.summary, null, 2);

                        // Group lines into sections for clean rendering
                        const sections = [];
                        let current = { header: null, body: [] };
                        raw.split('\n').forEach(line => {
                            const t = line.trim();
                            if (t.startsWith('▸')) {
                                if (current.header !== null || current.body.length > 0) sections.push(current);
                                current = { header: t, body: [] };
                            } else if (t.startsWith('Note:')) {
                                if (current.header !== null || current.body.length > 0) sections.push(current);
                                sections.push({ header: 'NOTE', body: [t] });
                                current = { header: null, body: [] };
                            } else if (t !== '') {
                                current.body.push(t);
                            }
                        });
                        if (current.header !== null || current.body.length > 0) sections.push(current);

                        return (
                            <div className="border-2 border-ink bg-[#17130E] text-[#F2ECDC] p-6 sm:p-8">
                                <div className="flex items-center gap-2 border-b border-[#F2ECDC]/20 pb-3 mb-6">
                                    <Terminal className="h-4 w-4 text-[#EFC23A]" />
                                    <span className="font-mono text-xs font-bold uppercase tracking-[0.18em] text-[#EFC23A]">
                                        GEMINI THREAT DOSSIER &amp; REMEDIATION
                                    </span>
                                </div>
                                <div className="space-y-6">
                                    {sections.map((section, si) => {
                                        if (section.header === 'NOTE') {
                                            return (
                                                <p key={si} className="font-mono text-[10px] text-[#F2ECDC]/35 italic border-t border-[#F2ECDC]/10 pt-4">
                                                    {section.body[0]}
                                                </p>
                                            );
                                        }
                                        return (
                                            <div key={si} className="space-y-1.5">
                                                {section.header && (
                                                    <p className="font-mono text-[11px] font-bold text-[#EFC23A] uppercase tracking-[0.2em] mb-3">
                                                        {section.header}
                                                    </p>
                                                )}
                                                {section.body.map((line, li) =>
                                                    line.startsWith('•') ? (
                                                        <div key={li} className="flex gap-2.5 items-start">
                                                            <span className="text-[#EFC23A] shrink-0 mt-[3px] text-xs">•</span>
                                                            <span className="text-sm text-[#F2ECDC]/85 leading-relaxed">{line.replace(/^•\s*/, '')}</span>
                                                        </div>
                                                    ) : (
                                                        <p key={li} className="text-sm text-[#F2ECDC]/75 leading-relaxed">{line}</p>
                                                    )
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        );
                    })()}



                    {/* Disarmament Action Banner (If Malicious) */}
                    {scanResult.disarmament && (
                        <div className="border-2 border-ink bg-[#EFC23A] p-6 sm:p-8 text-[#17130E] shadow-[6px_6px_0px_#17130E]">
                            <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                                <div className="space-y-2">
                                    <div className="inline-flex items-center gap-2 rounded-full border border-ink bg-[#F2ECDC] px-3 py-0.5 font-mono text-[10.5px] font-bold">
                                        REMEDIATION READY · ONE-TIME VERIFIED RELEASE
                                    </div>
                                    <h3 className="font-serif text-2xl sm:text-3xl font-black">
                                        Review complete. Continue to disarm.
                                    </h3>
                                    <p className="font-sans text-xs sm:text-sm text-[#17130E]/80 max-w-xl">
                                        Your selected case file for <b>{scanResult.filename}</b> is ready for one-time surgery, verification, and clean-model release.
                                    </p>
                                </div>

                                <button
                                    onClick={continueToDisarm}
                                    className="inline-flex items-center justify-center gap-2 rounded-full border-2 border-ink bg-[#17130E] px-8 py-4 font-mono text-xs font-extrabold uppercase tracking-[0.2em] text-[#F2ECDC] transition hover:bg-[#2d251c] hover:shadow-[3px_3px_0px_#F2ECDC] disabled:opacity-50"
                                >
                                    CONTINUE TO DISARM <ArrowRight className="h-4 w-4 text-[#EFC23A]" />
                                </button>
                            </div>
                        </div>
                    )}
                </Reveal>
            )}
        </div>
    );
}
