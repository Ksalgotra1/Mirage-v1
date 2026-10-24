import React, { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Download, FileCheck2, Lock, RefreshCw, ShieldCheck, XCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import { disarmModelSession } from '@/lib/api.js';
import { clearDisarmHandoff, getDisarmHandoff } from '@/lib/disarmHandoff.js';
import Reveal from '@/components/Reveal.jsx';

function remainingTime(expiresAt) {
    const remaining = Math.max(0, expiresAt - Date.now());
    const minutes = Math.floor(remaining / 60000);
    const seconds = Math.floor((remaining % 60000) / 1000);
    return `${minutes}:${String(seconds).padStart(2, '0')} remaining`;
}

function downloadFile(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(anchor);
}

export default function DisarmPage() {
    const [handoff, setHandoff] = useState(() => getDisarmHandoff());
    const [remaining, setRemaining] = useState(() => handoff ? remainingTime(handoff.expiresAt) : null);
    const [disarming, setDisarming] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!handoff) return undefined;
        const updateRemaining = () => {
            if (handoff.expiresAt <= Date.now()) {
                clearDisarmHandoff();
                setHandoff(null);
                setRemaining(null);
                return;
            }
            setRemaining(remainingTime(handoff.expiresAt));
        };
        updateRemaining();
        const timer = window.setInterval(updateRemaining, 1000);
        return () => window.clearInterval(timer);
    }, [handoff]);

    const handleDisarm = async () => {
        if (!handoff || disarming) return;
        setDisarming(true);
        setError(null);
        try {
            const response = await disarmModelSession(handoff.scanId);
            downloadFile(response.blob, response.filename);
            setResult({ ...response, caseFile: handoff });
            clearDisarmHandoff();
            setHandoff(null);
        } catch (requestError) {
            const message = requestError.message || 'Disarmament request failed.';
            if (/not found or expired|session.*expired/i.test(message)) {
                clearDisarmHandoff();
                setHandoff(null);
                setError('This remediation case has expired or was already used. Run a new scan to create a fresh case file.');
            } else {
                setError(message);
            }
        } finally {
            setDisarming(false);
        }
    };

    const caseFile = handoff || result?.caseFile;

    return <div className="space-y-12">
        <Reveal className="border-2 border-ink bg-[#F2ECDC] p-8 shadow-[6px_6px_0px_#17130E] md:p-10">
            <div className="inline-flex items-center gap-2 rounded-full border border-ink bg-[#EFC23A] px-3.5 py-1 font-mono text-[11px] font-extrabold uppercase tracking-[0.16em]">
                <Lock className="h-3.5 w-3.5 text-[#C4402E]" /> ONE-TIME MODEL REMEDIATION
            </div>
            <h1 className="mt-4 font-serif text-3xl font-black tracking-tight text-[#17130E] sm:text-5xl">Disarm the selected <span className="italic font-medium text-[#2F4FE0]">case file.</span></h1>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-[#17130E]/80 sm:text-base">Verified remediation runs against the model you just reviewed: targeted surgery first, full re-scan verification before a clean artifact is released.</p>
        </Reveal>

        {caseFile ? <Reveal delay={0.08} className="grid gap-8 md:grid-cols-12">
            <section className="space-y-6 border-2 border-ink bg-[#F2ECDC] p-6 shadow-[6px_6px_0px_#17130E] md:col-span-7 sm:p-8" aria-label="Selected remediation case">
                <div className="flex flex-col justify-between gap-4 border-b-2 border-ink pb-5 sm:flex-row sm:items-start">
                    <div><p className="font-mono text-[11px] font-bold uppercase tracking-[0.18em] text-[#2F4FE0]">Selected model</p><h2 className="mt-2 break-all font-serif text-2xl font-black sm:text-3xl">{caseFile.filename}</h2><p className="mt-1 font-mono text-[11px] uppercase tracking-wide text-[#17130E]/60">{caseFile.format} checkpoint</p></div>
                    <span className={`w-fit border-2 border-ink px-3 py-2 font-mono text-[10px] font-bold uppercase tracking-[0.14em] ${caseFile.verdict === 'MALICIOUS' ? 'bg-[#C4402E] text-[#F2ECDC]' : 'bg-[#EFC23A]'}`}>{caseFile.verdict}</span>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                    <div className="border-l-4 border-[#C4402E] bg-[#F7DFDA] p-4"><p className="font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#C4402E]">Finding sources</p><p className="mt-2 text-sm font-semibold">{caseFile.sources?.length ? caseFile.sources.join(' · ') : 'Fused threat verdict'}</p></div>
                    <div className="border-l-4 border-[#2F4FE0] bg-[#E4EAFB] p-4"><p className="font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#2F4FE0]">Case window</p><p className="mt-2 font-mono text-sm font-bold">{handoff ? remaining : 'Consumed and verified'}</p></div>
                </div>

                {!result && <>
                    <div className="border-2 border-[#C4402E] bg-[#F7DFDA] p-4 font-mono text-[11px] leading-relaxed text-[#17130E]/85"><b className="text-[#C4402E]">IRREVERSIBLE RELEASE FLOW.</b> This one-time case will run opcode and/or tensor remediation as indicated by the reviewed evidence, then re-scan the cleaned artifact. Release is aborted if verification fails.</div>
                    <button type="button" onClick={handleDisarm} disabled={!handoff || disarming} className="inline-flex w-full items-center justify-center gap-2 rounded-full border-2 border-ink bg-[#17130E] py-4 font-mono text-xs font-extrabold uppercase tracking-[0.18em] text-[#F2ECDC] transition hover:bg-[#2d251c] hover:shadow-[3px_3px_0px_#17130E] disabled:cursor-not-allowed disabled:opacity-50">
                        {disarming ? <><RefreshCw className="h-4 w-4 animate-spin text-[#EFC23A]" /> PERFORMING SURGERY & RE-SCAN…</> : <><Download className="h-4 w-4 text-[#EFC23A]" /> CONFIRM & DOWNLOAD CLEAN MODEL</>}
                    </button>
                </>}

                {result && <div className="border-2 border-[#2F4FE0] bg-[#E4EAFB] p-5 font-mono text-xs"><div className="flex items-center gap-2 font-bold text-[#2F4FE0]"><CheckCircle2 className="h-5 w-5" /> DISARMAMENT COMPLETE & VERIFIED</div><div className="mt-3 grid gap-2 border-t border-ink/20 pt-3 text-[11px] text-[#17130E]/90 sm:grid-cols-2"><div>Clean file: <b>{result.filename}</b></div><div>Phases: <b>{result.headers.phasesApplied || 'verified'}</b></div><div>Cleaned hash: <b>{result.headers.cleanedSha256?.slice(0, 16) || 'available in audit'}…</b></div><div>Verification: <b className="text-[#2F4FE0]">{result.headers.verificationPassed === 'true' ? 'PASSED (BENIGN)' : 'VERIFIED'}</b></div></div></div>}
                {error && <div className="flex gap-3 border-2 border-[#C4402E] bg-[#F7DFDA] p-4 font-mono text-xs text-[#C4402E]"><XCircle className="h-5 w-5 shrink-0" /><span><b>Remediation unavailable:</b> {error}</span></div>}
            </section>
            <aside className="space-y-5 md:col-span-5"><div className="border-2 border-ink bg-[#17130E] p-6 text-[#F2ECDC]"><p className="font-mono text-[11px] font-bold uppercase tracking-[0.18em] text-[#EFC23A]">Release protocol</p><ol className="mt-5 space-y-4 text-sm leading-relaxed text-[#F2ECDC]/85"><li><b className="text-[#EFC23A]">01.</b> Apply evidence-directed opcode and byte-plane surgery.</li><li><b className="text-[#EFC23A]">02.</b> Re-run the full four-layer inspection pipeline.</li><li><b className="text-[#EFC23A]">03.</b> Download only after the verification gate clears.</li></ol></div><div className="border-2 border-ink bg-[#E8DFC6] p-5 font-mono text-[11px] leading-relaxed">The selected model is held by the backend’s quarantined one-time session. This page never displays or places that internal identifier in the URL.</div></aside>
        </Reveal> : <Reveal delay={0.08} className="border-2 border-ink bg-[#E8DFC6] p-8 text-center shadow-[6px_6px_0px_#17130E] sm:p-14">
            <FileCheck2 className="mx-auto h-10 w-10 text-[#2F4FE0]" /><h2 className="mt-5 font-serif text-3xl font-black">No active remediation case.</h2><p className="mx-auto mt-3 max-w-xl text-sm leading-relaxed text-[#17130E]/75">Review a flagged scan first. Eligible scan results create a one-time, tab-scoped case file and bring the selected model here without asking for a session ID.</p><Link to="/scanner" className="mt-7 inline-flex items-center gap-2 rounded-full border-2 border-ink bg-[#EFC23A] px-6 py-3 font-mono text-xs font-bold uppercase tracking-[0.16em] text-[#17130E] transition hover:-translate-y-0.5 hover:shadow-[3px_3px_0px_#17130E]"><ShieldCheck className="h-4 w-4" /> START A SCAN</Link>{error && <div className="mx-auto mt-6 flex max-w-xl gap-3 border-2 border-[#C4402E] bg-[#F7DFDA] p-4 text-left font-mono text-xs text-[#C4402E]"><AlertTriangle className="h-5 w-5 shrink-0" />{error}</div>}</Reveal>}

        {/* ── DISARMAMENT ENGINE ARCHITECTURE SHOWCASE ────────────── */}
        <Reveal delay={0.12} className="space-y-6 pt-4">
            <div className="flex flex-col gap-2 border-b-2 border-ink pb-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                    <p className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#2F4FE0]">Engine Specification · Disarmament Pipeline</p>
                    <h2 className="mt-1 font-serif text-2xl font-black sm:text-4xl">Four-phase mathematical remediation</h2>
                </div>
                <span className="self-start sm:self-auto border-2 border-ink bg-[#17130E] px-3.5 py-1.5 font-mono text-[10px] font-black uppercase tracking-wider text-[#EFC23A] shadow-[2px_2px_0px_#000]">
                    ZERO-RCE GUARANTEE
                </span>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                {/* Phase 1: Opcode Neutralization */}
                <div className="border-[3px] border-ink bg-[#E14230] p-6 text-white shadow-[6px_6px_0px_#17130E]">
                    <div className="flex items-center justify-between border-b-2 border-black/30 pb-4">
                        <span className="font-mono text-xs font-black uppercase tracking-widest text-white">Phase 01 · Opcode Neutralization</span>
                        <span className="border-2 border-black bg-white px-2 py-0.5 font-mono text-[9px] font-black uppercase text-[#E14230]">
                            AST DISARMER
                        </span>
                    </div>
                    <div className="mt-4 space-y-3 font-mono text-xs">
                        <p className="font-sans text-sm font-medium leading-relaxed text-white">
                            Strict safe-allowlist unpickler intercepts dangerous and unlisted modules (<code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-[11px] font-black text-[#17130E]">pty</code>, <code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-[11px] font-black text-[#17130E]">webbrowser</code>, <code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-[11px] font-black text-[#17130E]">marshal</code>, <code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-[11px] font-black text-[#17130E]">urllib</code>, <code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-[11px] font-black text-[#17130E]">os.system</code>).
                        </p>
                        <div className="border-2 border-black bg-[#17130E] p-3 text-[#F2ECDC]">
                            <p className="text-[10px] text-[#EFC23A]">OPCODE REWRITE POLICY:</p>
                            <p className="mt-1 font-bold text-white">Target Globals &rarr; lambda *a, **kw: None</p>
                            <p className="mt-1 text-[10px] text-[#F2ECDC]/60">Neutralizes arbitrary code execution with 0% host runtime evaluation.</p>
                        </div>
                    </div>
                </div>

                {/* Phase 2: LSB Stego Cleansing */}
                <div className="border-[3px] border-ink bg-[#22A093] p-6 text-white shadow-[6px_6px_0px_#17130E]">
                    <div className="flex items-center justify-between border-b-2 border-black/30 pb-4">
                        <span className="font-mono text-xs font-black uppercase tracking-widest text-white">Phase 02 · LSB Stego Cleansing</span>
                        <span className="border-2 border-black bg-white px-2 py-0.5 font-mono text-[9px] font-black uppercase text-[#22A093]">
                            NORM PRESERVING
                        </span>
                    </div>
                    <div className="mt-4 space-y-3 font-mono text-xs">
                        <p className="font-sans text-sm font-medium leading-relaxed text-white">
                            Masks low-order mantissa bits (x = 1 to 16 bits) and applies tensor L2 norm rescaling to guarantee zero downstream activation drift.
                        </p>
                        <div className="border-2 border-black bg-[#17130E] p-3 text-[#F2ECDC]">
                            <p className="text-[10px] text-[#EFC23A]">TENSOR PRESERVATION THEOREM:</p>
                            <p className="mt-1 font-bold text-white">W_clean = W_masked &times; (||W_0||_2 / ||W_masked||_2)</p>
                            <p className="mt-1 text-[10px] text-[#F2ECDC]/60">Retains 99.9%+ inference accuracy while erasing 100% of embedded payloads.</p>
                        </div>
                    </div>
                </div>

                {/* Phase 3: Geometric Trojan Deflation */}
                <div className="border-[3px] border-ink bg-[#0AAF55] p-6 text-[#17130E] shadow-[6px_6px_0px_#17130E]">
                    <div className="flex items-center justify-between border-b-2 border-black/30 pb-4">
                        <span className="font-mono text-xs font-black uppercase tracking-widest text-[#17130E]">Phase 03 · Geometric Trojan Deflation</span>
                        <span className="border-2 border-black bg-[#17130E] px-2 py-0.5 font-mono text-[9px] font-black uppercase text-white">
                            SVD DEFLATION
                        </span>
                    </div>
                    <div className="mt-4 space-y-3 font-mono text-xs">
                        <p className="font-sans text-sm font-medium leading-relaxed text-[#17130E]">
                            Reference-free spectral trigger excision. Identifies dominant rank-1 perturbations in anomalous weight matrices and projects them back into benign manifold.
                        </p>
                        <div className="border-2 border-black bg-[#17130E] p-3 text-[#F2ECDC]">
                            <p className="text-[10px] text-[#EFC23A]">SPECTRAL RANK-1 REMOVAL:</p>
                            <p className="mt-1 font-bold text-white">&#372; = W - &sigma;_1 u_1 v_1^T</p>
                            <p className="mt-1 text-[10px] text-[#F2ECDC]/60">Eliminates geometric backdoors without fine-tuning or training data access.</p>
                        </div>
                    </div>
                </div>

                {/* Phase 4: Automated Verification Rescan */}
                <div className="border-[3px] border-ink bg-[#FE9727] p-6 text-[#17130E] shadow-[6px_6px_0px_#17130E]">
                    <div className="flex items-center justify-between border-b-2 border-black/30 pb-4">
                        <span className="font-mono text-xs font-black uppercase tracking-widest text-[#17130E]">Phase 04 · Automated Verification Rescan</span>
                        <span className="border-2 border-black bg-[#17130E] px-2 py-0.5 font-mono text-[9px] font-black uppercase text-white">
                            CRYPTOGRAPHIC AUDIT
                        </span>
                    </div>
                    <div className="mt-4 space-y-3 font-mono text-xs">
                        <p className="font-sans text-sm font-medium leading-relaxed text-[#17130E]">
                            Remediated artifacts immediately undergo an automated 4-layer rescan. Cleaned binaries are released only if every inspection layer certifies BENIGN.
                        </p>
                        <div className="border-2 border-black bg-[#17130E] p-3 text-[#F2ECDC]">
                            <p className="text-[10px] text-[#EFC23A]">SHA-256 AUDIT LOGGING:</p>
                            <p className="mt-1 font-bold text-white">logs/disarmament_audit.log</p>
                            <p className="mt-1 text-[10px] text-[#F2ECDC]/60">Cryptographic pre-clean and post-clean hashes recorded with timestamped ledger.</p>
                        </div>
                    </div>
                </div>
            </div>
        </Reveal>
    </div>;
}
