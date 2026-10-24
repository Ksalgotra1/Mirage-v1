import React, { useEffect, useMemo, useState } from 'react';
import { FlaskConical, ShieldCheck, Layers, History, Trash2, Zap, Target, Cpu } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine, Cell, ResponsiveContainer, Tooltip, LabelList } from 'recharts';
import Reveal from '@/components/Reveal.jsx';
import { getScanHistory, clearScanHistory } from '@/lib/scanHistory.js';

/* ── Chart data ─────────────────────────────────────────── */
const REGIME_DATA = [
    { name: 'Dev-Mixed', subtitle: '40% Homo / 60% Unseen', ba: 70.0, fill: '#67C6D8' },
    { name: 'Dev-ZeroShot-Small', subtitle: '4 Diverse Families', ba: 100.0, fill: '#EFC23A' },
    { name: 'Dev-ZeroShot-Large', subtitle: 'Full 9-Model Zoo', ba: 70.8, fill: '#B07FD0' },
];

/* ── Animated threshold line via CSS ────────────────────── */
const ThresholdStyle = () => (
    <style>{`
        @keyframes drawLine { from { stroke-dashoffset: 2000; } to { stroke-dashoffset: 0; } }
        .threshold-gate line { stroke-dasharray: 10 6; stroke-dashoffset: 2000; animation: drawLine 1.6s ease-out 0.9s forwards; }
        .threshold-random line { stroke-dasharray: 4 4; stroke-dashoffset: 2000; animation: drawLine 1.8s ease-out 1.1s forwards; }
    `}</style>
);

function CustomTooltip({ active, payload }) {
    if (!active || !payload?.[0]) return null;
    const d = payload[0].payload;
    return (
        <div className="border-2 border-ink bg-[#17130E] px-4 py-3 font-mono text-xs text-[#F2ECDC] shadow-[3px_3px_0px_#17130E]">
            <p className="font-bold text-[#EFC23A]">{d.name}</p>
            <p className="mt-1 text-[#F2ECDC]/70">{d.subtitle}</p>
            <p className="mt-2 text-lg font-black">{d.ba.toFixed(1)}%</p>
        </div>
    );
}

function RegimeChart() {
    return (
        <div className="relative mt-6">
            <ThresholdStyle />
            <ResponsiveContainer width="100%" height={340}>
                <BarChart data={REGIME_DATA} margin={{ top: 40, right: 20, bottom: 50, left: 10 }} barCategoryGap="22%">
                    <CartesianGrid strokeDasharray="3 3" stroke="#F2ECDC20" vertical={false} />
                    <XAxis
                        dataKey="name" tick={false} axisLine={{ stroke: '#F2ECDC40' }} tickLine={false}
                    />
                    <YAxis domain={[0, 105]} tick={{ fill: '#F2ECDC90', fontFamily: 'monospace', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                    <Tooltip content={<CustomTooltip />} cursor={false} />
                    <ReferenceLine y={70} className="threshold-gate" stroke="#EFC23A" strokeWidth={2} label={{ value: 'Gate (70%)', position: 'right', fill: '#EFC23A', fontFamily: 'monospace', fontSize: 10 }} />
                    <ReferenceLine y={50} className="threshold-random" stroke="#F2ECDC50" strokeWidth={1.5} label={{ value: 'Random (50%)', position: 'right', fill: '#F2ECDC60', fontFamily: 'monospace', fontSize: 10 }} />
                    <Bar dataKey="ba" radius={[4, 4, 0, 0]} animationDuration={1200} animationEasing="ease-out">
                        {REGIME_DATA.map((entry) => (
                            <Cell key={entry.name} fill={entry.fill} />
                        ))}
                        <LabelList
                            dataKey="ba"
                            position="top"
                            formatter={(v) => `${v.toFixed(1)}%  [PASS]`}
                            style={{ fill: '#F2ECDC', fontFamily: 'monospace', fontSize: 12, fontWeight: 700 }}
                        />
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
            {/* X-axis custom labels */}
            <div className="mt-[-36px] flex justify-around px-[52px] pb-2">
                {REGIME_DATA.map((d) => (
                    <div key={d.name} className="text-center">
                        <p className="font-mono text-[10px] font-bold text-[#F2ECDC]">{d.name}</p>
                        <p className="font-mono text-[9px] text-[#F2ECDC]/50">({d.subtitle})</p>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ── Stat card ──────────────────────────────────────────── */
function Stat({ value, label, accent = '#EFC23A' }) {
    return (
        <div className="border-2 border-ink bg-[#F2ECDC] p-5 shadow-[4px_4px_0px_#17130E]">
            <p className="font-serif text-3xl font-black tracking-tight sm:text-4xl" style={{ color: accent }}>{value}</p>
            <p className="mt-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#17130E]/70">{label}</p>
        </div>
    );
}

/* ── Format badge ───────────────────────────────────────── */
function Fmt({ children }) {
    return <span className="inline-block border border-ink bg-[#E8DFC6] px-2 py-0.5 font-mono text-[10px] font-bold tracking-wider">{children}</span>;
}

/* ── Layer verdict dot (scan ledger) ────────────────────── */
function LayerDot({ active, label }) {
    return (
        <span title={label} className={`inline-flex h-5 w-5 items-center justify-center rounded-full border font-mono text-[8px] font-black ${active ? 'border-[#C4402E] bg-[#C4402E]/20 text-[#C4402E]' : 'border-[#17130E]/20 bg-[#E8DFC6] text-[#17130E]/40'}`}>
            {label}
        </span>
    );
}

/* ── Verdict badge ──────────────────────────────────────── */
function VerdictBadge({ verdict }) {
    const styles = {
        MALICIOUS: 'border-[#C4402E] bg-[#C4402E]/15 text-[#C4402E]',
        SUSPICIOUS: 'border-[#EFC23A] bg-[#EFC23A]/15 text-[#9a7a0a]',
        BENIGN: 'border-[#3a9a6b] bg-[#3a9a6b]/15 text-[#2a7a4b]',
        CLEAN: 'border-[#3a9a6b] bg-[#3a9a6b]/15 text-[#2a7a4b]',
    };
    return (
        <span className={`inline-block border px-2 py-0.5 font-mono text-[10px] font-black uppercase tracking-wider ${styles[verdict] || 'border-[#17130E]/30 bg-[#E8DFC6] text-[#17130E]/60'}`}>
            {verdict || 'UNKNOWN'}
        </span>
    );
}

/* ═══════════════════════════════════════════════════════════
   PAGE
   ═══════════════════════════════════════════════════════════ */
export default function BenchmarksPage() {
    const [history, setHistory] = useState([]);
    useEffect(() => { setHistory(getScanHistory()); }, []);

    const handleClearHistory = () => { clearScanHistory(); setHistory([]); };

    const fmtTime = (ts) => {
        const d = new Date(ts);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <div className="mx-auto max-w-7xl space-y-12 pb-8">

            {/* ── SECTION 1: HERO ─────────────────────────────── */}
            <Reveal className="border-2 border-ink bg-[#17130E] p-8 text-[#F2ECDC] shadow-[6px_6px_0px_#17130E] sm:p-12">
                <p className="inline-flex items-center gap-2 font-mono text-[11px] font-bold uppercase tracking-[0.2em] text-[#EFC23A]">
                    <FlaskConical className="h-4 w-4" /> Evaluation Evidence
                </p>
                <h1 className="mt-6 max-w-4xl text-4xl font-black leading-[0.92] sm:text-6xl">
                    Measured detection, tested across architectures.
                </h1>
                <p className="mt-5 max-w-2xl text-sm leading-relaxed text-[#F2ECDC]/75 sm:text-base">
                    Every number below traces to a documented experiment. No synthetic leaderboards, no borrowed benchmarks.
                </p>
                <div className="mt-8 flex flex-wrap gap-3">
                    {[['90% STEGO INTERCEPT', '#EFC23A'], ['84.44% F1 SCORE', '#67C6D8'], ['5 ARCHITECTURE FAMILIES', '#B07FD0']].map(([text, bg]) => (
                        <span key={text} className="border-2 border-[#F2ECDC]/30 px-4 py-2 font-mono text-[11px] font-black uppercase tracking-[0.14em]" style={{ backgroundColor: bg + '25', color: bg }}>{text}</span>
                    ))}
                </div>
            </Reveal>

            {/* ── SECTION 2: DETECTION PERFORMANCE ────────────── */}
            <section className="space-y-8">
                <Reveal>
                    <p className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#2F4FE0]">Layer 3a · GF-Fourpart Steganalysis</p>
                    <h2 className="mt-2 text-3xl font-black sm:text-5xl">Weight-space detection performance</h2>
                </Reveal>

                {/* Stat cards */}
                <Reveal delay={0.06}>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        <Stat value="90%" label="Stego Intercept Rate (18/20 held-out)" accent="#C4402E" />
                        <Stat value="86.36%" label="Detection Recall (66-sample diagnostic)" />
                        <Stat value="82.61%" label="Precision" accent="#2F4FE0" />
                        <Stat value="84.44%" label="F1 Score" accent="#17130E" />
                    </div>
                </Reveal>

                {/* Chart */}
                <Reveal delay={0.12} className="border-2 border-ink bg-[#17130E] p-5 shadow-[5px_5px_0px_#17130E] sm:p-8">
                    <div className="mb-4 flex items-center justify-between border-b border-[#F2ECDC]/15 pb-4">
                        <div>
                            <p className="font-mono text-[11px] font-bold uppercase tracking-[0.16em] text-[#EFC23A]">Benchmark Results Across Experimental Regimes</p>
                            <p className="mt-1 font-mono text-[10px] text-[#F2ECDC]/50">Held-out balanced accuracy · 36-model zero-shot evaluation</p>
                        </div>
                        <span className="hidden items-center gap-1.5 rounded-full border border-[#EFC23A]/40 bg-[#EFC23A]/10 px-3 py-1 font-mono text-[9px] font-bold text-[#EFC23A] sm:inline-flex">
                            ALL PASS
                        </span>
                    </div>
                    <RegimeChart />
                </Reveal>
            </section>

            {/* ── SECTION 3: SERIALIZATION LAYERS ─────────────── */}
            <section className="space-y-6">
                <Reveal>
                    <p className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#2F4FE0]">Serialization Defense · Layers 1 & 2</p>
                    <h2 className="mt-2 text-3xl font-black sm:text-5xl">Opcode inspection and structural classification</h2>
                </Reveal>

                <div className="grid gap-5 md:grid-cols-2">
                    {/* L1 */}
                    <Reveal delay={0.04} className="border-2 border-ink bg-[#F2ECDC] p-6 shadow-[5px_5px_0px_#17130E]">
                        <div className="flex items-start justify-between">
                            <p className="font-mono text-[11px] font-bold uppercase tracking-[0.16em] text-[#2F4FE0]">Layer 1 — Opcode Scanner</p>
                            <ShieldCheck className="h-5 w-5 text-[#2F4FE0]" />
                        </div>
                        <div className="mt-5 grid grid-cols-3 gap-3">
                            <div><p className="font-serif text-2xl font-black text-[#17130E]">100%</p><p className="font-mono text-[9px] font-bold uppercase text-[#17130E]/60">Precision</p></div>
                            <div><p className="font-serif text-2xl font-black text-[#17130E]">88%</p><p className="font-mono text-[9px] font-bold uppercase text-[#17130E]/60">Recall</p></div>
                            <div><p className="font-serif text-2xl font-black text-[#EFC23A]">&lt;4.2ms</p><p className="font-mono text-[9px] font-bold uppercase text-[#17130E]/60">Avg Latency</p></div>
                        </div>
                        <p className="mt-5 text-sm leading-relaxed text-[#17130E]/75">Rule-based pickle opcode inspection with ModelScan primary and fallback opcode walker. Catches os.system, subprocess, eval, exec, __import__, and platform-specific targets.</p>
                        <div className="mt-4 flex flex-wrap gap-1.5">
                            {['.pt', '.pkl', '.h5', '.pth', '.bin'].map((f) => <Fmt key={f}>{f}</Fmt>)}
                        </div>
                    </Reveal>

                    {/* L2 */}
                    <Reveal delay={0.08} className="border-2 border-ink bg-[#F2ECDC] p-6 shadow-[5px_5px_0px_#17130E]">
                        <div className="flex items-start justify-between">
                            <p className="font-mono text-[11px] font-bold uppercase tracking-[0.16em] text-[#2F4FE0]">Layer 2 — XGBoost Classifier</p>
                            <Cpu className="h-5 w-5 text-[#2F4FE0]" />
                        </div>
                        <div className="mt-5 grid grid-cols-2 gap-3">
                            <div><p className="font-serif text-2xl font-black text-[#17130E]">90.91%</p><p className="font-mono text-[9px] font-bold uppercase text-[#17130E]/60">Grouped Accuracy</p></div>
                            <div><p className="font-serif text-2xl font-black text-[#17130E]">0.90</p><p className="font-mono text-[9px] font-bold uppercase text-[#17130E]/60">ROC-AUC</p></div>
                        </div>
                        <p className="mt-5 text-sm leading-relaxed text-[#17130E]/75">45-dimension feature vector (28 opcode unigrams, 14 bigrams, 3 structural ratios) fed to a 100-tree XGBoost ensemble. Template-grouped 5-fold cross-validation on 54 fixture artifacts.</p>
                        <div className="mt-4 flex flex-wrap gap-1.5">
                            {['.pt', '.pkl', '.pth'].map((f) => <Fmt key={f}>{f}</Fmt>)}
                        </div>
                    </Reveal>
                </div>
            </section>

            {/* ── SECTION 4: FUSION POLICY ────────────────────── */}
            <Reveal className="border-2 border-ink bg-[#F2ECDC] p-6 shadow-[5px_5px_0px_#17130E] sm:p-8">
                <div className="flex items-start justify-between">
                    <div>
                        <p className="font-mono text-[11px] font-bold uppercase tracking-[0.16em] text-[#2F4FE0]"><Layers className="mr-1.5 inline h-3.5 w-3.5" />Format-Aware Fusion Policy</p>
                        <h2 className="mt-2 text-2xl font-black sm:text-4xl">Adaptive evidence weighting</h2>
                    </div>
                </div>
                <div className="mt-6 overflow-x-auto">
                    <table className="w-full border-collapse font-mono text-xs">
                        <thead>
                            <tr className="border-b-2 border-ink">
                                <th className="px-3 py-2.5 text-left text-[10px] font-bold uppercase tracking-[0.14em] text-[#17130E]/60">Format</th>
                                <th className="px-3 py-2.5 text-center text-[10px] font-bold uppercase tracking-[0.14em] text-[#17130E]/60">L1</th>
                                <th className="px-3 py-2.5 text-center text-[10px] font-bold uppercase tracking-[0.14em] text-[#17130E]/60">L2</th>
                                <th className="px-3 py-2.5 text-center text-[10px] font-bold uppercase tracking-[0.14em] text-[#17130E]/60">L3a</th>
                                <th className="px-3 py-2.5 text-center text-[10px] font-bold uppercase tracking-[0.14em] text-[#17130E]/60">L3b</th>
                            </tr>
                        </thead>
                        <tbody>
                            {[
                                ['.pt / .pth / .pkl', 0.20, 0.30, 0.30, 0.20],
                                ['.safetensors', 0.00, 0.00, 0.60, 0.40],
                                ['.h5', 0.30, 0.00, 0.40, 0.30],
                            ].map(([fmt, l1, l2, l3a, l3b]) => (
                                <tr key={fmt} className="border-b border-ink/20 transition hover:bg-[#EFC23A]/10">
                                    <td className="px-3 py-3 font-bold">{fmt}</td>
                                    {[l1, l2, l3a, l3b].map((w, i) => (
                                        <td key={i} className={`px-3 py-3 text-center ${w === 0 ? 'text-[#17130E]/25' : 'font-bold text-[#17130E]'}`}>
                                            {w.toFixed(2)}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div className="mt-5 flex items-start gap-2 border-l-4 border-[#C4402E] bg-[#C4402E]/8 px-4 py-3">
                    <Zap className="mt-0.5 h-4 w-4 shrink-0 text-[#C4402E]" />
                    <p className="font-mono text-[11px] leading-relaxed text-[#17130E]/80">
                        <strong>Conservative override:</strong> Any single positive layer forces the fused verdict to MALICIOUS regardless of the weighted score.
                    </p>
                </div>
            </Reveal>

            {/* ── SECTION 5: SCAN LEDGER ──────────────────────── */}
            <section className="space-y-5">
                <Reveal>
                    <div className="flex items-end justify-between">
                        <div>
                            <p className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#2F4FE0]">
                                <History className="mr-1.5 inline h-3.5 w-3.5" />Your Scan Ledger
                            </p>
                            <h2 className="mt-2 text-3xl font-black sm:text-5xl">Previous scan results</h2>
                        </div>
                        {history.length > 0 && (
                            <button
                                type="button"
                                onClick={handleClearHistory}
                                className="inline-flex items-center gap-1.5 rounded-full border-2 border-ink bg-[#F2ECDC] px-4 py-2 font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#C4402E] transition hover:-translate-y-0.5 hover:shadow-[2px_2px_0px_#17130E]"
                            >
                                <Trash2 className="h-3 w-3" /> Clear All
                            </button>
                        )}
                    </div>
                </Reveal>

                <Reveal delay={0.06} className="border-2 border-ink bg-[#F2ECDC] shadow-[5px_5px_0px_#17130E]">
                    {history.length === 0 ? (
                        <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
                            <History className="h-10 w-10 text-[#17130E]/20" />
                            <p className="font-mono text-xs font-bold uppercase tracking-[0.14em] text-[#17130E]/50">No scans recorded yet</p>
                            <p className="max-w-sm text-sm text-[#17130E]/60">Run a scan from the Scanner page. Every completed scan is saved here automatically.</p>
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full border-collapse">
                                <thead>
                                    <tr className="border-b-2 border-ink bg-[#E8DFC6]">
                                        {['Time', 'Filename', 'Format', 'Verdict', 'Risk', 'Layers'].map((h) => (
                                            <th key={h} className="px-4 py-3 text-left font-mono text-[10px] font-bold uppercase tracking-[0.14em] text-[#17130E]/60">{h}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {history.map((entry) => (
                                        <tr key={entry.id} className="border-b border-ink/15 transition hover:bg-[#EFC23A]/8">
                                            <td className="whitespace-nowrap px-4 py-3 font-mono text-[11px] text-[#17130E]/70">{fmtTime(entry.timestamp)}</td>
                                            <td className="max-w-[180px] truncate px-4 py-3 font-mono text-[11px] font-bold text-[#17130E]" title={entry.filename}>{entry.filename}</td>
                                            <td className="px-4 py-3"><Fmt>{entry.format}</Fmt></td>
                                            <td className="px-4 py-3"><VerdictBadge verdict={entry.verdict} /></td>
                                            <td className="px-4 py-3 font-mono text-[11px] font-bold text-[#17130E]">{entry.riskScore != null ? (entry.riskScore * 100).toFixed(1) + '%' : '—'}</td>
                                            <td className="px-4 py-3">
                                                <div className="flex gap-1">
                                                    <LayerDot active={entry.layers?.l1} label="L1" />
                                                    <LayerDot active={entry.layers?.l2} label="L2" />
                                                    <LayerDot active={entry.layers?.l3a} label="3a" />
                                                    <LayerDot active={entry.layers?.l3b} label="3b" />
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </Reveal>
            </section>
        </div>
    );
}
