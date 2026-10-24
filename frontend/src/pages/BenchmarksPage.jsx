import React, { useEffect, useMemo, useState } from 'react';
import { FlaskConical, ShieldCheck, Layers, History, Trash2, Zap, Target, Cpu, Radar, Activity, CheckCircle2, ShieldAlert, Binary, Check } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine, Cell, ResponsiveContainer, Tooltip, LabelList } from 'recharts';
import Reveal from '@/components/Reveal.jsx';
import { getScanHistory, clearScanHistory } from '@/lib/scanHistory.js';

/* ── Chart data (attack bit-depths & verification suites) ── */
const REGIME_DATA = [
    { name: 'Composite Zoo', subtitle: 'Blind Zoo (x ∈ {4, 8})', ba: 93.75, fill: '#EFC23A' },
    { name: 'Stress Test (Fam A-D)', subtitle: '60 Multi-Family Samples', ba: 98.0, fill: '#67C6D8' },
    { name: 'Byte LSB (x = 8)', subtitle: 'Full Byte Plane Q3', ba: 95.0, fill: '#22A093' },
    { name: 'Nibble LSB (x = 4)', subtitle: 'Mantissa Nibble N0', ba: 92.5, fill: '#FE9727' },
    { name: 'Sparse LSB (x ≤ 2)', subtitle: 'FFT Block Periodicity', ba: 100.0, fill: '#B07FD0' },
];

/* ── Multi-Rate Stego Matrix by Payload Bit-Depth (x) ───── */
const MULTIRATE_MATRIX = [
    {
        depth: 'x = 0 bits',
        plane: 'Clean Mantissa',
        payload: 'Clean Foundation Checkpoints (Unmodified)',
        recall: '—',
        specificity: '100.0% (10/10)',
        ba: '100.0%',
        engine: 'Parity-Compensated TVD Step-Cliff (b0 Isolation)',
        status: 'PASS',
        badge: 'SPECIFICITY GATE',
        badgeColor: '#15803D',
    },
    {
        depth: 'x = 1 or 2 bits',
        plane: 'Mantissa b1..b0',
        payload: 'Sparse Periodic / Sentinel Bitstream',
        recall: '100.0%',
        specificity: '100.0%',
        ba: '100.0%',
        engine: 'O(M log M) FFT Block Autocorrelation (z_per)',
        status: 'PASS',
        badge: 'PERIODICITY GATE',
        badgeColor: '#2F4FE0',
    },
    {
        depth: 'x = 4 bits',
        plane: 'Mantissa Nibble N0 (b3..b0)',
        payload: 'Structured / Shellcode / ASCII Injections',
        recall: '85.0% (17/20)',
        specificity: '100.0%',
        ba: '92.50%',
        engine: '1D Self-Calibrated Nibble TVD Step-Cliff (z_stat)',
        status: 'PASS',
        badge: 'CLIFF GATE',
        badgeColor: '#FE9727',
    },
    {
        depth: 'x = 8 bits',
        plane: 'Byte-Aligned Q3 (b7..b0)',
        payload: 'Full Byte Stream / Binary File Container Drop',
        recall: '90.0% (18/20)',
        specificity: '100.0%',
        ba: '95.00%',
        engine: '2D Contrastive GF-Fourpart Net + Cosine Margin',
        status: 'PASS',
        badge: 'METRIC GATE',
        badgeColor: '#22A093',
    },
    {
        depth: 'x = 16 bits',
        plane: 'Low Mantissa Half (b15..b0)',
        payload: 'Massive Exfiltration / Encrypted Archive Embed',
        recall: '100.0% (20/20)',
        specificity: '100.0%',
        ba: '100.0%',
        engine: 'Mantissa Entropy Divergence + Variance Collapse',
        status: 'PASS',
        badge: 'ENTROPY GATE',
        badgeColor: '#B07FD0',
    },
    {
        depth: 'Composite Zoo',
        plane: 'Multi-Rate Stego (x in {4, 8})',
        payload: 'Multi-Family Blind Zero-Shot Evaluation (50 Samples)',
        recall: '87.5% (35/40)',
        specificity: '100.0% (10/10)',
        ba: '93.75%',
        engine: 'Format-Aware Multi-Layer Wolpert Meta-Fusion',
        status: 'PASS',
        badge: 'OVERALL PASS',
        badgeColor: '#EFC23A',
    },
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
        <div className="pointer-events-none z-50 border-2 border-ink bg-[#17130E] px-4 py-3 font-mono text-xs text-[#F2ECDC] shadow-[4px_4px_0px_#17130E]">
            <p className="font-bold text-[#EFC23A]">{d.name}</p>
            <p className="mt-1 text-[#F2ECDC]/70">{d.subtitle}</p>
            <p className="mt-2 text-base font-black text-[#F2ECDC]">{d.ba.toFixed(1)}% Balanced Acc</p>
        </div>
    );
}

function CustomXAxisTick({ x, y, payload }) {
    const entry = REGIME_DATA[payload?.index];
    if (!entry) return null;
    return (
        <g transform={`translate(${x},${y + 14})`}>
            <text
                x={0}
                y={0}
                textAnchor="middle"
                fill="#F2ECDC"
                fontFamily="monospace"
                fontSize={10.5}
                fontWeight={700}
            >
                {entry.name}
            </text>
            <text
                x={0}
                y={15}
                textAnchor="middle"
                fill="#F2ECDC"
                fillOpacity={0.55}
                fontFamily="monospace"
                fontSize={9}
            >
                ({entry.subtitle})
            </text>
        </g>
    );
}

function RegimeChart() {
    return (
        <div className="relative mt-6">
            <ThresholdStyle />
            <ResponsiveContainer width="100%" height={350}>
                <BarChart data={REGIME_DATA} margin={{ top: 40, right: 85, bottom: 45, left: 10 }} barCategoryGap="18%">
                    <CartesianGrid strokeDasharray="3 3" stroke="#F2ECDC20" vertical={false} />
                    <XAxis
                        dataKey="name"
                        interval={0}
                        tick={<CustomXAxisTick />}
                        axisLine={{ stroke: '#F2ECDC40' }}
                        tickLine={false}
                    />
                    <YAxis domain={[0, 105]} tick={{ fill: '#F2ECDC90', fontFamily: 'monospace', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(242, 236, 220, 0.04)' }} wrapperStyle={{ pointerEvents: 'none', zIndex: 50 }} offset={14} />
                    <ReferenceLine y={70} className="threshold-gate" stroke="#EFC23A" strokeWidth={2} label={{ value: 'Gate (70%)', position: 'right', fill: '#EFC23A', fontFamily: 'monospace', fontSize: 10 }} />
                    <ReferenceLine y={50} className="threshold-random" stroke="#F2ECDC50" strokeWidth={1.5} label={{ value: 'Random (50%)', position: 'right', fill: '#F2ECDC60', fontFamily: 'monospace', fontSize: 10 }} />
                    <Bar dataKey="ba" radius={[4, 4, 0, 0]} animationDuration={1200} animationEasing="ease-out">
                        {REGIME_DATA.map((entry) => (
                            <Cell key={entry.name} fill={entry.fill} />
                        ))}
                        <LabelList
                            dataKey="ba"
                            position="top"
                            formatter={(v) => `${v.toFixed(1)}% [PASS]`}
                            style={{ fill: '#F2ECDC', fontFamily: 'monospace', fontSize: 11, fontWeight: 700, textAnchor: 'middle' }}
                        />
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
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
function Fmt({ children, color = 'default' }) {
    const variants = {
        default: 'border-ink bg-[#E8DFC6] text-[#17130E]',
        red: 'border-ink bg-[#FDF2F0] text-[#B91C1C] hover:bg-[#D93829] hover:text-white',
        orange: 'border-ink bg-[#FFF7ED] text-[#C2410C] hover:bg-[#EA580C] hover:text-white',
        green: 'border-ink bg-[#F0FDF4] text-[#15803D] hover:bg-[#15803D] hover:text-white',
    };
    return (
        <span className={`inline-block border-2 ${variants[color] || variants.default} px-2.5 py-0.5 font-mono text-[10px] font-black tracking-wider shadow-[1.5px_1.5px_0px_#17130E] transition-all duration-150`}>
            {children}
        </span>
    );
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
                    <FlaskConical className="h-4 w-4" /> Official v5 Evaluation Evidence
                </p>
                <h1 className="mt-6 max-w-4xl text-4xl font-black leading-[0.92] sm:text-6xl">
                    Measured detection, tested across architectures.
                </h1>
                <p className="mt-5 max-w-2xl text-sm leading-relaxed text-[#F2ECDC]/75 sm:text-base">
                    Every number below traces to our official locked verification benchmarks. 10 unseen foundation architectures, 0% training overlap, multi-rate payload depths (x=1 to 16 bits), and 143/143 passing test suites.
                </p>
                <div className="mt-8 flex flex-wrap gap-3">
                    {[['93.75% ZERO-SHOT BA', '#EFC23A'], ['100% CLEAN SPECIFICITY', '#67C6D8'], ['87.5% STEGO RECALL', '#C4402E'], ['143/143 PYTEST PASSING', '#B07FD0']].map(([text, bg]) => (
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

                {/* Primary Stat cards */}
                <Reveal delay={0.06}>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        <Stat value="93.75%" label="Zero-Shot Balanced Accuracy (10-Model Benchmark)" accent="#EFC23A" />
                        <Stat value="100.0%" label="Clean Specificity (0 False Positives / 10 Models)" accent="#67C6D8" />
                        <Stat value="87.5%" label="Stego Recall (35/40 True Positives)" accent="#C4402E" />
                        <Stat value="93.33%" label="F1 Score (100% Precision)" accent="#2F4FE0" />
                    </div>
                </Reveal>

                {/* Chart */}
                <Reveal delay={0.12} className="border-2 border-ink bg-[#17130E] p-5 shadow-[5px_5px_0px_#17130E] sm:p-8">
                    <div className="mb-4 flex items-center justify-between border-b border-[#F2ECDC]/15 pb-4">
                        <div>
                            <p className="font-mono text-[11px] font-bold uppercase tracking-[0.16em] text-[#EFC23A]">Benchmark Results Across Attack Bit-Depths & Regimes</p>
                            <p className="mt-1 font-mono text-[10px] text-[#F2ECDC]/50">Held-out balanced accuracy across discrete LSB embedding rates (x in &#123;1, 2, 4, 8&#125;) & blind verification suites</p>
                        </div>
                        <span className="hidden items-center gap-1.5 rounded-full border border-[#EFC23A]/40 bg-[#EFC23A]/10 px-3 py-1 font-mono text-[9px] font-bold text-[#EFC23A] sm:inline-flex">
                            143/143 PASS
                        </span>
                    </div>
                    <RegimeChart />
                </Reveal>

                {/* Multi-Rate Stego Matrix by Payload Bit-Depth (x) */}
                <Reveal delay={0.16} className="overflow-hidden border-[3px] border-ink bg-[#FAF6EE] shadow-[6px_6px_0px_#17130E]">
                    <div className="border-b-[3px] border-ink bg-[#17130E] px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[#F2ECDC]">
                        <div className="flex items-center gap-2.5">
                            <Radar className="h-5 w-5 text-[#EFC23A] shrink-0" />
                            <div>
                                <p className="font-mono text-xs font-black uppercase tracking-[0.16em] text-[#F2ECDC]">Multi-Rate Payload Steganalysis Matrix</p>
                                <p className="font-mono text-[10px] text-[#F2ECDC]/60 mt-0.5">Empirical Detection Rates Across Discrete Payload Bit Depths (x in &#123;0, 1, 2, 4, 8, 16&#125;)</p>
                            </div>
                        </div>
                        <span className="self-start sm:self-auto border-2 border-ink bg-[#EFC23A] px-3 py-1 font-mono text-[10px] font-black uppercase tracking-wider text-[#17130E] shadow-[2px_2px_0px_#F2ECDC]">
                            50-Sample Evaluation Zoo
                        </span>
                    </div>

                    <div className="p-6 sm:p-8">
                        <h3 className="text-2xl font-black text-[#17130E]">Precision across mantissa depth regimes</h3>
                        <p className="mt-2 text-sm text-[#17130E]/70 max-w-3xl">
                            Each row evaluates detection sensitivity across distinct bit planes—from single-bit periodic markers to full-mantissa exfiltration containers. Clean baseline specificity remains pinned at 100% across all foundation checkpoints.
                        </p>

                        <div className="mt-6 overflow-x-auto border-2 border-ink bg-white shadow-[4px_4px_0px_#17130E]">
                            <table className="w-full border-collapse font-mono text-xs">
                                <thead>
                                    <tr className="border-b-2 border-ink bg-[#17130E] text-[#F2ECDC]">
                                        <th className="px-4 py-3 text-left text-[11px] font-black uppercase tracking-[0.14em]">PAYLOAD DEPTH (x)</th>
                                        <th className="px-4 py-3 text-left text-[11px] font-black uppercase tracking-[0.14em]">TARGET MANTISSA PLANE</th>
                                        <th className="px-4 py-3 text-center text-[11px] font-black uppercase tracking-[0.14em]">STEGO RECALL</th>
                                        <th className="px-4 py-3 text-center text-[11px] font-black uppercase tracking-[0.14em]">CLEAN SPECIFICITY</th>
                                        <th className="px-4 py-3 text-center text-[11px] font-black uppercase tracking-[0.14em]">BALANCED ACCURACY</th>
                                        <th className="px-4 py-3 text-left text-[11px] font-black uppercase tracking-[0.14em]">PRIMARY FORENSIC ENGINE</th>
                                        <th className="px-4 py-3 text-center text-[11px] font-black uppercase tracking-[0.14em]">GATE</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y-2 divide-ink/20">
                                    {MULTIRATE_MATRIX.map((row) => (
                                        <tr key={row.depth} className="transition hover:bg-[#EFC23A]/10">
                                            <td className="px-4 py-3.5 font-black text-sm text-[#17130E] whitespace-nowrap">
                                                {row.depth}
                                            </td>
                                            <td className="px-4 py-3.5 text-[#17130E]/80">
                                                <span className="font-bold text-[#17130E]">{row.plane}</span>
                                                <p className="text-[10px] text-[#17130E]/60 mt-0.5">{row.payload}</p>
                                            </td>
                                            <td className="px-4 py-3.5 text-center font-black text-sm text-[#C4402E]">
                                                {row.recall}
                                            </td>
                                            <td className="px-4 py-3.5 text-center font-black text-sm text-[#15803D]">
                                                {row.specificity}
                                            </td>
                                            <td className="px-4 py-3.5 text-center font-black text-sm text-[#2F4FE0]">
                                                {row.ba}
                                            </td>
                                            <td className="px-4 py-3.5 text-[11px] font-semibold text-[#17130E]">
                                                {row.engine}
                                            </td>
                                            <td className="px-4 py-3.5 text-center whitespace-nowrap">
                                                <span className="inline-block border-2 border-black bg-[#15803D] px-2.5 py-0.5 text-[9px] font-black uppercase text-white shadow-[1.5px_1.5px_0px_#000]">
                                                    {row.status}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* Disarmament & Remediation Assurance Banner */}
                        <div className="mt-6 border-2 border-[#15803D] bg-[#F0FDF4] p-5 shadow-[3px_3px_0px_#17130E]">
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                <div className="flex items-center gap-2.5">
                                    <ShieldCheck className="h-6 w-6 text-[#15803D] shrink-0" />
                                    <div>
                                        <p className="font-mono text-xs font-black uppercase tracking-wider text-[#15803D]">Automated Disarmament & Verification Rescan Guarantee</p>
                                        <p className="mt-0.5 font-mono text-[11px] text-[#17130E]/80">
                                            100% of remediated models undergo automatic 4-layer rescan. Bitwise LSB zeroing preserves exact tensor L2 norms (||W_clean||_2 = ||W_0||_2) with zero degradation in inference activations.
                                        </p>
                                    </div>
                                </div>
                                <span className="self-start sm:self-auto border-2 border-black bg-[#15803D] px-3 py-1 font-mono text-[10px] font-black uppercase tracking-wider text-white shadow-[2px_2px_0px_#000] whitespace-nowrap">
                                    100% RESCAN PASS
                                </span>
                            </div>
                        </div>
                    </div>
                </Reveal>
            </section>

            {/* ── SECTION 3: MULTI-LAYER DEFENSE ─────────────── */}
            <section className="space-y-6">
                <Reveal>
                    <div className="inline-flex items-center gap-2 border-2 border-ink bg-[#17130E] text-[#F2ECDC] px-3.5 py-1 font-mono text-[11px] font-black uppercase tracking-[0.18em] shadow-[3px_3px_0px_#17130E]">
                        <Layers className="h-3.5 w-3.5 text-[#EFC23A]" />
                        Multi-Layer Defense · Tri-Color Brutalist Audit
                    </div>
                    <h2 className="mt-3 text-3xl font-black sm:text-5xl">Opcode inspection, structural classification & weight geometry</h2>
                </Reveal>

                <div className="grid gap-6 md:grid-cols-2">
                    {/* L1 - FULL SOLID RED (#E14230) */}
                    <Reveal delay={0.04} className="group overflow-hidden border-[3px] border-ink bg-[#E14230] p-6 sm:p-7 text-white shadow-[6px_6px_0px_#17130E] transition-all duration-200 hover:-translate-y-1 hover:shadow-[10px_10px_0px_#17130E]">
                        <div className="flex items-center justify-between pb-5 border-b-2 border-black/30">
                            <div className="flex items-center gap-2.5">
                                <div className="flex h-8 w-8 items-center justify-center border-2 border-black bg-white text-[#E14230] shadow-[2px_2px_0px_#000]">
                                    <ShieldCheck className="h-5 w-5" />
                                </div>
                                <span className="font-mono text-sm font-black uppercase tracking-[0.14em] text-white">Layer 1 — Opcode Scanner</span>
                            </div>
                            <span className="border-2 border-black bg-[#17130E] px-2.5 py-1 font-mono text-[9px] font-black uppercase tracking-wider text-white shadow-[2px_2px_0px_#000]">
                                SERIALIZATION GATE
                            </span>
                        </div>

                        <div className="pt-6">
                            <div className="grid grid-cols-3 gap-3 border-b-2 border-black/30 pb-5">
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-white">100%</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-white/80">Precision</p>
                                </div>
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-white">88%</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-white/80">Recall</p>
                                </div>
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-[#FFE600]">&lt;4.2ms</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-white/80">Avg Latency</p>
                                </div>
                            </div>
                            <p className="mt-4 text-sm font-medium leading-relaxed text-white">
                                Rule-based pickle opcode inspection with ModelScan primary and fallback opcode walker. Catches <code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-xs font-black text-[#17130E] shadow-[1.5px_1.5px_0px_#000]">os.system</code>, <code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-xs font-black text-[#17130E] shadow-[1.5px_1.5px_0px_#000]">subprocess</code>, <code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-xs font-black text-[#17130E] shadow-[1.5px_1.5px_0px_#000]">eval</code>, <code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-xs font-black text-[#17130E] shadow-[1.5px_1.5px_0px_#000]">exec</code>, and platform-specific exploits.
                            </p>
                            <div className="mt-5 flex flex-wrap gap-2 pt-3 border-t-2 border-black/20">
                                {['.pt', '.pkl', '.h5', '.pth', '.bin'].map((f) => (
                                    <span key={f} className="inline-block border-2 border-black bg-white px-2.5 py-1 font-mono text-[10px] font-black tracking-wider text-[#17130E] shadow-[2px_2px_0px_#000] hover:bg-[#17130E] hover:text-white transition-colors cursor-default">
                                        {f}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </Reveal>

                    {/* L2 - FULL SOLID ORANGE (#FE9727) */}
                    <Reveal delay={0.08} className="group overflow-hidden border-[3px] border-ink bg-[#FE9727] p-6 sm:p-7 text-[#17130E] shadow-[6px_6px_0px_#17130E] transition-all duration-200 hover:-translate-y-1 hover:shadow-[10px_10px_0px_#17130E]">
                        <div className="flex items-center justify-between pb-5 border-b-2 border-black/30">
                            <div className="flex items-center gap-2.5">
                                <div className="flex h-8 w-8 items-center justify-center border-2 border-black bg-white text-[#FE9727] shadow-[2px_2px_0px_#000]">
                                    <Cpu className="h-5 w-5 text-[#17130E]" />
                                </div>
                                <span className="font-mono text-sm font-black uppercase tracking-[0.14em] text-[#17130E]">Layer 2 — XGBoost Classifier</span>
                            </div>
                            <span className="border-2 border-black bg-[#17130E] px-2.5 py-1 font-mono text-[9px] font-black uppercase tracking-wider text-white shadow-[2px_2px_0px_#000]">
                                43-D STATISTICAL ML
                            </span>
                        </div>

                        <div className="pt-6">
                            <div className="grid grid-cols-2 gap-4 border-b-2 border-black/30 pb-5">
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-[#17130E]">90.91%</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-[#17130E]/80">Grouped Accuracy</p>
                                </div>
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-[#17130E]">0.90</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-[#17130E]/80">ROC-AUC</p>
                                </div>
                            </div>
                            <p className="mt-4 text-sm font-medium leading-relaxed text-[#17130E]">
                                43-dimension feature vector (26 opcode unigrams, 14 bigrams, 3 structural ratios) fed to a 100-tree XGBoost ensemble. Template-grouped 5-fold cross-validation on 54 fixture artifacts with zero data leakage.
                            </p>
                            <div className="mt-5 flex flex-wrap gap-2 pt-3 border-t-2 border-black/20">
                                {['.pt', '.pkl', '.pth'].map((f) => (
                                    <span key={f} className="inline-block border-2 border-black bg-white px-2.5 py-1 font-mono text-[10px] font-black tracking-wider text-[#17130E] shadow-[2px_2px_0px_#000] hover:bg-[#17130E] hover:text-white transition-colors cursor-default">
                                        {f}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </Reveal>

                    {/* L3b - FULL SOLID GREEN (#0AAF55) */}
                    <Reveal delay={0.12} className="group overflow-hidden border-[3px] border-ink bg-[#0AAF55] p-6 sm:p-7 text-[#17130E] shadow-[6px_6px_0px_#17130E] transition-all duration-200 hover:-translate-y-1 hover:shadow-[10px_10px_0px_#17130E] md:col-span-2">
                        <div className="flex items-center justify-between pb-5 border-b-2 border-black/30">
                            <div className="flex items-center gap-2.5">
                                <div className="flex h-8 w-8 items-center justify-center border-2 border-black bg-white text-[#0AAF55] shadow-[2px_2px_0px_#000]">
                                    <Radar className="h-5 w-5 text-[#17130E]" />
                                </div>
                                <span className="font-mono text-sm font-black uppercase tracking-[0.14em] text-[#17130E]">Layer 3b — Weight Anomaly & Spectral Detector</span>
                            </div>
                            <span className="border-2 border-black bg-[#17130E] px-2.5 py-1 font-mono text-[9px] font-black uppercase tracking-wider text-white shadow-[2px_2px_0px_#000]">
                                REFERENCE-FREE SPECTRAL GEOMETRY
                            </span>
                        </div>

                        <div className="pt-6">
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 border-b-2 border-black/30 pb-5">
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-[#17130E]">2.80</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-[#17130E]/80">Meta Weight (L3b_Score)</p>
                                </div>
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-[#17130E]">&ge; 32</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-[#17130E]/80">Active SVD Min Dim</p>
                                </div>
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-[#17130E]">15.0×</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-[#17130E]/80">Neuron Thresh (P90 &ge; 3.0×)</p>
                                </div>
                                <div>
                                    <p className="font-serif text-3xl sm:text-4xl font-black text-[#17130E]">+0.40</p>
                                    <p className="mt-1 font-mono text-[9px] font-black uppercase tracking-wider text-[#17130E]/80">Consensus Bonus</p>
                                </div>
                            </div>
                            <p className="mt-4 text-sm font-medium leading-relaxed text-[#17130E]">
                                Reference-free geometric anomaly scanner. Features pruning-safe active-neuron L2 norm filtering (eliminates MobileNetV3 SE false positives), active-submatrix SVD rank collapse detection, and spatial-kernel kurtosis gating (<code className="border-2 border-black bg-white px-1.5 py-0.5 font-mono text-xs font-black text-[#17130E] shadow-[1.5px_1.5px_0px_#000]">kH × kW &gt; 1</code>, eliminating ResNeXt50 pointwise downsample false alarms).
                            </p>
                            <div className="mt-5 flex flex-wrap gap-2 pt-3 border-t-2 border-black/20">
                                {['.pt', '.pth', '.safetensors', '.h5', '.bin'].map((f) => (
                                    <span key={f} className="inline-block border-2 border-black bg-white px-2.5 py-1 font-mono text-[10px] font-black tracking-wider text-[#17130E] shadow-[2px_2px_0px_#000] hover:bg-[#17130E] hover:text-white transition-colors cursor-default">
                                        {f}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </Reveal>
                </div>
            </section>

            {/* ── SECTION 4: FUSION POLICY ────────────────────── */}
            <Reveal className="overflow-hidden border-[3px] border-ink bg-[#FAF6EE] shadow-[6px_6px_0px_#17130E]">
                {/* Brutalist Header Bar */}
                <div className="border-b-[3px] border-ink bg-[#17130E] px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[#F2ECDC]">
                    <div className="flex items-center gap-2.5">
                        <Layers className="h-5 w-5 text-[#FEC901] shrink-0" />
                        <div>
                            <p className="font-mono text-xs font-black uppercase tracking-[0.16em] text-[#F2ECDC]">Format-Aware Fusion Policy & Safety Anchors</p>
                            <p className="font-mono text-[10px] text-[#F2ECDC]/60 mt-0.5">Runtime Dynamic Re-Weighting Matrix</p>
                        </div>
                    </div>
                    <span className="self-start sm:self-auto border-2 border-ink bg-[#FEC901] px-3 py-1 font-mono text-[10px] font-black uppercase tracking-wider text-[#17130E] shadow-[2px_2px_0px_#F2ECDC]">
                        Adaptive Evidence Weighting
                    </span>
                </div>

                <div className="p-6 sm:p-8">
                    <h2 className="text-2xl font-black sm:text-4xl text-[#17130E]">Calibrated decision boundaries across formats</h2>
                    <p className="mt-2 text-sm text-[#17130E]/70 max-w-3xl">
                        Each serialization format directs weight to its specialized forensic layers. SafeTensors bypasses opcode inspectors entirely, routing 100% of conviction to weight-space tensor geometry and Galois-field steganalysis.
                    </p>

                    <div className="mt-6 overflow-x-auto border-2 border-ink bg-white shadow-[4px_4px_0px_#17130E]">
                        <table className="w-full border-collapse font-mono text-xs">
                            <thead>
                                <tr className="border-b-2 border-ink bg-[#17130E] text-[#F2ECDC]">
                                    <th className="px-4 py-3 text-left text-[11px] font-black uppercase tracking-[0.14em]">FORMAT FAMILY</th>
                                    <th className="px-4 py-3 text-center">
                                        <span className="inline-block border-2 border-black bg-[#E14230] px-2.5 py-1 text-[10px] font-black uppercase text-white shadow-[2px_2px_0px_#000]">
                                            L1 · RED
                                        </span>
                                    </th>
                                    <th className="px-4 py-3 text-center">
                                        <span className="inline-block border-2 border-black bg-[#FE9727] px-2.5 py-1 text-[10px] font-black uppercase text-[#17130E] shadow-[2px_2px_0px_#000]">
                                            L2 · ORANGE
                                        </span>
                                    </th>
                                    <th className="px-4 py-3 text-center">
                                        <span className="inline-block border-2 border-black bg-[#22A093] px-2.5 py-1 text-[10px] font-black uppercase text-white shadow-[2px_2px_0px_#000]">
                                            L3A · TEAL
                                        </span>
                                    </th>
                                    <th className="px-4 py-3 text-center">
                                        <span className="inline-block border-2 border-black bg-[#0AAF55] px-2.5 py-1 text-[10px] font-black uppercase text-white shadow-[2px_2px_0px_#000]">
                                            L3B · GREEN
                                        </span>
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="divide-y-2 divide-ink/20">
                                {[
                                    ['.pt / .pth / .pkl', 0.20, 0.30, 0.30, 0.20],
                                    ['.safetensors', 0.00, 0.00, 0.60, 0.40],
                                    ['.h5', 0.30, 0.00, 0.40, 0.30],
                                ].map(([fmt, l1, l2, l3a, l3b]) => (
                                    <tr key={fmt} className="transition hover:bg-[#FEC901]/15">
                                        <td className="px-4 py-3.5 font-black text-sm text-[#17130E]">{fmt}</td>
                                        <td className={`px-4 py-3.5 text-center font-black ${l1 === 0 ? 'text-[#17130E]/25' : 'text-[#E14230] text-sm'}`}>{l1.toFixed(2)}</td>
                                        <td className={`px-4 py-3.5 text-center font-black ${l2 === 0 ? 'text-[#17130E]/25' : 'text-[#FE9727] text-sm'}`}>{l2.toFixed(2)}</td>
                                        <td className={`px-4 py-3.5 text-center font-black ${l3a === 0 ? 'text-[#17130E]/25' : 'text-[#22A093] text-sm'}`}>{l3a.toFixed(2)}</td>
                                        <td className={`px-4 py-3.5 text-center font-black ${l3b === 0 ? 'text-[#17130E]/25' : 'text-[#0AAF55] text-sm'}`}>{l3b.toFixed(2)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Solid Full-Color Callout cards */}
                    <div className="mt-6 grid gap-4 sm:grid-cols-2">
                        {/* Solid Red Anchor Card (#E14230) */}
                        <div className="border-[3px] border-ink bg-[#E14230] p-5 text-white shadow-[4px_4px_0px_#17130E]">
                            <div className="flex items-center gap-2 mb-2.5">
                                <div className="flex h-6 w-6 items-center justify-center border-2 border-black bg-white text-[#E14230] shadow-[1.5px_1.5px_0px_#000]">
                                    <Zap className="h-3.5 w-3.5" />
                                </div>
                                <span className="font-mono text-xs font-black uppercase tracking-wider text-white">Safety Hard Anchors</span>
                            </div>
                            <p className="font-mono text-[11px] leading-relaxed text-white/95">
                                L1 positive forces prob &ge; <strong>0.999</strong>. L3a positive forces prob &ge; <strong>0.850</strong>. High-conviction L3b forces prob &ge; <strong>0.850</strong> (score &ge; 0.70) or prob &ge; <strong>0.500</strong> (score &ge; 0.35).
                            </p>
                        </div>

                        {/* Solid Green/Teal Meta Card (#22A093) */}
                        <div className="border-[3px] border-ink bg-[#22A093] p-5 text-white shadow-[4px_4px_0px_#17130E]">
                            <div className="flex items-center gap-2 mb-2.5">
                                <div className="flex h-6 w-6 items-center justify-center border-2 border-black bg-white text-[#22A093] shadow-[1.5px_1.5px_0px_#000]">
                                    <Cpu className="h-3.5 w-3.5" />
                                </div>
                                <span className="font-mono text-xs font-black uppercase tracking-wider text-white">Meta-Learner Coefficients</span>
                            </div>
                            <p className="font-mono text-[11px] leading-relaxed text-white/95">
                                L1: <strong>1.12</strong> | L2: <strong>3.55</strong> | L3a Margin: <strong>0.57</strong> | L3a Conviction: <strong>0.46</strong> | L3b Score: <strong>2.80</strong> | L1×L2: <strong>1.09</strong>.
                            </p>
                        </div>
                    </div>
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
