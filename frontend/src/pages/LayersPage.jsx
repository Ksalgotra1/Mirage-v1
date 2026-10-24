import React, { useState } from 'react';
import Layer1ScanAnimation from '@/components/Layer1ScanAnimation.jsx';
import Layer2ScanAnimation from '@/components/Layer2ScanAnimation.jsx';
import Layer3aScanAnimation from '@/components/Layer3aScanAnimation.jsx';
import Layer3bScanAnimation from '@/components/Layer3bScanAnimation.jsx';
import { Cpu, Terminal, Sparkles, Layers, ShieldCheck } from 'lucide-react';
import Reveal from '@/components/Reveal.jsx';

const TABS = [
    { id: 'l1', num: '01', name: 'STATIC OPCODE', sub: 'Zero-Host Deserialization' },
    { id: 'l2', num: '02', name: 'XGBOOST STATISTICAL', sub: '43-D N-Gram Ensemble' },
    { id: 'l3a', num: '03a', name: 'GALOIS STEGANALYSIS', sub: '512×512 GF + 5-Signals' },
    { id: 'l3b', num: '03b', name: 'SPECTRAL INVARIANTS', sub: 'Reference-Free SVD & Kurtosis' },
];

export default function LayersPage() {
    const [activeTab, setActiveTab] = useState('l1');

    return (
        <div className="space-y-10">
            {/* Header banner */}
            <Reveal className="border-2 border-ink bg-[#F2ECDC] p-8 md:p-10">
                <div className="inline-flex items-center gap-2 rounded-full border border-ink bg-[#EFC23A] px-3.5 py-1 font-mono text-[11px] font-extrabold uppercase tracking-[0.16em] mb-4">
                    <Layers className="h-3.5 w-3.5" />
                    INTERACTIVE DEFENSE SANDBOX
                </div>
                <h1 className="font-serif text-3xl sm:text-5xl font-black tracking-tight text-[#17130E] leading-tight">
                    Four Layers. <span className="italic font-medium text-[#2F4FE0]">Zero Compromise.</span>
                </h1>
                <p className="mt-3 font-sans text-sm sm:text-base text-[#17130E]/80 max-w-3xl leading-relaxed">
                    Test each defense layer in isolation using our live interactive simulation sandboxes. Trigger malicious exploit payloads, borderline quantization anomalies, and clean control models to observe real-time feature extraction and rubber-stamp verdicts.
                </p>
            </Reveal>

            {/* Tab Selector */}
            <Reveal delay={0.08} role="tablist" aria-label="Defense layers" className="grid grid-cols-2 lg:grid-cols-4 gap-3 border-2 border-ink bg-[#17130E] p-2">
                {TABS.map((t) => (
                    <button
                        key={t.id}
                        onClick={() => setActiveTab(t.id)}
                        role="tab"
                        aria-selected={activeTab === t.id}
                        aria-controls="layer-sandbox"
                        className={`text-left p-4 transition border-2 ${
                            activeTab === t.id
                                ? 'bg-[#EFC23A] text-[#17130E] border-ink shadow-[2px_2px_0px_#F2ECDC]'
                                : 'bg-[#17130E] text-[#F2ECDC]/75 border-transparent hover:text-[#F2ECDC] hover:bg-[#251f18]'
                        }`}
                    >
                        <div className="font-mono text-xs font-bold opacity-60">LAYER {t.num}</div>
                        <div className="font-serif text-base font-bold tracking-tight mt-0.5">{t.name}</div>
                        <div className="font-mono text-[10px] tracking-wider opacity-80 mt-1">{t.sub}</div>
                    </button>
                ))}
            </Reveal>

            {/* Interactive Visualizer Canvas */}
            <Reveal delay={0.12} id="layer-sandbox" role="tabpanel" className="border-2 border-ink bg-[#E8DFC6] p-4 sm:p-8">
                {activeTab === 'l1' && <Layer1ScanAnimation />}
                {activeTab === 'l2' && <Layer2ScanAnimation />}
                {activeTab === 'l3a' && <Layer3aScanAnimation />}
                {activeTab === 'l3b' && <Layer3bScanAnimation />}
            </Reveal>
        </div>
    );
}
