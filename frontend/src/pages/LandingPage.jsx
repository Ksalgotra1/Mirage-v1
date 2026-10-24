import React from 'react';
import { ArrowUpRight, BookOpen, FlaskConical, LockKeyhole, Radar, ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';
import Reveal from '@/components/Reveal.jsx';
import HexScanPanel from '@/components/HexScanPanel.jsx';

const JOURNEY = [
    { number: '01', title: 'Scanner', description: 'Inspect the model before it runs.', detail: 'Trace pickle opcodes, grammar signatures, tensor byte planes, and spectral invariants without loading untrusted code.', href: '/scanner', icon: Radar, accent: 'bg-[#EFC23A]' },
    { number: '02', title: 'Disarm', description: 'Contain confirmed threats without losing the evidence.', detail: 'Operate from a quarantined scan session, preserve the case file, and verify the cleaned artifact before release.', href: '/disarm', icon: LockKeyhole, accent: 'bg-[#C4402E] text-[#F2ECDC]' },
    { number: '03', title: 'Benchmarks', description: 'Measure the defense before you trust the verdict.', detail: 'Review the blind benchmark matrix, false-positive rate, and threat-vector coverage behind each result.', href: '/benchmarks', icon: FlaskConical, accent: 'bg-[#2F4FE0] text-[#F2ECDC]' },
    { number: '04', title: 'Documentation', description: 'Read and test the four-layer method.', detail: 'Move through the field manual, decision boundaries, inputs, and on-demand demonstrations for every layer.', href: '/docs', icon: BookOpen, accent: 'bg-[#72C9AE]' },
];

const LAYERS = [
    ['01', 'STATIC OPCODE', 'Safe pickle instruction tracing'],
    ['02', 'STATISTICAL GRAMMAR', '43-dimensional structural risk profile'],
    ['03A', 'TENSOR STEGANALYSIS', 'Galois field byte-plane evidence'],
    ['03B', 'SPECTRAL INVARIANTS', 'Reference-free backdoor geometry'],
];

export default function LandingPage() {
    return (
        <div className="space-y-16 pb-8 sm:space-y-24">
            <section className="relative overflow-hidden border-2 border-ink bg-[#F2ECDC] px-6 py-14 shadow-[6px_6px_0px_#17130E] sm:px-10 sm:py-20 lg:px-14">
                <div className="absolute right-0 top-0 hidden h-5/6 w-2/5 border-b-2 border-l-2 border-ink md:block overflow-hidden" aria-hidden="true">
                    <HexScanPanel />
                </div>
                <div className="relative max-w-5xl">
                    <p className="mb-8 inline-flex -rotate-1 border-2 border-ink bg-[#2F4FE0] px-4 py-2 font-mono text-[11px] font-bold uppercase tracking-[0.2em] text-[#F2ECDC] shadow-[3px_3px_0px_#17130E]">
                        PROJECT-CLIMAX / MODEL SECURITY GATE
                    </p>
                    <h1 className="max-w-4xl font-serif text-5xl font-black leading-[0.9] tracking-tight text-[#17130E] sm:text-7xl lg:text-8xl">
                        Inspect the model before it inspects you.
                    </h1>
                    <p className="mt-7 max-w-2xl text-base leading-relaxed text-[#17130E]/85 sm:text-xl">
                        A four-layer checkpoint defense for unsafe deserialization, structural anomalies, hidden tensor payloads, and spectral backdoors—built around evidence, not guesswork.
                    </p>
                    <div className="mt-9 flex flex-col gap-3 sm:flex-row">
                        <Link to="/scanner" className="inline-flex items-center justify-center gap-2 rounded-full border-2 border-ink bg-[#17130E] px-6 py-3 font-mono text-xs font-bold uppercase tracking-[0.18em] text-[#F2ECDC] transition hover:-translate-y-0.5 hover:shadow-[3px_3px_0px_#17130E]">
                            START A SCAN <ArrowUpRight className="h-4 w-4" />
                        </Link>
                        <Link to="/docs" className="inline-flex items-center justify-center gap-2 rounded-full border-2 border-ink bg-[#F2ECDC] px-6 py-3 font-mono text-xs font-bold uppercase tracking-[0.18em] text-[#17130E] transition hover:-translate-y-0.5 hover:shadow-[3px_3px_0px_#17130E]">
                            READ THE METHOD <BookOpen className="h-4 w-4" />
                        </Link>
                    </div>
                </div>
                <div className="relative mt-14 grid gap-px border-2 border-ink bg-[#17130E] sm:grid-cols-3">
                    {['4 LAYERS', 'ZERO UNSAFE LOADS', 'ONE VERIFIED VERDICT'].map((metric) => <p key={metric} className="bg-[#F2ECDC] px-4 py-3 text-center font-mono text-[10px] font-bold uppercase tracking-[0.15em]">{metric}</p>)}
                </div>
            </section>

            <section aria-labelledby="journey-title" className="space-y-7">
                <Reveal>
                    <p className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#2F4FE0]">CASE-FILE WORKFLOW</p>
                    <div className="mt-3 flex flex-col justify-between gap-5 md:flex-row md:items-end">
                        <h2 id="journey-title" className="max-w-3xl text-4xl font-black sm:text-6xl">A security decision that you can follow.</h2>
                        <p className="max-w-sm text-sm leading-relaxed text-[#17130E]/70">Start with the artifact, follow the evidence, and only then choose remediation.</p>
                    </div>
                </Reveal>
                <div className="grid gap-5 md:grid-cols-2">
                    {JOURNEY.map((item, index) => {
                        const Icon = item.icon;
                        return <Reveal key={item.title} delay={index * 0.08}>
                            <Link to={item.href} className="group block h-full border-2 border-ink bg-[#F2ECDC] p-6 shadow-[5px_5px_0px_#17130E] transition hover:-translate-y-1 hover:shadow-[8px_8px_0px_#17130E]">
                                <div className="flex items-start justify-between gap-4">
                                    <span className={`flex h-12 w-12 items-center justify-center rounded-full border-2 border-ink font-serif text-xl font-black ${item.accent}`}>{item.number}</span>
                                    <Icon className="h-5 w-5 text-[#17130E]/70 transition group-hover:text-[#2F4FE0]" />
                                </div>
                                <h3 className="mt-10 text-3xl font-black">{item.title}</h3>
                                <p className="mt-2 text-lg font-semibold leading-snug">{item.description}</p>
                                <p className="mt-5 border-t-2 border-dashed border-ink/60 pt-5 text-sm leading-relaxed text-[#17130E]/75">{item.detail}</p>
                                <span className="mt-6 inline-flex items-center gap-2 font-mono text-[11px] font-bold uppercase tracking-[0.18em]">OPEN {item.title} <ArrowUpRight className="h-4 w-4" /></span>
                            </Link>
                        </Reveal>;
                    })}
                </div>
            </section>

            <Reveal className="border-2 border-ink bg-[#17130E] p-6 text-[#F2ECDC] shadow-[6px_6px_0px_#17130E] sm:p-10">
                <div className="flex flex-col justify-between gap-8 lg:flex-row lg:items-end">
                    <div><p className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#EFC23A]">THE FOUR-LAYER DOSSIER</p><h2 className="mt-4 max-w-2xl text-4xl font-black sm:text-5xl">Independent checks. One evidence trail.</h2></div>
                    <Link to="/docs" className="inline-flex items-center gap-2 self-start rounded-full border-2 border-[#F2ECDC] px-5 py-3 font-mono text-xs font-bold uppercase tracking-[0.16em] transition hover:bg-[#F2ECDC] hover:text-[#17130E]">OPEN FIELD MANUAL <ArrowUpRight className="h-4 w-4" /></Link>
                </div>
                <div className="mt-10 grid gap-px border-2 border-[#F2ECDC]/50 bg-[#F2ECDC]/50 sm:grid-cols-2 lg:grid-cols-4">
                    {LAYERS.map(([number, title, detail]) => <div key={number} className="bg-[#17130E] p-4"><p className="font-mono text-xs font-bold text-[#EFC23A]">{number}</p><h3 className="mt-4 font-mono text-sm font-bold uppercase tracking-wide">{title}</h3><p className="mt-2 text-xs leading-relaxed text-[#F2ECDC]/70">{detail}</p></div>)}
                </div>
            </Reveal>

            <Reveal className="border-2 border-ink bg-[#EFC23A] p-8 shadow-[6px_6px_0px_#17130E] sm:p-12">
                <p className="font-mono text-xs font-bold uppercase tracking-[0.2em]">READY FOR INTAKE</p>
                <div className="mt-5 flex flex-col justify-between gap-6 md:flex-row md:items-end"><h2 className="max-w-3xl text-4xl font-black sm:text-6xl">Make the checkpoint answer for itself.</h2><Link to="/scanner" className="inline-flex items-center justify-center gap-2 rounded-full border-2 border-ink bg-[#17130E] px-6 py-3 font-mono text-xs font-bold uppercase tracking-[0.18em] text-[#F2ECDC] transition hover:-translate-y-0.5 hover:shadow-[3px_3px_0px_#17130E]">BEGIN CASE FILE <ShieldCheck className="h-4 w-4" /></Link></div>
            </Reveal>
        </div>
    );
}
