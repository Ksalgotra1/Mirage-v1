import React, { Suspense, lazy, useEffect, useRef, useState } from 'react';
import { BookOpen, Cpu, ShieldAlert } from 'lucide-react';
import Reveal from '@/components/Reveal.jsx';

const Layer1Sandbox = lazy(() => import('@/components/Layer1ScanAnimation.jsx'));
const Layer2Sandbox = lazy(() => import('@/components/Layer2ScanAnimation.jsx'));
const Layer3aSandbox = lazy(() => import('@/components/Layer3aScanAnimation.jsx'));
const Layer3bSandbox = lazy(() => import('@/components/Layer3bScanAnimation.jsx'));

const chapters = [
    { id: 'layer-01', number: '01', nav: 'Opcode', title: 'Static opcode inspection & zero-RCE allowlist', purpose: 'Layer 01 reads a checkpoint as a byte-level instruction stream, tracing pickle behavior offline before an artifact can execute in the host runtime.', output: 'Unsafe callable chains, modules, and reconstruction instructions are neutralized before deserialization.', spec: 'Zero-RCE allowlist unpickler enforces a strict whitelist of safe torch/collections classes, neutralizing dangerous or unlisted globals (os, subprocess, sys, posix, nt, pty, webbrowser, marshal, urllib, runpy, functools, operator) into safe no-ops before host evaluation.', flagged: 'This is an execution-path finding—not a tensor or steganography result.', Sandbox: Layer1Sandbox },
    { id: 'layer-02', number: '02', nav: 'Grammar', title: 'Statistical grammar analysis', purpose: 'Layer 02 models the structural grammar of a checkpoint to identify risky construction patterns when adversaries pad, encode, or rearrange an exploit chain.', output: 'A calibrated structural risk score identifies suspicious opcode grammar even when direct strings are hidden.', spec: 'A 43-dimensional feature vector combines opcode frequencies (26 unigrams), n-gram transitions (14 bigrams), and callable-risk ratios. A 100-tree XGBoost ensemble produces the calibrated outcome with 0.90 ROC-AUC.', flagged: 'It can coexist with a clean steganalysis path because the layers inspect different evidence.', Sandbox: Layer2Sandbox },
    { id: 'layer-03', number: '03', nav: 'Stego', title: 'Tensor steganalysis & multi-rate mantissa engine', purpose: 'Layer 03A maps float32 weights into Galois-field rasters and analyzes mantissa planes for hidden multi-rate steganography (x = 1 to 16 bits).', output: 'Clean, suspicious, or malicious byte-plane evidence decides whether forensic extraction and disassembly are required.', spec: 'Dual 2D/1D architecture. 2D OSLEmbeddingNet computes CL2 cosine margin across GF-Fourpart planes for 8-bit payloads. 1D auxiliary engine applies self-calibrated mantissa nibble TVD with Banker\'s rounding parity compensation (x=4 bits) and O(M log M) FFT periodic autocorrelation (x=1, 2 bits). Zero-LSB fast-path and FP16/BF16 upcast vetoes prevent false alarms.', flagged: 'Identifies structured shellcode, encrypted ciphers, and covert bitstreams across floating-point tensors.', Sandbox: Layer3aSandbox },
    { id: 'layer-04', number: '04', nav: 'Spectral', title: 'Spectral invariant analysis & geometric trojans', purpose: 'Layer 03B examines weight geometry directly for backdoors and poisoning behavior that may leave no pickle or hidden-byte signature.', output: 'Reference-free weight geometry exposes backdoor-like distortions without a clean baseline model.', spec: 'Pruning-safe active-neuron L2 norm filtering with p90 tail isolation, active-submatrix SVD singular value ratio collapse (dim >= 32), and spatial-kernel kurtosis gating (kH x kW > 1) ensure zero clean false positives across SE and pointwise networks.', flagged: 'This is independent from Layer 03A’s steganalysis branch and can drive a malicious fused verdict.', Sandbox: Layer3bSandbox },
    { id: 'layer-05', number: '05', nav: 'Disarm', title: 'Surgical disarmament & mathematical remediation', purpose: 'Remediates flagged checkpoints in-memory: strips malicious serialization callables, zeroes steganographic mantissa planes while rescaling tensor L2 norms, and deflates geometric trojans.', output: 'Cleaned, sanitized checkpoint with cryptographic audit trail (pre/post SHA-256) and guaranteed pass on automated 4-layer verification rescan.', spec: 'Phase 1: In-memory AST opcode rewrite to neutral lambdas with zero RCE. Phase 2: Bitwise LSB zeroing with exact norm preservation (W_clean = W_masked * ||W_0||_2 / ||W_masked||_2). Phase 3: SVD rank-1 deflation (W_hat = W - sigma_1 u_1 v_1^T). Phase 4: Automated full-pipeline rescan gate.', flagged: 'Releases sanitized checkpoint only if all 4 layers return BENIGN/CLEAN. Disarmament is aborted and quarantined if rescan verification fails.', Sandbox: null },
];

function Sandbox({ chapter, active }) {
    if (!chapter.Sandbox) return null;
    const [loaded, setLoaded] = useState(active);
    const targetRef = useRef(null);
    useEffect(() => { if (active) setLoaded(true); }, [active]);
    useEffect(() => {
        if (loaded || !targetRef.current) return undefined;
        const observer = new IntersectionObserver(([entry]) => {
            if (entry.isIntersecting) { setLoaded(true); observer.disconnect(); }
        }, { rootMargin: '240px 0px' });
        observer.observe(targetRef.current);
        return () => observer.disconnect();
    }, [loaded]);
    const Demo = chapter.Sandbox;
    return <div ref={targetRef} className="mt-7 border-2 border-ink bg-[#E8DFC6] p-4 sm:p-6">
        <div className="mb-4 flex items-center justify-between border-b-2 border-ink pb-3"><span className="font-mono text-[11px] font-bold uppercase tracking-[0.16em]">Interactive demonstration</span><span className="font-mono text-[10px] uppercase text-[#17130E]/60">On-demand module</span></div>
        {loaded ? <Suspense fallback={<div className="min-h-64 p-6 font-mono text-xs">Loading layer demonstration…</div>}><Demo /></Suspense> : <div className="flex min-h-64 items-center justify-center border border-dashed border-ink/50 p-6 text-center font-mono text-xs text-[#17130E]/65">The sandbox loads when this chapter enters the case file.</div>}
    </div>;
}

export default function DocsPage() {
    const [active, setActive] = useState('layer-01');
    useEffect(() => {
        const observer = new IntersectionObserver((entries) => {
            const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
            if (visible) setActive(visible.target.id);
        }, { rootMargin: '-22% 0px -60% 0px', threshold: [0.05, 0.25, 0.5] });
        chapters.forEach(({ id }) => { const element = document.getElementById(id); if (element) observer.observe(element); });
        return () => observer.disconnect();
    }, []);
    const scrollToChapter = (id) => { setActive(id); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); };
    return <div className="mx-auto max-w-7xl space-y-10">
        <Reveal className="border-2 border-ink bg-[#17130E] p-8 text-[#F2ECDC] shadow-[6px_6px_0px_#17130E] sm:p-12">
            <p className="inline-flex items-center gap-2 font-mono text-[11px] font-bold uppercase tracking-[0.2em] text-[#EFC23A]"><BookOpen className="h-4 w-4" /> Documentation / four-layer field manual</p>
            <h1 className="mt-6 max-w-4xl text-4xl font-black leading-[0.92] sm:text-6xl">Read the method. Test the decision boundary.</h1>
            <p className="mt-5 max-w-2xl text-sm leading-relaxed text-[#F2ECDC]/75 sm:text-base">Operating notes, decision boundaries, and on-demand demonstrations for each checkpoint layer.</p>
        </Reveal>
        <div className="grid gap-8 lg:grid-cols-[190px_minmax(0,1fr)]">
            <aside className="z-20 lg:sticky lg:top-28 lg:self-start"><nav className="flex gap-2 overflow-x-auto border-2 border-ink bg-[#F2ECDC] p-2 lg:flex-col" aria-label="Four layer chapters">
                {chapters.map((chapter) => <button key={chapter.id} type="button" onClick={() => scrollToChapter(chapter.id)} aria-current={active === chapter.id ? 'step' : undefined} className={`flex min-w-max items-center gap-3 border-2 px-3 py-3 text-left transition ${active === chapter.id ? 'border-[#17130E] bg-[#EFC23A] shadow-[2px_2px_0px_#17130E]' : 'border-transparent hover:border-ink/50 hover:bg-[#E8DFC6]'}`}><span className="font-serif text-xl font-black">{chapter.number}</span><span className="font-mono text-[10px] font-bold uppercase tracking-[0.14em]">{chapter.nav}</span></button>)}
            </nav></aside>
            <main className="space-y-10">
                {chapters.map((chapter) => <section key={chapter.id} id={chapter.id} aria-labelledby={`${chapter.id}-title`} className="scroll-mt-32 border-2 border-ink bg-[#F2ECDC] p-6 shadow-[5px_5px_0px_#17130E] sm:p-8">
                    <div className="flex flex-col gap-5 border-b-2 border-ink pb-6 sm:flex-row sm:items-start sm:justify-between"><div><p className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#2F4FE0]">Chapter {chapter.number}</p><h2 id={`${chapter.id}-title`} className="mt-2 text-3xl font-black sm:text-5xl">{chapter.title}</h2></div><span className="inline-flex w-fit items-center gap-2 border border-ink bg-[#EFC23A] px-3 py-2 font-mono text-[10px] font-bold uppercase tracking-[0.12em]"><Cpu className="h-3.5 w-3.5" /> Layer decision</span></div>
                    <div className="mt-6 grid gap-6 md:grid-cols-2"><div><p className="font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#17130E]/60">Purpose</p><p className="mt-2 text-sm leading-relaxed text-[#17130E]/85">{chapter.purpose}</p></div><div className="border-l-4 border-[#2F4FE0] bg-[#E8DFC6] p-4"><p className="font-mono text-[10px] font-bold uppercase tracking-[0.16em]">Decision output</p><p className="mt-2 text-sm leading-relaxed">{chapter.output}</p></div></div>
                    <div className="mt-6 grid gap-5 md:grid-cols-2"><div className="border-2 border-ink bg-[#17130E] p-5 text-[#F2ECDC]"><p className="font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#EFC23A]">Technical notes</p><p className="mt-3 text-sm leading-relaxed text-[#F2ECDC]/80">{chapter.spec}</p></div><div className="border-2 border-[#C4402E] bg-[#F7DFDA] p-5"><p className="flex items-center gap-2 font-mono text-[10px] font-bold uppercase tracking-[0.16em] text-[#C4402E]"><ShieldAlert className="h-4 w-4" /> What is flagged</p><p className="mt-3 text-sm leading-relaxed text-[#17130E]/85">{chapter.flagged}</p></div></div>
                    <Sandbox chapter={chapter} active={active === chapter.id} />
                </section>)}
            </main>
        </div>
    </div>;
}
