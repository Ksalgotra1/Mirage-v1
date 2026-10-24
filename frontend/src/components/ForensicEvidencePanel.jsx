import React from 'react';
import { Binary, FileSearch, Fingerprint, Gauge, Layers3, ShieldAlert } from 'lucide-react';

function Detail({ label, value }) {
    if (value === undefined || value === null || value === '') return null;
    return (
        <div className="flex items-start justify-between gap-4 border-b border-ink/15 py-1.5 last:border-0">
            <span className="shrink-0 text-[#17130E]/60">{label}</span>
            <span className="text-right font-bold text-[#17130E]">{String(value)}</span>
        </div>
    );
}

function EvidenceCard({ icon: Icon, title, children }) {
    return (
        <article className="border-2 border-ink bg-[#F2ECDC] p-4">
            <div className="mb-3 flex items-center gap-2 border-b border-ink/20 pb-2">
                <Icon className="h-4 w-4 text-[#C4402E]" />
                <h4 className="font-mono text-[11px] font-extrabold uppercase tracking-wider">{title}</h4>
            </div>
            <div className="font-mono text-[11px] leading-relaxed">{children}</div>
        </article>
    );
}

export default function ForensicEvidencePanel({ forensics }) {
    if (!forensics) return null;

    if (forensics.error) {
        return (
            <section className="border-2 border-[#C4402E] bg-[#F7DFDA] p-5 text-[#C4402E]">
                <div className="flex items-center gap-2 font-mono text-xs font-bold uppercase tracking-wider">
                    <ShieldAlert className="h-4 w-4" />
                    Forensic analysis unavailable
                </div>
                <p className="mt-2 font-mono text-xs">{forensics.error}</p>
            </section>
        );
    }

    const { bit_depth: bitDepth, distribution, entropy, capacity, magic_bytes: magicBytes } = forensics;

    return (
        <section className="border-2 border-ink bg-[#E8DFC6] p-6 sm:p-8 space-y-5">
            <div className="flex flex-col gap-3 border-b-2 border-ink pb-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                    <h3 className="font-mono text-sm font-extrabold uppercase tracking-[0.18em]">Live forensic evidence</h3>
                    <p className="mt-1 font-mono text-[11px] text-[#17130E]/65">
                        Backend-generated evidence for this scan · {forensics.analysis_depth || `${forensics.band || 'unknown'} analysis`}
                    </p>
                </div>
                <span className="w-fit border border-ink bg-[#17130E] px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-wider text-[#F2ECDC]">
                    {forensics.band || 'forensic'} band
                </span>
            </div>

            {forensics.note && <p className="font-mono text-xs text-[#17130E]/75">{forensics.note}</p>}

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                <EvidenceCard icon={Binary} title="Signal 1 · Bit-plane depth">
                    <Detail label="Estimated depth" value={bitDepth?.estimated_x && `${bitDepth.estimated_x} bits`} />
                    <Detail label="Affected planes" value={bitDepth?.anomalous_planes?.join(', ')} />
                    <Detail label="Assessment" value={bitDepth?.stealth_assessment} />
                    <Detail label="Maximum capacity" value={bitDepth?.max_theoretical_capacity_bytes && `${bitDepth.max_theoretical_capacity_bytes} bytes`} />
                    {bitDepth?.description && <p className="mt-2 text-[#17130E]/75">{bitDepth.description}</p>}
                </EvidenceCard>

                <EvidenceCard icon={Gauge} title="Signal 3 · Payload entropy">
                    <Detail label="Entropy" value={entropy?.bits_per_byte && `${entropy.bits_per_byte} bits/byte`} />
                    <Detail label="Classification" value={entropy?.classification} />
                    {entropy?.assessment && <p className="mt-2 text-[#17130E]/75">{entropy.assessment}</p>}
                </EvidenceCard>

                {distribution && (
                    <EvidenceCard icon={Layers3} title="Signal 2 · Distribution">
                        <Detail label="Pattern" value={distribution.pattern} />
                        <Detail label="Affected layers" value={distribution.affected_layer_count} />
                        <Detail label="Concentration" value={distribution.concentration_ratio} />
                        {distribution.description && <p className="mt-2 text-[#17130E]/75">{distribution.description}</p>}
                    </EvidenceCard>
                )}

                {capacity && (
                    <EvidenceCard icon={FileSearch} title="Signal 4 · Capacity">
                        <Detail label="Estimated payload" value={capacity.human_readable} />
                        <Detail label="Classification" value={capacity.classification} />
                        <Detail label="Affected weights" value={capacity.n_anomalous_weights} />
                        {capacity.description && <p className="mt-2 text-[#17130E]/75">{capacity.description}</p>}
                    </EvidenceCard>
                )}

                {magicBytes && (
                    <EvidenceCard icon={Fingerprint} title="Signal 5 · Magic bytes">
                        <Detail label="Detected" value={magicBytes.detected ? 'Yes' : 'No'} />
                        <Detail label="Signature" value={magicBytes.signature} />
                        <Detail label="Header" value={magicBytes.raw_header_hex} />
                        {magicBytes.description && <p className="mt-2 text-[#17130E]/75">{magicBytes.description}</p>}
                    </EvidenceCard>
                )}
            </div>
        </section>
    );
}
