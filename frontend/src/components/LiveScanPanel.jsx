import React, { useEffect, useMemo, useRef } from 'react';
import { Activity, Radar, Terminal, XCircle } from 'lucide-react';

const NODES = [
    ['upload', 92, 270, 'MODEL INGEST', 'upload sealed'], ['layer1', 258, 132, 'LAYER 01', 'opcode paths'], ['layer2', 420, 316, 'LAYER 02', 'grammar profile'], ['layer3a', 580, 204, 'LAYER 03A', 'tensor byte planes'], ['forensics', 808, 318, 'BYTE FORENSICS', 'conditional analysis'], ['layer3b', 930, 218, 'LAYER 03B', 'weight geometry'], ['fusion', 1030, 100, 'FUSION', 'signal correlation'], ['summary', 1115, 258, 'DOSSIER', 'risk synthesis'], ['output', 1170, 100, 'OUTPUT', 'verdict ready'],
];
const BANDS = [['clean', 705, 84, 'CLEAN', 'forensics skipped'], ['suspicious', 705, 204, 'SUSPICIOUS', 'full byte analysis'], ['malicious', 705, 364, 'MALICIOUS', 'full byte analysis']];
const PATHS = {
    l1: 'M92 270 C145 270 182 132 258 132', l2: 'M258 132 C312 132 350 316 420 316', l3a: 'M420 316 C472 316 514 204 580 204', cleanIn: 'M580 204 C624 204 654 84 705 84', cleanOut: 'M705 84 C800 84 832 218 930 218', suspiciousIn: 'M580 204 C624 204 654 204 705 204', suspiciousForensics: 'M705 204 C752 204 760 318 808 318', maliciousIn: 'M580 204 C628 204 656 364 705 364', maliciousForensics: 'M705 364 C752 364 760 318 808 318', forensicOut: 'M808 318 C852 318 875 218 930 218', fusion: 'M930 218 C970 218 990 100 1030 100', summary: 'M1030 100 C1060 100 1082 258 1115 258', output: 'M1115 258 C1140 258 1140 100 1170 100',
};

function latest(events, stage) { return [...events].reverse().find((event) => event.stage === stage); }
function state(events, stage) { const event = latest(events, stage); return !event ? 'queued' : event.status === 'started' ? 'active' : event.status === 'skipped' ? 'skipped' : event.status === 'failed' ? 'failed' : 'complete'; }
function outcome(events, stage) { return latest(events, stage)?.metadata?.outcome || 'clear'; }
function colorFor(outcomeValue, active = false) { if (active) return '#EFC23A'; if (outcomeValue === 'flagged' || outcomeValue === 'failed') return '#C4402E'; if (outcomeValue === 'elevated') return '#EFC23A'; return '#2F4FE0'; }

function Node({ item, nodeState, nodeOutcome, bypassed = false }) {
    const [, x, y, title, subtitle] = item;
    const active = nodeState === 'active';
    const visited = nodeState === 'complete';
    const fill = visited || active ? colorFor(nodeOutcome, active) : '#F2ECDC';
    const stroke = visited || active ? fill : '#BFB8A9';
    const glyph = bypassed ? <path d="M-9 0 H9" stroke="#746E64" strokeWidth="3" strokeLinecap="round" /> : visited ? <path d="M-8 0 L-2 7 L11 -9" fill="none" stroke={nodeOutcome === 'elevated' ? '#17130E' : '#fff'} strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" /> : active ? <circle r="7" fill="none" stroke="#17130E" strokeWidth="3" /> : <circle r="4" fill="#A39B8D" />;
    return <g transform={`translate(${x} ${y})`}>
        {active && <circle r="35" fill="none" stroke="#EFC23A" strokeOpacity=".32" strokeWidth="10" className="animate-pulse" />}
        <circle r="23" fill={fill} stroke={stroke} strokeWidth="3" />{glyph}
        <text y="43" textAnchor="middle" fill="#17130E" fontFamily="JetBrains Mono, monospace" fontSize="11" fontWeight="800">{title}</text><text y="59" textAnchor="middle" fill="#746E64" fontFamily="JetBrains Mono, monospace" fontSize="9">{subtitle}</text>
    </g>;
}

function Edge({ d, visible, tone = 'clear', active = false }) {
    if (!visible) return null;
    const stroke = colorFor(tone, active);
    return <path d={d} fill="none" stroke={stroke} strokeOpacity=".82" strokeWidth="5" strokeLinecap="round" strokeDasharray={active ? '12 10' : undefined}>{active && <animate attributeName="stroke-dashoffset" from="22" to="0" dur=".55s" repeatCount="indefinite" />}</path>;
}

export default function LiveScanPanel({ events, scanning, connectionError }) {
    const sectionRef = useRef(null); const logRef = useRef(null); const startedRef = useRef(false);
    const l3a = useMemo(() => latest(events, 'layer3a'), [events]);
    const band = l3a?.metadata?.forensic_band;
    const forensicState = state(events, 'forensics');
    const stage = (key) => state(events, key); const tone = (key) => outcome(events, key);
    useEffect(() => { if (scanning && !startedRef.current) sectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' }); startedRef.current = scanning; }, [scanning]);
    useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [events]);
    const hasStarted = (key) => stage(key) !== 'queued';
    const pathTone = (key) => tone(key);
    const activeTo = (key) => stage(key) === 'active';
    const branchVisible = Boolean(band);
    return <section ref={sectionRef} className="border-2 border-ink bg-[#F2ECDC] p-4 shadow-[6px_6px_0px_#17130E] sm:p-8" aria-live="polite">
        <div className="flex flex-col gap-3 border-b-2 border-ink pb-4 sm:flex-row sm:items-center sm:justify-between"><div className="flex items-center gap-3"><Radar className="h-5 w-5 text-[#2F4FE0]" /><div><h2 className="font-mono text-sm font-extrabold uppercase tracking-[.18em]">Live inspection map</h2><p className="font-mono text-[10px] text-[#17130E]/60">BACKEND-PACED · REAL EXECUTION PATH</p></div></div><span className={`inline-flex items-center gap-2 font-mono text-[10px] font-bold uppercase ${connectionError ? 'text-[#C4402E]' : 'text-[#2F4FE0]'}`}>{connectionError ? <XCircle className="h-3.5 w-3.5" /> : <Activity className={`h-3.5 w-3.5 ${scanning ? 'animate-pulse' : ''}`} />}{connectionError || (scanning ? 'Telemetry live' : 'Trace complete')}</span></div>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[9px] font-bold uppercase text-[#17130E]/70"><span className="text-[#2F4FE0]">● completed route</span><span className="text-[#EFC23A]">● active / elevated</span><span className="text-[#C4402E]">● confirmed finding</span><span className="text-[#746E64]">● unvisited / bypassed</span></div>
        <div className="mt-4 overflow-x-auto border-2 border-ink bg-[#E8DFC6] p-2 sm:p-5"><div className="min-w-[1200px]"><svg viewBox="0 0 1200 470" className="h-auto w-full" role="img" aria-label="Live model security scan graph"><defs><pattern id="scan-grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0 L0 0 0 24" fill="none" stroke="#17130E" strokeOpacity=".07" /></pattern></defs><rect width="1200" height="470" fill="url(#scan-grid)" /><text x="32" y="38" fill="#17130E" fontFamily="JetBrains Mono, monospace" fontSize="10" fontWeight="800" letterSpacing="1.5">EXECUTION GRAPH / CONDITIONAL FORENSICS ROUTING</text><text x="32" y="55" fill="#746E64" fontFamily="JetBrains Mono, monospace" fontSize="9">Ticks are emitted only for stages the backend has completed.</text>
            <g fill="none" stroke="#17130E" strokeOpacity=".22" strokeWidth="5" strokeLinecap="round">{Object.values(PATHS).map((d) => <path key={d} d={d} />)}</g>
            <Edge d={PATHS.l1} visible={hasStarted('layer1')} tone={pathTone('upload')} active={activeTo('layer1')} /><Edge d={PATHS.l2} visible={hasStarted('layer2')} tone={pathTone('layer1')} active={activeTo('layer2')} /><Edge d={PATHS.l3a} visible={hasStarted('layer3a')} tone={pathTone('layer2')} active={activeTo('layer3a')} />
            <Edge d={PATHS.cleanIn} visible={band === 'clean'} tone="clear" active={false} /><Edge d={PATHS.cleanOut} visible={band === 'clean' && hasStarted('layer3b')} tone="clear" active={activeTo('layer3b')} />
            <Edge d={PATHS.suspiciousIn} visible={band === 'suspicious'} tone="elevated" active={forensicState === 'active'} /><Edge d={PATHS.suspiciousForensics} visible={band === 'suspicious' && forensicState !== 'queued'} tone="elevated" active={forensicState === 'active'} />
            <Edge d={PATHS.maliciousIn} visible={band === 'malicious'} tone="flagged" active={forensicState === 'active'} /><Edge d={PATHS.maliciousForensics} visible={band === 'malicious' && forensicState !== 'queued'} tone="flagged" active={forensicState === 'active'} /><Edge d={PATHS.forensicOut} visible={band !== 'clean' && hasStarted('layer3b')} tone={tone('forensics')} active={activeTo('layer3b')} />
            <Edge d={PATHS.fusion} visible={hasStarted('fusion')} tone={pathTone('layer3b')} active={activeTo('fusion')} /><Edge d={PATHS.summary} visible={hasStarted('summary')} tone={pathTone('fusion')} active={activeTo('summary')} /><Edge d={PATHS.output} visible={hasStarted('output')} tone={pathTone('summary')} active={activeTo('output')} />
            {NODES.map((item) => <Node key={item[0]} item={item} nodeState={stage(item[0])} nodeOutcome={tone(item[0])} bypassed={item[0] === 'forensics' && forensicState === 'skipped'} />)}
            {BANDS.map((item) => <Node key={item[0]} item={item} nodeState={band === item[0] ? 'complete' : 'queued'} nodeOutcome={band === item[0] ? item[0] === 'malicious' ? 'flagged' : item[0] === 'suspicious' ? 'elevated' : 'clear' : 'clear'} />)}
            {branchVisible && <text x="640" y="440" fill={band === 'malicious' ? '#C4402E' : band === 'suspicious' ? '#A17200' : '#2F4FE0'} fontFamily="JetBrains Mono, monospace" fontSize="11" fontWeight="800" textAnchor="middle">L3A ROUTED TO {band.toUpperCase()} BAND</text>}
        </svg></div></div>
        <div className="mt-5 border-2 border-[#17130E] bg-[#17130E]"><div className="flex items-center gap-2 border-b border-[#F2ECDC]/20 px-4 py-2 text-[#EFC23A]"><Terminal className="h-3.5 w-3.5" /><span className="font-mono text-[10px] font-bold uppercase tracking-wider">Live security log</span></div><div ref={logRef} className="max-h-48 overflow-y-auto p-4 font-mono text-[11px] leading-relaxed text-[#F2ECDC]/90">{events.length === 0 ? <div className="text-[#F2ECDC]/55">Establishing scan telemetry…</div> : events.map((event) => <div key={event.id} className={event.metadata?.outcome === 'flagged' || event.status === 'failed' ? 'text-[#FF9384]' : ''}><span className="mr-2 text-[#EFC23A]">[{event.stage.toUpperCase()}]</span>{event.message}</div>)}</div></div>
    </section>;
}
