import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { ShieldAlert, ShieldCheck, AlertTriangle, Play, RotateCcw, Lock, Terminal } from "lucide-react";

/**
 * LAYER 3a — 512×512 GALOIS FIELD STEGANALYSIS & 5-SIGNAL FORENSIC ENGINE
 * -----------------------------------------------------------------------
 * Same "Field Notes" case-file system as Layers 1 & 2: cream paper, warm
 * near-black ink, three flat accent colors, Fraunces for voice,
 * JetBrains Mono for every technical surface, the rubber-stamp verdict
 * as the one bold move. Three.js is scoped to the same two jobs: the
 * faint paper-grain field and the 3D stamp-slam.
 *
 * Surfaces specific to Layer 3a:
 *  - 512×512 Galois Field quadrant rasterization (Q0 MSB -> Q3 LSB across 4 256×256 planes)
 *  - Contrastive OSL embedding score against the calibrated margin τ = 0.072236
 *  - 3-zone forensic barometer (CLEAN < 0.40 <= SUSPICIOUS < 0.70 <= MALICIOUS)
 *  - 5-signal forensic breakdown, gated strictly by band (Signals 1 & 3 at SUSPICIOUS; 1–5 at MALICIOUS)
 *  - Live Gemini threat dossier terminal
 */

const TAU = 0.072236;

// ---------------------------------------------------------------------
// Scenario data — mirrors the three test cases in the Layer 3a spec.
// ---------------------------------------------------------------------
const SCENARIOS = {
    malicious: {
        id: "malicious",
        file: "resnet50_v3_prod.pt",
        band: "MALICIOUS",
        bandTone: "malicious",
        bandColor: "#C4402E",
        stampLabel: "MALICIOUS",
        score: 0.884,
        delta: 0.2854,
        latency: 41.2,
        quadrantNoise: [0.3, 0.42, 0.68, 0.93],
        signals: [
            {
                key: "s1",
                title: "SIGNAL 1 — BIT-PLANE DEPTH",
                status: "run",
                value: "x = 8 bits / float",
                detail: "Balanced embedding across Q1–Q3. Does not spill into Q0 (MSB, ~3.50 baseline entropy).",
            },
            {
                key: "s2",
                title: "SIGNAL 2 — DISTRIBUTION GEOMETRY",
                status: "run",
                value: "2 hotspot layers",
                detail: "layer3.2.conv2, layer4.1.conv1. BatchNorm and first-conv statistics left clean — inference accuracy preserved.",
            },
            {
                key: "s3",
                title: "SIGNAL 3 — SHANNON PAYLOAD ENTROPY",
                status: "run",
                value: "7.2 bits / byte",
                detail: "Falls in the compressed range (6.8–7.4). Consistent with a Gzip/Zip-packed payload.",
            },
            {
                key: "s4",
                title: "SIGNAL 4 — PAYLOAD CAPACITY",
                status: "run",
                value: "180.0 KB",
                detail: "184,320 anomalous weights. Above the 50 KB line — classified Embedded Executable.",
            },
            {
                key: "s5",
                title: "SIGNAL 5 — MAGIC BYTE INSPECTION",
                status: "run",
                value: "4D 5A 90 00 ...",
                detail: "0x4D 0x5A → Windows Portable Executable (PE / DLL / EXE).",
            },
        ],
        gemini: [
            { h: "THREAT ASSESSMENT", b: "A compressed Windows PE payload is embedded in the LSB stream. Severity: critical." },
            { h: "ATTACKER PROFILE", b: "High sophistication. BatchNorm and first-conv statistics were deliberately preserved to evade accuracy-drift detectors." },
            { h: "PAYLOAD INTENT", b: "Capacity and format point to a dropped executable rather than a beacon — likely lateral-movement tooling." },
            { h: "RECOMMENDED ACTIONS", b: "Quarantine the artifact, isolate the serving host, and hash every checkpoint in the same release." },
            { h: "IF ALREADY DEPLOYED", b: "Contain network egress from the inference host and capture a memory dump before any restart." },
        ],
    },
    suspicious: {
        id: "suspicious",
        file: "resnet152_deep_v2.pt",
        band: "SUSPICIOUS",
        bandTone: "suspicious",
        bandColor: "#EFC23A",
        stampLabel: "SUSPICIOUS",
        score: 0.538,
        delta: 0.1108,
        latency: 9.4,
        quadrantNoise: [0.08, 0.12, 0.22, 0.61],
        signals: [
            {
                key: "s1",
                title: "SIGNAL 1 — BIT-PLANE DEPTH",
                status: "run",
                value: "x = 2 bits / float",
                detail: "Marginal. Sits in the high-stealth range, only just above the clean baseline.",
            },
            {
                key: "s2",
                title: "SIGNAL 2 — DISTRIBUTION GEOMETRY",
                status: "gated",
                value: "GATED",
                detail: "Only runs at the MALICIOUS band (score ≥ 0.70).",
            },
            {
                key: "s3",
                title: "SIGNAL 3 — SHANNON PAYLOAD ENTROPY",
                status: "run",
                value: "7.98 bits / byte",
                detail: "High entropy alone can't separate ciphertext from ordinary noise. Inconclusive on its own.",
            },
            {
                key: "s4",
                title: "SIGNAL 4 — PAYLOAD CAPACITY",
                status: "gated",
                value: "GATED",
                detail: "Only runs at the MALICIOUS band (score ≥ 0.70).",
            },
            {
                key: "s5",
                title: "SIGNAL 5 — MAGIC BYTE INSPECTION",
                status: "gated",
                value: "GATED",
                detail: "Only runs at the MALICIOUS band (score ≥ 0.70).",
            },
        ],
        gemini: [
            { h: "ELEVATED SIGNALS", b: "Margin above threshold is small (Δ = +0.1108) and entropy alone can't separate encryption from noise." },
            { h: "RECOMMENDED HUMAN CHECKS", b: "Diff this checkpoint against its last known-clean version and re-run bit-plane depth on isolated layers." },
        ],
    },
    clean: {
        id: "clean",
        file: "mobilenet_v3_edge.pt",
        band: "CLEAN",
        bandTone: "clean",
        bandColor: "#2F4FE0",
        stampLabel: "CLEARED",
        score: 0.342,
        delta: -0.0017,
        latency: 1.8,
        quadrantNoise: [0.05, 0.06, 0.07, 0.09],
        signals: [],
        gemini: [],
    },
};

const QUAD_META = [
    { key: "q0", label: "Q0 · MSB", hint: "512×512 (~3.50 base)" },
    { key: "q1", label: "Q1 · UPPER", hint: "512×512" },
    { key: "q2", label: "Q2 · MID", hint: "512×512" },
    { key: "q3", label: "Q3 · LSB", hint: "512×512 (~7.996 base)" },
];

// ---------------------------------------------------------------------
// Three.js layer: paper grain + rubber-stamp slam. Identical mechanism
// to Layer 1 — generic label/color in, animated slam out.
// ---------------------------------------------------------------------
function useStampScene(mountRef) {
    const stateRef = useRef(null);

    useEffect(() => {
        const mount = mountRef.current;
        if (!mount) return undefined;

        const reduceMotion =
            typeof window !== "undefined" &&
            window.matchMedia &&
            window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        const width = mount.clientWidth;
        const height = mount.clientHeight;

        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(width, height);
        renderer.domElement.style.position = "absolute";
        renderer.domElement.style.inset = "0";
        renderer.domElement.style.pointerEvents = "none";
        mount.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 100);
        camera.position.set(0, 0, 10);
        const camBase = { x: 0, y: 0 };

        // --- paper grain field -------------------------------------------------
        const grainGeo = new THREE.PlaneGeometry(20, 20);
        const grainMat = new THREE.ShaderMaterial({
            transparent: true,
            uniforms: { uTime: { value: 0 }, uOpacity: { value: 0.05 } },
            vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
            fragmentShader: `
        varying vec2 vUv;
        uniform float uTime;
        uniform float uOpacity;
        float rand(vec2 co) {
          return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
        }
        void main() {
          float n = rand(vUv * 800.0 + floor(uTime * 8.0));
          gl_FragColor = vec4(0.09, 0.08, 0.06, n * uOpacity);
        }
      `,
        });
        const grain = new THREE.Mesh(grainGeo, grainMat);
        grain.position.z = -1;
        scene.add(grain);

        // --- stamp (canvas texture, redrawn per verdict) ------------------------
        function makeStampTexture(label, colorHex) {
            const c = document.createElement("canvas");
            c.width = 640;
            c.height = 300;
            const ctx = c.getContext("2d");
            ctx.clearRect(0, 0, c.width, c.height);
            ctx.translate(c.width / 2, c.height / 2);

            ctx.strokeStyle = colorHex;
            ctx.lineWidth = 10;
            for (let i = 0; i < 2; i++) {
                ctx.beginPath();
                const r = 130 - i * 16;
                for (let a = 0; a <= 360; a += 6) {
                    const jitter = (Math.random() - 0.5) * 5;
                    const rad = (a * Math.PI) / 180;
                    const rr = r + jitter;
                    const x = Math.cos(rad) * rr * 2.15;
                    const y = Math.sin(rad) * rr;
                    if (a === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }

            ctx.fillStyle = colorHex;
            const size = label.length > 10 ? 60 : 88;
            ctx.font = `800 ${size}px 'JetBrains Mono', 'Courier New', monospace`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.globalAlpha = 0.92;
            ctx.fillText(label, 0, 6);
            ctx.globalAlpha = 1;

            const tex = new THREE.CanvasTexture(c);
            tex.needsUpdate = true;
            return tex;
        }

        const stampGeo = new THREE.PlaneGeometry(6, 2.8);
        const stampMat = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0,
            depthWrite: false,
        });
        const stamp = new THREE.Mesh(stampGeo, stampMat);
        stamp.rotation.z = -0.14;
        stamp.position.set(0, 3, 2);
        scene.add(stamp);

        function makeShadowTexture() {
            const c = document.createElement("canvas");
            c.width = 512;
            c.height = 256;
            const ctx = c.getContext("2d");
            const grad = ctx.createRadialGradient(256, 128, 0, 256, 128, 250);
            grad.addColorStop(0, "rgba(23,19,14,0.55)");
            grad.addColorStop(0.6, "rgba(23,19,14,0.25)");
            grad.addColorStop(1, "rgba(23,19,14,0)");
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, 512, 256);
            return new THREE.CanvasTexture(c);
        }
        const shadowGeo = new THREE.PlaneGeometry(6.4, 3.2);
        const shadowMat = new THREE.MeshBasicMaterial({
            map: makeShadowTexture(),
            transparent: true,
            opacity: 0,
            depthWrite: false,
        });
        const shadow = new THREE.Mesh(shadowGeo, shadowMat);
        shadow.position.set(0.15, -0.2, 1.5);
        shadow.rotation.z = -0.14;
        scene.add(shadow);

        let raf = null;
        const clock = new THREE.Clock();
        let anim = null;
        let shake = null;

        function onResize() {
            const w = mount.clientWidth;
            const h = mount.clientHeight;
            if (!w || !h) return;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        }
        const ro = new ResizeObserver(onResize);
        ro.observe(mount);

        function tick() {
            raf = requestAnimationFrame(tick);
            const t = clock.getElapsedTime();
            grainMat.uniforms.uTime.value = t;

            if (anim) {
                const p = Math.min(1, (performance.now() - anim.start) / anim.duration);
                const ease = 1 - Math.pow(1 - p, 3);
                const overshoot = p < 1 ? Math.sin(p * Math.PI) * 0.06 : 0;
                stamp.position.y = anim.fromY + (anim.toY - anim.fromY) * ease;
                stamp.position.z = anim.fromZ + (anim.toZ - anim.fromZ) * ease;
                const s = anim.fromScale + (anim.toScale - anim.fromScale) * ease + overshoot;
                stamp.scale.set(s, s, 1);
                stampMat.opacity = ease;
                shadow.material.opacity = ease * 0.45;
                shadow.scale.set(0.7 + ease * 0.3, 0.7 + ease * 0.3, 1);
                if (p >= 1) anim = null;
            }

            if (shake && !reduceMotion) {
                const p = (performance.now() - shake.start) / shake.duration;
                if (p < 1) {
                    const decay = 1 - p;
                    camera.position.x = camBase.x + (Math.random() - 0.5) * 0.08 * decay;
                    camera.position.y = camBase.y + (Math.random() - 0.5) * 0.08 * decay;
                } else {
                    camera.position.x = camBase.x;
                    camera.position.y = camBase.y;
                    shake = null;
                }
            }

            renderer.render(scene, camera);
        }
        tick();

        stateRef.current = {
            fire(label, colorHex) {
                stamp.material.map = makeStampTexture(label, colorHex);
                stamp.material.needsUpdate = true;
                stamp.position.set(0, 3, 2);
                anim = {
                    start: performance.now(),
                    duration: reduceMotion ? 1 : 620,
                    fromY: 3,
                    toY: 0.05,
                    fromZ: 2,
                    toZ: 0,
                    fromScale: 1.5,
                    toScale: 1,
                };
                if (!reduceMotion) shake = { start: performance.now() + 480, duration: 260 };
            },
            reset() {
                stampMat.opacity = 0;
                shadow.material.opacity = 0;
                stamp.position.set(0, 3, 2);
                anim = null;
                shake = null;
            },
        };

        return () => {
            cancelAnimationFrame(raf);
            ro.disconnect();
            mount.removeChild(renderer.domElement);
            grainGeo.dispose();
            grainMat.dispose();
            stampGeo.dispose();
            stampMat.dispose();
            shadowGeo.dispose();
            shadowMat.dispose();
            renderer.dispose();
        };
    }, [mountRef]);

    return stateRef;
}

// ---------------------------------------------------------------------
// Small presentational bits
// ---------------------------------------------------------------------
function Eyebrow({ children, tone = "yellow" }) {
    return <span className={`l3a-pill l3a-pill--${tone}`}>{children}</span>;
}

function QuadrantCanvas({ density, label, hint, runId }) {
    const ref = useRef(null);

    useEffect(() => {
        const canvas = ref.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        ctx.fillStyle = "#E8DFC6";
        ctx.fillRect(0, 0, w, h);
        const dots = Math.floor(density * 900);
        for (let i = 0; i < dots; i++) {
            const x = Math.random() * w;
            const y = Math.random() * h;
            const a = 0.15 + Math.random() * 0.6;
            ctx.fillStyle = `rgba(23,19,14,${a})`;
            ctx.fillRect(x, y, 1.6, 1.6);
        }
    }, [density, runId]);

    const anomalous = density > 0.5;

    return (
        <div className="l3a-quad">
            <canvas
                key={runId}
                ref={ref}
                width={96}
                height={96}
                className={anomalous ? "is-anomalous" : ""}
            />
            <div className="l3a-quad-label">{label}</div>
            <div className="l3a-quad-hint">
                {hint || (anomalous ? "spillover" : "\u00A0")}
            </div>
        </div>
    );
}

function SignalCard({ signal, active }) {
    const gated = signal.status === "gated";
    return (
        <div
            className={`l3a-signal ${gated ? "is-gated" : "is-run"} ${active ? "is-active" : ""}`}
        >
            <div className="l3a-signal-head">
                {gated ? <Lock size={12} /> : null}
                <span className="l3a-signal-title">{signal.title}</span>
            </div>
            <div className="l3a-signal-value">{signal.value}</div>
            <div className="l3a-signal-detail">{signal.detail}</div>
        </div>
    );
}

export default function Layer3aScanAnimation() {
    const mountRef = useRef(null);
    const stamp = useStampScene(mountRef);

    const [scenario, setScenario] = useState(null); // 'malicious' | 'suspicious' | 'clean' | null
    const [runId, setRunId] = useState(0);
    const [scoreVal, setScoreVal] = useState(0);
    const [bandRevealed, setBandRevealed] = useState(false);
    const [signalsRevealed, setSignalsRevealed] = useState([]);
    const [geminiRevealed, setGeminiRevealed] = useState([]);
    const [verdict, setVerdict] = useState(null);
    const [latency, setLatency] = useState(0);
    const [running, setRunning] = useState(false);

    const timers = useRef([]);
    const clearTimers = useCallback(() => {
        timers.current.forEach(clearTimeout);
        timers.current = [];
    }, []);

    const runLatencyCount = useCallback((target) => {
        const start = performance.now();
        const dur = 500;
        function step() {
            const p = Math.min(1, (performance.now() - start) / dur);
            setLatency(Number((target * p).toFixed(1)));
            if (p < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }, []);

    const run = useCallback(
        (which) => {
            clearTimers();
            stamp.current && stamp.current.reset();
            const s = SCENARIOS[which];

            setScenario(which);
            setRunId((n) => n + 1);
            setScoreVal(0);
            setBandRevealed(false);
            setSignalsRevealed([]);
            setGeminiRevealed([]);
            setVerdict(null);
            setLatency(0);
            setRunning(true);

            const scoreStart = 250;
            const scoreDur = 700;

            const scoreTimer = setTimeout(() => {
                const t0 = performance.now();
                function step() {
                    const p = Math.min(1, (performance.now() - t0) / scoreDur);
                    setScoreVal(Number((s.score * p).toFixed(3)));
                    if (p < 1) requestAnimationFrame(step);
                    else setBandRevealed(true);
                }
                requestAnimationFrame(step);
            }, scoreStart);
            timers.current.push(scoreTimer);

            let cursor = scoreStart + scoreDur + 200;

            if (s.signals.length > 0) {
                s.signals.forEach((sig, i) => {
                    const t = setTimeout(() => {
                        setSignalsRevealed((prev) => [...prev, sig]);
                    }, cursor + i * 220);
                    timers.current.push(t);
                });
                cursor = cursor + s.signals.length * 220 + 200;
            }

            if (s.gemini.length > 0) {
                s.gemini.forEach((sec, i) => {
                    const t = setTimeout(() => {
                        setGeminiRevealed((prev) => [...prev, sec]);
                    }, cursor + i * 340);
                    timers.current.push(t);
                });
                cursor = cursor + s.gemini.length * 340 + 220;
            } else {
                cursor += 300;
            }

            const finalTimer = setTimeout(() => {
                setVerdict(s.band);
                stamp.current && stamp.current.fire(s.stampLabel, s.bandColor);
                runLatencyCount(s.latency);
                setRunning(false);
            }, cursor);
            timers.current.push(finalTimer);
        },
        [clearTimers, runLatencyCount, stamp]
    );

    const reset = useCallback(() => {
        clearTimers();
        stamp.current && stamp.current.reset();
        setScenario(null);
        setScoreVal(0);
        setBandRevealed(false);
        setSignalsRevealed([]);
        setGeminiRevealed([]);
        setVerdict(null);
        setLatency(0);
        setRunning(false);
    }, [clearTimers, stamp]);

    useEffect(() => () => clearTimers(), [clearTimers]);

    const s = scenario ? SCENARIOS[scenario] : null;
    const markerPct = Math.max(0, Math.min(100, scoreVal * 100));

    return (
        <div className="l3a-wrap">
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,900;1,9..144,500&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

        .l3a-wrap {
          --paper: #F2ECDC;
          --paper-2: #E8DFC6;
          --ink: #17130E;
          --ink-70: rgba(23,19,14,0.72);
          --ink-45: rgba(23,19,14,0.45);
          --line: rgba(23,19,14,0.16);
          --red: #C4402E;
          --blue: #2F4FE0;
          --yellow: #EFC23A;
          --amber: #C2790E;
          font-family: 'JetBrains Mono', monospace;
          color: var(--ink);
          background: var(--paper);
          border: 2px solid var(--ink);
          max-width: 920px;
          margin: 0 auto;
          position: relative;
          overflow: hidden;
        }
        .l3a-wrap * { box-sizing: border-box; }

        .l3a-ticker {
          background: var(--ink);
          color: var(--paper);
          font-size: 11px;
          letter-spacing: 0.14em;
          padding: 7px 0;
          overflow: hidden;
          white-space: nowrap;
          border-bottom: 2px solid var(--ink);
        }
        .l3a-ticker-track {
          display: inline-block;
          padding-left: 100%;
          animation: l3a-scroll 20s linear infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .l3a-ticker-track { animation: none; padding-left: 2%; }
        }
        @keyframes l3a-scroll {
          from { transform: translateX(0); }
          to { transform: translateX(-100%); }
        }

        .l3a-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          padding: 22px 24px 16px;
          border-bottom: 2px solid var(--ink);
        }
        .l3a-title-serif {
          font-family: 'Fraunces', serif;
          font-weight: 900;
          font-size: 28px;
          line-height: 1.08;
          margin: 10px 0 4px;
          letter-spacing: -0.01em;
        }
        .l3a-title-serif em { font-style: italic; font-weight: 500; }
        .l3a-sub {
          font-size: 12px;
          color: var(--ink-70);
          max-width: 50ch;
          line-height: 1.5;
        }
        .l3a-meta {
          text-align: right;
          font-size: 11px;
          color: var(--ink-45);
          letter-spacing: 0.06em;
          white-space: nowrap;
        }

        .l3a-pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 10.5px;
          font-weight: 700;
          letter-spacing: 0.1em;
          padding: 5px 12px;
          border-radius: 999px;
          border: 1.5px solid var(--ink);
          transform: rotate(-1.2deg);
        }
        .l3a-pill--yellow { background: var(--yellow); }
        .l3a-pill--blue { background: var(--blue); color: var(--paper); }
        .l3a-pill--red { background: var(--red); color: var(--paper); }
        .l3a-pill--paper { background: var(--paper); }
        .l3a-pill--clean { background: var(--blue); color: var(--paper); }
        .l3a-pill--suspicious { background: var(--yellow); color: var(--ink); }
        .l3a-pill--malicious { background: var(--red); color: var(--paper); }

        .l3a-stage {
          position: relative;
          display: grid;
          grid-template-columns: 1fr 1.2fr;
          gap: 0;
          min-height: 300px;
        }
        @media (max-width: 640px) {
          .l3a-stage { grid-template-columns: 1fr; }
        }

        .l3a-three-mount {
          position: absolute;
          inset: 0;
          z-index: 5;
          pointer-events: none;
        }

        .l3a-panel { padding: 18px 22px 22px; position: relative; z-index: 2; }
        .l3a-panel + .l3a-panel { border-left: 2px solid var(--ink); }
        @media (max-width: 640px) {
          .l3a-panel { min-height: 260px; }
          .l3a-panel + .l3a-panel { border-left: none; border-top: 2px solid var(--ink); }
        }

        .l3a-panel-label {
          font-size: 10.5px;
          font-weight: 700;
          letter-spacing: 0.14em;
          color: var(--ink-45);
          margin-bottom: 12px;
        }
        .l3a-panel-label + .l3a-panel-label { margin-top: 20px; }

        .l3a-placeholder {
          font-size: 11.5px;
          color: var(--ink-45);
          line-height: 1.5;
          max-width: 34ch;
        }

        .l3a-quad-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
          margin-bottom: 4px;
        }
        .l3a-quad { display: flex; flex-direction: column; align-items: center; gap: 4px; }
        .l3a-quad canvas {
          width: 100%;
          max-width: 96px;
          aspect-ratio: 1 / 1;
          border: 1.5px solid var(--ink);
          animation: l3a-enter 0.35s ease forwards;
        }
        .l3a-quad canvas.is-anomalous { border-color: var(--red); border-width: 2px; }
        .l3a-quad-label { font-size: 10px; font-weight: 700; letter-spacing: 0.04em; }
        .l3a-quad-hint { font-size: 9.5px; color: var(--ink-45); }

        .l3a-baro { margin-top: 4px; }
        .l3a-score-row {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          margin-bottom: 8px;
        }
        .l3a-score-num { font-size: 22px; font-weight: 700; }
        .l3a-score-delta { font-size: 10.5px; color: var(--ink-45); }

        .l3a-baro-track {
          position: relative;
          display: flex;
          height: 16px;
          border: 1.5px solid var(--ink);
          background: var(--paper-2);
          margin-top: 18px;
        }
        .l3a-baro-zone { border-right: 1px dashed var(--line); }
        .l3a-baro-zone:last-child { border-right: none; }
        .l3a-baro-zone--clean { flex: 0 0 40%; }
        .l3a-baro-zone--suspicious { flex: 0 0 30%; }
        .l3a-baro-zone--malicious { flex: 0 0 30%; }

        .l3a-baro-marker {
          position: absolute;
          top: -22px;
          transform: translateX(-50%);
          font-size: 10px;
          font-weight: 700;
          padding: 3px 8px;
          border-radius: 999px;
          border: 1.5px solid var(--ink);
          white-space: nowrap;
          transition: left 0.85s cubic-bezier(0.22,1,0.36,1);
        }
        .l3a-baro-marker::after {
          content: '';
          position: absolute;
          left: 50%;
          bottom: -20px;
          transform: translateX(-50%);
          width: 2px;
          height: 16px;
          background: var(--ink);
        }

        .l3a-baro-labels {
          display: flex;
          justify-content: space-between;
          font-size: 9px;
          color: var(--ink-45);
          margin-top: 6px;
          letter-spacing: 0.04em;
        }
        .l3a-baro-ticks {
          position: relative;
          height: 12px;
          font-size: 8.5px;
          color: var(--ink-45);
        }
        .l3a-baro-tick {
          position: absolute;
          top: 0;
          transform: translateX(-50%);
        }

        .l3a-signal-list { display: flex; flex-direction: column; gap: 8px; min-height: 210px; }
        .l3a-signal {
          border: 1.5px solid var(--line);
          border-left-width: 4px;
          padding: 8px 10px;
          background: var(--paper);
          opacity: 0;
          transform: translateX(-8px);
          animation: l3a-enter 0.35s ease forwards;
        }
        .l3a-signal.is-run { border-left-color: var(--yellow); }
        .l3a-signal.is-run.is-active { border-left-color: var(--red); background: #F7DFDA; }
        .l3a-signal.is-gated { border-left-color: var(--ink-45); opacity: 0.55 !important; transform: none !important; }
        .l3a-signal-head { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
        .l3a-signal-title { font-size: 10.5px; font-weight: 700; letter-spacing: 0.03em; }
        .l3a-signal-value { font-size: 12.5px; font-weight: 700; margin-bottom: 2px; }
        .l3a-signal-detail { font-size: 10.5px; color: var(--ink-70); line-height: 1.4; }

        @keyframes l3a-enter { to { opacity: 1; transform: translateX(0); } }

        .l3a-dossier { border-top: 2px solid var(--ink); padding: 18px 22px 22px; position: relative; z-index: 2; }
        .l3a-terminal {
          background: var(--ink);
          color: var(--paper);
          border: 1.5px solid var(--ink);
          padding: 14px 16px;
          font-size: 11.5px;
          line-height: 1.6;
          min-height: 90px;
        }
        .l3a-term-dim { color: rgba(242,236,220,0.45); }
        .l3a-term-block {
          margin-bottom: 10px;
          opacity: 0;
          transform: translateY(6px);
          animation: l3a-term-enter 0.4s ease forwards;
        }
        .l3a-term-block:last-child { margin-bottom: 0; }
        @keyframes l3a-term-enter { to { opacity: 1; transform: translateY(0); } }
        .l3a-term-heading { font-weight: 700; letter-spacing: 0.08em; color: var(--yellow); margin-bottom: 2px; }
        .l3a-term-body { color: rgba(242,236,220,0.88); }
        .l3a-cursor { display: inline-block; animation: l3a-blink 1s step-end infinite; color: var(--yellow); }
        @keyframes l3a-blink { 50% { opacity: 0; } }

        .l3a-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
          padding: 16px 22px;
          border-top: 2px solid var(--ink);
          background: var(--paper-2);
        }
        .l3a-metrics { display: flex; gap: 22px; font-size: 11px; letter-spacing: 0.05em; }
        .l3a-metrics b { display: block; font-size: 14px; letter-spacing: 0; margin-top: 2px; }
        .l3a-verdict-tag { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; }
        .l3a-verdict-tag--MALICIOUS { color: var(--red); }
        .l3a-verdict-tag--SUSPICIOUS { color: #9c7a12; }
        .l3a-verdict-tag--CLEAN { color: var(--blue); }

        .l3a-controls { display: flex; gap: 8px; flex-wrap: wrap; }
        .l3a-btn {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10.5px;
          font-weight: 700;
          letter-spacing: 0.06em;
          border: 2px solid var(--ink);
          background: var(--yellow);
          padding: 9px 14px;
          border-radius: 999px;
          display: inline-flex;
          align-items: center;
          gap: 6px;
          cursor: pointer;
        }
        .l3a-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .l3a-btn--ghost { background: var(--paper); }
        .l3a-btn--red { background: var(--red); color: var(--paper); }
        .l3a-btn:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }
      `}</style>

            <div className="l3a-ticker" aria-hidden="true">
                <span className="l3a-ticker-track">
                    512×512 GALOIS FIELD RASTERIZATION&nbsp;&nbsp;*&nbsp;&nbsp;CONTRASTIVE OSL EMBEDDING&nbsp;&nbsp;*&nbsp;&nbsp;{"\u03C4"} = 0.072236&nbsp;&nbsp;*&nbsp;&nbsp;5-SIGNAL FORENSIC BREAKDOWN&nbsp;&nbsp;*&nbsp;&nbsp;GEMINI THREAT DOSSIER&nbsp;&nbsp;*&nbsp;&nbsp;512×512 GALOIS FIELD RASTERIZATION&nbsp;&nbsp;*&nbsp;&nbsp;CONTRASTIVE OSL EMBEDDING&nbsp;&nbsp;*&nbsp;&nbsp;{"\u03C4"} = 0.072236&nbsp;&nbsp;*&nbsp;&nbsp;5-SIGNAL FORENSIC BREAKDOWN&nbsp;&nbsp;*&nbsp;&nbsp;
                </span>
            </div>

            <div className="l3a-header">
                <div>
                    <Eyebrow tone="yellow">LAYER 03a — 512×512 STEGANALYSIS &amp; FORENSIC ENGINE</Eyebrow>
                    <h2 className="l3a-title-serif">
                        Rasterizes the weights.<br /><em>Interrogates</em> the noise.
                    </h2>
                    <p className="l3a-sub">
                        Flat float32 weights are rasterized into a unified 512{"\u00D7"}512 Galois Field
                        4-quadrant plane (Q0 MSB {"\u2192"} Q3 LSB) and scored against a fixed contrastive
                        margin ({"\u03C4"} = {TAU}). Crossing it opens a signal-gated forensic breakdown and a live
                        Gemini incident dossier.
                    </p>
                </div>
                <div className="l3a-meta">
                    CASE FILE<br />{s ? s.file : "awaiting sample"}
                </div>
            </div>

            <div className="l3a-stage">
                <div className="l3a-three-mount" ref={mountRef} />

                <div className="l3a-panel">
                    <div className="l3a-panel-label">512×512 GALOIS FIELD RASTERIZATION</div>
                    {s ? (
                        <div className="l3a-quad-grid">
                            {QUAD_META.map((q, i) => (
                                <QuadrantCanvas
                                    key={q.key}
                                    runId={runId}
                                    density={s.quadrantNoise[i]}
                                    label={q.label}
                                    hint={q.hint}
                                />
                            ))}
                        </div>
                    ) : (
                        <p className="l3a-placeholder">
                            Run a sample below to rasterize its weights into a unified 512{"\u00D7"}512 Galois Field
                            4-quadrant plane (Q0 MSB {"\u2192"} Q3 LSB).
                        </p>
                    )}

                    <div className="l3a-panel-label">FORENSIC BAROMETER</div>
                    <div className="l3a-baro">
                        <div className="l3a-score-row">
                            <span className="l3a-score-num">{s ? scoreVal.toFixed(3) : "\u2014"}</span>
                            {s && s.delta !== null && bandRevealed ? (
                                <span className="l3a-score-delta">
                                    {"\u0394"} = {s.delta >= 0 ? "+" : ""}
                                    {s.delta.toFixed(4)} ({"\u03C4"} = {TAU})
                                </span>
                            ) : null}
                            {s && bandRevealed ? (
                                <Eyebrow tone={s.bandTone}>{s.band}</Eyebrow>
                            ) : null}
                        </div>
                        <div className="l3a-baro-track">
                            <div className="l3a-baro-zone l3a-baro-zone--clean" />
                            <div className="l3a-baro-zone l3a-baro-zone--suspicious" />
                            <div className="l3a-baro-zone l3a-baro-zone--malicious" />
                            {s ? (
                                <div
                                    className="l3a-baro-marker"
                                    style={{
                                        left: `${markerPct}%`,
                                        background: s.bandColor,
                                        color: s.bandTone === "suspicious" ? "var(--ink)" : "var(--paper)",
                                    }}
                                >
                                    {scoreVal.toFixed(2)}
                                </div>
                            ) : null}
                        </div>
                        <div className="l3a-baro-ticks">
                            <span className="l3a-baro-tick" style={{ left: "40%" }}>0.40</span>
                            <span className="l3a-baro-tick" style={{ left: "70%" }}>0.70</span>
                        </div>
                        <div className="l3a-baro-labels">
                            <span>CLEAN</span>
                            <span>SUSPICIOUS</span>
                            <span>MALICIOUS</span>
                        </div>
                    </div>
                </div>

                <div className="l3a-panel">
                    <div className="l3a-panel-label">5-SIGNAL FORENSIC BREAKDOWN</div>
                    {!s && (
                        <p className="l3a-placeholder">
                            Signal gating: 1 &amp; 3 run at SUSPICIOUS, all five run at MALICIOUS.
                            Nothing runs below 0.40.
                        </p>
                    )}
                    {s && s.signals.length === 0 && (
                        <div className="l3a-signal is-run" style={{ opacity: 1, transform: "none", borderLeftColor: "var(--blue)" }}>
                            <div className="l3a-signal-head">
                                <ShieldCheck size={13} color="#2F4FE0" />
                                <span className="l3a-signal-title">ZERO FORENSIC BREAKDOWN</span>
                            </div>
                            <div className="l3a-signal-detail">
                                Score below 0.40 {"\u2014"} routed directly to production ingestion. No signals executed.
                            </div>
                        </div>
                    )}
                    {s && s.signals.length > 0 && (
                        <div className="l3a-signal-list">
                            {signalsRevealed.map((sig) => (
                                <SignalCard key={sig.key} signal={sig} active={sig.key === "s5" && s.id === "malicious"} />
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <div className="l3a-dossier">
                <div className="l3a-panel-label">
                    <Terminal size={11} style={{ verticalAlign: "-2px", marginRight: 5 }} />
                    GEMINI THREAT DOSSIER {"\u2014"} LIVE
                </div>
                <div className="l3a-terminal">
                    {!s && <span className="l3a-term-dim">Awaiting escalation. A dossier compiles only for SUSPICIOUS and MALICIOUS verdicts.</span>}
                    {s && s.gemini.length === 0 && bandRevealed && (
                        <span className="l3a-term-dim">$ NO DOSSIER REQUIRED {"\u2014"} clean verdict, no signals escalated.</span>
                    )}
                    {s && s.gemini.length === 0 && !bandRevealed && (
                        <span className="l3a-term-dim">Scoring sample against contrastive margin...</span>
                    )}
                    {s && geminiRevealed.length === 0 && s.gemini.length > 0 && bandRevealed && (
                        <span className="l3a-term-dim">Compiling forensic signals into a prompt for Gemini...</span>
                    )}
                    {geminiRevealed.map((sec) => (
                        <div className="l3a-term-block" key={sec.h}>
                            <div className="l3a-term-heading">{"\u25B8"} {sec.h}</div>
                            <div className="l3a-term-body">{sec.b}</div>
                        </div>
                    ))}
                    {running && bandRevealed ? <span className="l3a-cursor">{"\u258C"}</span> : null}
                </div>
            </div>

            <div className="l3a-footer">
                <div className="l3a-metrics">
                    <div>
                        LATENCY
                        <b>{latency.toFixed(1)}ms</b>
                    </div>
                    <div>
                        VERDICT
                        <b>
                            {verdict === "MALICIOUS" && (
                                <span className="l3a-verdict-tag l3a-verdict-tag--MALICIOUS">
                                    <ShieldAlert size={14} /> MALICIOUS
                                </span>
                            )}
                            {verdict === "SUSPICIOUS" && (
                                <span className="l3a-verdict-tag l3a-verdict-tag--SUSPICIOUS">
                                    <AlertTriangle size={14} /> SUSPICIOUS
                                </span>
                            )}
                            {verdict === "CLEAN" && (
                                <span className="l3a-verdict-tag l3a-verdict-tag--CLEAN">
                                    <ShieldCheck size={14} /> CLEARED
                                </span>
                            )}
                            {!verdict && "\u2014"}
                        </b>
                    </div>
                </div>

                <div className="l3a-controls">
                    <button className="l3a-btn l3a-btn--red" disabled={running} onClick={() => run("malicious")}>
                        <Play size={12} /> MALICIOUS
                    </button>
                    <button className="l3a-btn" disabled={running} onClick={() => run("suspicious")}>
                        <Play size={12} /> SUSPICIOUS
                    </button>
                    <button className="l3a-btn l3a-btn--ghost" disabled={running} onClick={() => run("clean")}>
                        <Play size={12} /> CLEAN
                    </button>
                    <button className="l3a-btn l3a-btn--ghost" onClick={reset} aria-label="Reset">
                        <RotateCcw size={13} />
                    </button>
                </div>
            </div>
        </div>
    );
}