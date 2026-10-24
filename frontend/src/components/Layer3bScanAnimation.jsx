import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import * as THREE from "three";
import {
    ShieldAlert,
    ShieldCheck,
    AlertTriangle,
    Play,
    RotateCcw,
    Activity,
    Radar,
    FileText,
} from "lucide-react";

/**
 * LAYER 3b — PRUNING-SAFE REFERENCE-FREE SPECTRAL BACKDOOR DETECTOR
 * -----------------------------------------------------------------------
 * Same "Field Notes" case-file system as Layer 3a: cream paper, warm
 * near-black ink, three flat accent colors, Fraunces for voice,
 * JetBrains Mono for every technical surface, the rubber-stamp verdict
 * as the one bold move. Three.js is scoped to the same two jobs: the
 * faint paper-grain field and the 3D stamp-slam.
 *
 * Surfaces specific to Layer 3b (v5 Pruning-Safe Engine):
 *  - SVD singular-value spectral decay curve (σ1 → σ10), cliff vs. smooth
 *    (active-submatrix only, min dim ≥ 32, eliminates low-rank bottleneck FPs)
 *  - Neuron L2 energy profile bar chart with p90 tail outlier isolation
 *    (pruning-safe: dead channels excluded; prevents SE/MobileNetV3 FPs)
 *  - Excess-kurtosis distribution gauge, Gaussian → heavy-tailed
 *    (gated to spatial kernels kH×kW > 1; bypasses 1×1 pointwise projections)
 *  - 3-signal geometric breakdown (always runs — reference-free, no gating)
 *  - Flagged-layer dossier with exact consensus-scoring arithmetic (+0.40 boost)
 *  - Disarmament integration: flagged tensors (σ1/σ2 ≥ 10.0) are passed to
 *    Phase 3 SVD rank-1 deflation (W ← W − σ1 u1 v1ᵀ) in the 4-phase pipeline
 */

const THRESHOLDS = {
    neuron: { trigger: 15.0, baseline: "1.2×–2.5×" },
    svd: { trigger: 10.0, baseline: "≤ 3.2×" },
    kurtosis: { trigger: 100.0, baseline: "< 40.0" },
};

// ---------------------------------------------------------------------
// Scenario data — mirrors the three test cases in the Layer 3b spec.
// ---------------------------------------------------------------------
const SCENARIOS = {
    trojan: {
        id: "trojan",
        file: "resnet50_backdoor_trojan.pt",
        targetLayer: "layer4.1.conv2.weight",
        verdictBool: true,
        band: "FLAGGED",
        stampLabel: "TROJAN DETECTED",
        bandColor: "#C4402E",
        score: 0.75,
        scoreFormula: "clip(1 layer × 0.35 + 0.40 consensus, 0.35, 1.0) = 0.750",
        latency: 14.2,
        consensusCount: 3,
        consensusLabel: "3/3 SIGNALS — CRITICAL CONSENSUS",
        svd: [100, 6.8, 5.8, 4.9, 4.2, 3.5, 3.0, 2.6, 2.2, 1.8],
        svdRatio: "14.8×",
        neurons: [34, 42, 29, 45, 38, 31, 47, 36, 40, 33, 100, 39],
        neuronOutlierIdx: 10,
        neuronRatio: "22.4×",
        kurtosis: 142.0,
        kurtosisBand: "extreme",
        signals: [
            {
                key: "neuron",
                title: "SIGNAL 1 — NEURON L2 NORM RATIO",
                value: "22.4×",
                threshold: "≥ 15.0×",
                state: "flagged",
                detail:
                    "One channel carries a magnitude spike far outside every other neuron in the layer — the geometric fingerprint of an inserted trigger pathway.",
            },
            {
                key: "svd",
                title: "SIGNAL 2 — SVD SPECTRAL RATIO (σ1/σ2)",
                value: "14.8×",
                threshold: "≥ 10.0×",
                state: "flagged",
                detail:
                    "The spectrum collapses onto a single dominant direction. Clean layers decay smoothly; this one doesn't.",
            },
            {
                key: "kurtosis",
                title: "SIGNAL 3 — EXCESS KURTOSIS (γ2)",
                value: "142.0",
                threshold: "≥ 100.0",
                state: "flagged",
                detail:
                    "Heavy-tailed fourth moment. A bimodal trigger distribution is superimposed on otherwise benign weights.",
            },
        ],
        dossierNote:
            "All three geometric invariants cross their calibrated line on the same tensor. Consensus scoring adds the +0.40 bonus on top of the per-layer base.",
    },
    drift: {
        id: "drift",
        file: "resnet152_quant_drift.pt",
        targetLayer: "conv1.weight",
        verdictBool: false,
        band: "CLEARED",
        stampLabel: "CLEARED",
        bandColor: "#2F4FE0",
        score: 0.0,
        scoreFormula: "no layer crosses threshold → score = 0.000",
        latency: 12.1,
        consensusCount: 0,
        consensusLabel: "0/3 SIGNALS — BORDERLINE ONLY",
        svd: [100, 16.1, 13.7, 11.7, 9.9, 8.4, 7.2, 6.1, 5.2, 4.4],
        svdRatio: "6.2×",
        neurons: [36, 44, 31, 47, 40, 33, 49, 38, 42, 35, 68, 41],
        neuronOutlierIdx: 10,
        neuronRatio: "9.2×",
        kurtosis: 58.4,
        kurtosisBand: "elevated",
        signals: [
            {
                key: "neuron",
                title: "SIGNAL 1 — NEURON L2 NORM RATIO",
                value: "9.2×",
                threshold: "≥ 15.0×",
                state: "elevated",
                detail:
                    "Above the 1.2×–2.5× clean baseline but well short of the trigger line — consistent with post-training quantization drift, not a trojan.",
            },
            {
                key: "svd",
                title: "SIGNAL 2 — SVD SPECTRAL RATIO (σ1/σ2)",
                value: "6.2×",
                threshold: "≥ 10.0×",
                state: "elevated",
                detail:
                    "Decay past σ2 stays smooth — no rank collapse. An elevated ratio alone doesn't clear the spectral-collapse bar.",
            },
            {
                key: "kurtosis",
                title: "SIGNAL 3 — EXCESS KURTOSIS (γ2)",
                value: "58.4",
                threshold: "≥ 100.0",
                state: "elevated",
                detail:
                    "Above the 40.0 clean ceiling but far under the 100.0 extreme line. Not flagged on its own.",
            },
        ],
        dossierNote:
            "Every signal reads above its clean baseline on this tensor, but none crosses its individual threshold — zero consensus, zero score. Worth a second look if drift continues across checkpoints.",
    },
    clean: {
        id: "clean",
        file: "mobilenet_v3_edge.pt",
        targetLayer: null,
        verdictBool: false,
        band: "CLEARED",
        stampLabel: "CLEARED",
        bandColor: "#2F4FE0",
        score: 0.0,
        scoreFormula: "flagged_layer_names = [] → score = 0.000",
        latency: 3.8,
        consensusCount: 0,
        consensusLabel: "0/3 SIGNALS — NOMINAL",
        svd: [100, 47.6, 40.5, 34.4, 29.2, 24.8, 21.1, 17.9, 15.2, 12.9],
        svdRatio: "2.1×",
        neurons: [38, 45, 33, 48, 41, 35, 50, 40, 44, 37, 46, 42],
        neuronOutlierIdx: -1,
        neuronRatio: "1.3×",
        kurtosis: 4.2,
        kurtosisBand: "nominal",
        signals: [
            {
                key: "neuron",
                title: "SIGNAL 1 — NEURON L2 NORM RATIO",
                value: "1.3×",
                threshold: "≥ 15.0×",
                state: "nominal",
                detail: "Sits inside the 1.2×–2.5× clean baseline across every layer in the model.",
            },
            {
                key: "svd",
                title: "SIGNAL 2 — SVD SPECTRAL RATIO (σ1/σ2)",
                value: "2.1×",
                threshold: "≥ 10.0×",
                state: "nominal",
                detail: "Textbook power-law decay end to end — no dominant directional collapse.",
            },
            {
                key: "kurtosis",
                title: "SIGNAL 3 — EXCESS KURTOSIS (γ2)",
                value: "4.2",
                threshold: "≥ 100.0",
                state: "nominal",
                detail: "Comfortably inside the zero-centered Gaussian / Laplace range.",
            },
        ],
        dossierNote:
            "No layer in this state_dict violates a calibrated physical threshold. Nothing to name, nothing to escalate.",
    },
};

// ---------------------------------------------------------------------
// Distribution-shape SVG paths for the kurtosis gauge icon.
// Hand-tuned silhouettes, not literal density curves — nominal reads as
// a plain Gaussian bell, extreme reads as a narrow spike with fat tails.
// ---------------------------------------------------------------------
const DIST_PATHS = {
    nominal: "M6,62 C42,62 54,10 100,10 C146,10 158,62 194,62",
    elevated: "M6,62 C50,62 62,5 100,5 C138,5 150,62 194,62",
    extreme: "M6,50 C58,50 90,4 100,4 C110,4 142,50 194,50",
};

// ---------------------------------------------------------------------
// Three.js layer: paper grain + rubber-stamp slam. Identical mechanism
// to Layer 3a — generic label/color in, animated slam out.
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
    return <span className={`l3b-pill l3b-pill--${tone}`}>{children}</span>;
}

function svdPaths(values, w = 300, h = 108) {
    const step = w / (values.length - 1);
    const pts = values.map((v, i) => [i * step, h - (v / 100) * h]);
    const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    const area = `${line} L${w},${h} L0,${h} Z`;
    return { line, area, pts };
}

function SvdChart({ scenario, revealed }) {
    const { line, area, pts } = useMemo(() => svdPaths(scenario.svd), [scenario]);
    return (
        <div className="l3b-svd">
            <div
                className="l3b-svd-clip"
                style={{ clipPath: revealed ? "inset(0 0% 0 0)" : "inset(0 100% 0 0)" }}
            >
                <svg viewBox="0 0 300 108" className="l3b-svd-svg" preserveAspectRatio="none">
                    <line x1="0" y1="108" x2="300" y2="108" stroke="var(--line)" strokeWidth="1" />
                    <path d={area} fill={scenario.bandColor} opacity="0.14" />
                    <path d={line} fill="none" stroke={scenario.bandColor} strokeWidth="2.4" />
                    {pts.map((p, i) => (
                        <circle
                            key={i}
                            cx={p[0]}
                            cy={p[1]}
                            r={i === 0 || i === 1 ? 3.4 : 2.2}
                            fill={scenario.bandColor}
                            stroke="var(--paper)"
                            strokeWidth="1"
                        />
                    ))}
                </svg>
            </div>
            <div className="l3b-svd-axis">
                <span>σ1</span>
                <span>σ10</span>
            </div>
            <div className="l3b-svd-readout">
                σ1 / σ2 = <b style={{ color: scenario.bandColor }}>{scenario.svdRatio}</b>
                <span className="l3b-dim"> · trigger ≥ {THRESHOLDS.svd.trigger.toFixed(1)}× · baseline {THRESHOLDS.svd.baseline}</span>
            </div>
        </div>
    );
}

function NeuronChart({ scenario, revealed }) {
    return (
        <div className="l3b-neurons">
            <div className="l3b-neuron-bars">
                {scenario.neurons.map((v, i) => {
                    const isOutlier = i === scenario.neuronOutlierIdx;
                    const cls = isOutlier
                        ? scenario.verdictBool
                            ? "is-flagged"
                            : "is-elevated"
                        : "";
                    return (
                        <div className="l3b-neuron-col" key={`${scenario.id}-${i}`}>
                            <div
                                className={`l3b-neuron-bar ${cls}`}
                                style={{
                                    height: revealed ? `${v}%` : "0%",
                                    transitionDelay: revealed ? `${i * 35}ms` : "0ms",
                                }}
                            />
                        </div>
                    );
                })}
            </div>
            <div className="l3b-svd-readout">
                max / median = <b style={{ color: scenario.neuronOutlierIdx >= 0 ? scenario.bandColor : "var(--ink)" }}>{scenario.neuronRatio}</b>
                <span className="l3b-dim"> · trigger ≥ {THRESHOLDS.neuron.trigger.toFixed(1)}× · baseline {THRESHOLDS.neuron.baseline}</span>
            </div>
        </div>
    );
}

function KurtosisGauge({ scenario, revealed }) {
    const scaleMax = 160;
    const pct = Math.max(0, Math.min(100, (scenario.kurtosis / scaleMax) * 100));
    return (
        <div className="l3b-kurt">
            <div className="l3b-kurt-row">
                <svg viewBox="0 0 200 70" className="l3b-kurt-dist">
                    <path
                        d={DIST_PATHS[scenario.kurtosisBand]}
                        fill="none"
                        stroke={scenario.bandColor}
                        strokeWidth="3"
                        strokeLinecap="round"
                    />
                </svg>
                <div className="l3b-kurt-num">
                    {"\u03B3"}<tspan style={{ fontSize: "9px" }}>2</tspan> = <b>{scenario.kurtosis.toFixed(1)}</b>
                </div>
            </div>
            <div className="l3b-baro-track">
                <div className="l3b-baro-zone l3b-baro-zone--nominal" />
                <div className="l3b-baro-zone l3b-baro-zone--elevated" />
                <div className="l3b-baro-zone l3b-baro-zone--extreme" />
                {revealed ? (
                    <div
                        className="l3b-baro-marker"
                        style={{ left: `${pct}%`, background: scenario.bandColor }}
                    >
                        {scenario.kurtosis.toFixed(1)}
                    </div>
                ) : null}
            </div>
            <div className="l3b-baro-labels">
                <span>NOMINAL &lt; 40</span>
                <span>ELEVATED</span>
                <span>EXTREME ≥ 100</span>
            </div>
        </div>
    );
}

function SignalCard({ signal, active }) {
    return (
        <div className={`l3b-signal is-${signal.state} ${active ? "is-active" : ""}`}>
            <div className="l3b-signal-head">
                <span className="l3b-signal-title">{signal.title}</span>
                <span className={`l3b-signal-tag l3b-signal-tag--${signal.state}`}>
                    {signal.state === "flagged" ? "FLAGGED" : signal.state === "elevated" ? "ELEVATED" : "NOMINAL"}
                </span>
            </div>
            <div className="l3b-signal-value">
                {signal.value} <span className="l3b-dim">/ {signal.threshold}</span>
            </div>
            <div className="l3b-signal-detail">{signal.detail}</div>
        </div>
    );
}

export default function Layer3bScanAnimation() {
    const mountRef = useRef(null);
    const stamp = useStampScene(mountRef);

    const [scenario, setScenario] = useState(null); // 'trojan' | 'drift' | 'clean' | null
    const [runId, setRunId] = useState(0);
    const [scoreVal, setScoreVal] = useState(0);
    const [bandRevealed, setBandRevealed] = useState(false);
    const [chartsRevealed, setChartsRevealed] = useState(false);
    const [signalsRevealed, setSignalsRevealed] = useState([]);
    const [kurtRevealed, setKurtRevealed] = useState(false);
    const [dossierRevealed, setDossierRevealed] = useState(false);
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
            setChartsRevealed(false);
            setSignalsRevealed([]);
            setKurtRevealed(false);
            setDossierRevealed(false);
            setVerdict(null);
            setLatency(0);
            setRunning(true);

            const scoreStart = 200;
            const scoreDur = 600;

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

            let cursor = scoreStart + scoreDur + 150;

            const chartsTimer = setTimeout(() => setChartsRevealed(true), cursor);
            timers.current.push(chartsTimer);
            cursor += 700;

            s.signals.forEach((sig, i) => {
                const t = setTimeout(() => {
                    setSignalsRevealed((prev) => [...prev, sig]);
                }, cursor + i * 220);
                timers.current.push(t);
            });
            cursor += s.signals.length * 220 + 150;

            const kurtTimer = setTimeout(() => setKurtRevealed(true), cursor);
            timers.current.push(kurtTimer);
            cursor += 500;

            const dossierTimer = setTimeout(() => setDossierRevealed(true), cursor);
            timers.current.push(dossierTimer);
            cursor += 400;

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
        setChartsRevealed(false);
        setSignalsRevealed([]);
        setKurtRevealed(false);
        setDossierRevealed(false);
        setVerdict(null);
        setLatency(0);
        setRunning(false);
    }, [clearTimers, stamp]);

    useEffect(() => () => clearTimers(), [clearTimers]);

    const s = scenario ? SCENARIOS[scenario] : null;
    const markerPct = Math.max(0, Math.min(100, (scoreVal / 1.0) * 100));

    return (
        <div className="l3b-wrap">
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,900;1,9..144,500&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

        .l3b-wrap {
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
        .l3b-wrap * { box-sizing: border-box; }

        .l3b-ticker {
          background: var(--ink);
          color: var(--paper);
          font-size: 11px;
          letter-spacing: 0.14em;
          padding: 7px 0;
          overflow: hidden;
          white-space: nowrap;
          border-bottom: 2px solid var(--ink);
        }
        .l3b-ticker-track {
          display: inline-block;
          padding-left: 100%;
          animation: l3b-scroll 22s linear infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .l3b-ticker-track { animation: none; padding-left: 2%; }
        }
        @keyframes l3b-scroll {
          from { transform: translateX(0); }
          to { transform: translateX(-100%); }
        }

        .l3b-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          padding: 22px 24px 16px;
          border-bottom: 2px solid var(--ink);
        }
        .l3b-title-serif {
          font-family: 'Fraunces', serif;
          font-weight: 900;
          font-size: 28px;
          line-height: 1.08;
          margin: 10px 0 4px;
          letter-spacing: -0.01em;
        }
        .l3b-title-serif em { font-style: italic; font-weight: 500; }
        .l3b-sub {
          font-size: 12px;
          color: var(--ink-70);
          max-width: 52ch;
          line-height: 1.5;
        }
        .l3b-meta {
          text-align: right;
          font-size: 11px;
          color: var(--ink-45);
          letter-spacing: 0.06em;
          white-space: nowrap;
        }

        .l3b-pill {
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
        .l3b-pill--yellow { background: var(--yellow); }
        .l3b-pill--blue { background: var(--blue); color: var(--paper); }
        .l3b-pill--red { background: var(--red); color: var(--paper); }
        .l3b-pill--paper { background: var(--paper); }
        .l3b-pill--cleared { background: var(--blue); color: var(--paper); }
        .l3b-pill--flagged { background: var(--red); color: var(--paper); }

        .l3b-stage {
          position: relative;
          display: grid;
          grid-template-columns: 1fr 1.05fr;
          gap: 0;
          min-height: 300px;
        }
        @media (max-width: 640px) {
          .l3b-stage { grid-template-columns: 1fr; }
        }

        .l3b-three-mount {
          position: absolute;
          inset: 0;
          z-index: 5;
          pointer-events: none;
        }

        .l3b-panel { padding: 18px 22px 22px; position: relative; z-index: 2; }
        .l3b-panel + .l3b-panel { border-left: 2px solid var(--ink); }
        @media (max-width: 640px) {
          .l3b-panel { min-height: 260px; }
          .l3b-panel + .l3b-panel { border-left: none; border-top: 2px solid var(--ink); }
        }

        .l3b-panel-label {
          font-size: 10.5px;
          font-weight: 700;
          letter-spacing: 0.14em;
          color: var(--ink-45);
          margin-bottom: 12px;
        }
        .l3b-panel-label + .l3b-panel-label { margin-top: 22px; }

        .l3b-placeholder {
          font-size: 11.5px;
          color: var(--ink-45);
          line-height: 1.5;
          max-width: 36ch;
        }

        .l3b-dim { color: var(--ink-45); }

        /* --- SVD spectral chart --- */
        .l3b-svd-clip {
          transition: clip-path 900ms cubic-bezier(0.22,1,0.36,1);
        }
        .l3b-svd-svg { width: 100%; height: 92px; display: block; }
        .l3b-svd-axis {
          display: flex;
          justify-content: space-between;
          font-size: 9px;
          color: var(--ink-45);
          letter-spacing: 0.06em;
          margin-top: 2px;
        }
        .l3b-svd-readout {
          font-size: 11px;
          font-weight: 700;
          margin-top: 8px;
        }
        .l3b-svd-readout .l3b-dim { font-weight: 500; font-size: 10px; }

        /* --- neuron bar chart --- */
        .l3b-neurons { margin-top: 4px; }
        .l3b-neuron-bars {
          display: flex;
          align-items: flex-end;
          gap: 4px;
          height: 80px;
          border-bottom: 1.5px solid var(--ink);
          padding-bottom: 1px;
        }
        .l3b-neuron-col { flex: 1; height: 100%; display: flex; align-items: flex-end; }
        .l3b-neuron-bar {
          width: 100%;
          background: var(--ink-45);
          height: 0%;
          transition: height 480ms cubic-bezier(0.22,1,0.36,1);
        }
        .l3b-neuron-bar.is-flagged { background: var(--red); }
        .l3b-neuron-bar.is-elevated { background: var(--amber); }

        /* --- score barometer (2-zone: clean / flagged) --- */
        .l3b-baro { margin-top: 4px; }
        .l3b-score-row {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          margin-bottom: 8px;
          gap: 8px;
        }
        .l3b-score-num { font-size: 22px; font-weight: 700; }
        .l3b-score-formula { font-size: 9.5px; color: var(--ink-45); }

        .l3b-baro-track {
          position: relative;
          display: flex;
          height: 16px;
          border: 1.5px solid var(--ink);
          background: var(--paper-2);
          margin-top: 18px;
        }
        .l3b-baro-zone { border-right: 1px dashed var(--line); }
        .l3b-baro-zone:last-child { border-right: none; }
        .l3b-baro-zone--clean { flex: 0 0 35%; }
        .l3b-baro-zone--flagged { flex: 0 0 65%; }
        .l3b-baro-zone--nominal { flex: 0 0 25%; }
        .l3b-baro-zone--elevated { flex: 0 0 37.5%; }
        .l3b-baro-zone--extreme { flex: 0 0 37.5%; }

        .l3b-baro-marker {
          position: absolute;
          top: -22px;
          transform: translateX(-50%);
          font-size: 10px;
          font-weight: 700;
          color: var(--paper);
          padding: 3px 8px;
          border-radius: 999px;
          border: 1.5px solid var(--ink);
          white-space: nowrap;
          transition: left 0.85s cubic-bezier(0.22,1,0.36,1);
        }
        .l3b-baro-marker::after {
          content: '';
          position: absolute;
          left: 50%;
          bottom: -20px;
          transform: translateX(-50%);
          width: 2px;
          height: 16px;
          background: var(--ink);
        }

        .l3b-baro-labels {
          display: flex;
          justify-content: space-between;
          font-size: 9px;
          color: var(--ink-45);
          margin-top: 6px;
          letter-spacing: 0.03em;
        }

        /* --- kurtosis gauge --- */
        .l3b-kurt { margin-top: 4px; }
        .l3b-kurt-row {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-bottom: 4px;
        }
        .l3b-kurt-dist { width: 96px; height: 34px; flex-shrink: 0; }
        .l3b-kurt-num { font-size: 13px; font-weight: 700; }

        /* --- signal cards --- */
        .l3b-signal-list { display: flex; flex-direction: column; gap: 8px; min-height: 190px; }
        .l3b-signal {
          border: 1.5px solid var(--line);
          border-left-width: 4px;
          padding: 8px 10px;
          background: var(--paper);
          opacity: 0;
          transform: translateX(-8px);
          animation: l3b-enter 0.35s ease forwards;
        }
        .l3b-signal.is-flagged { border-left-color: var(--red); }
        .l3b-signal.is-flagged.is-active { background: #F7DFDA; }
        .l3b-signal.is-elevated { border-left-color: var(--amber); }
        .l3b-signal.is-nominal { border-left-color: var(--blue); }
        .l3b-signal-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
          margin-bottom: 3px;
        }
        .l3b-signal-title { font-size: 10px; font-weight: 700; letter-spacing: 0.02em; }
        .l3b-signal-tag {
          font-size: 8.5px;
          font-weight: 700;
          letter-spacing: 0.08em;
          padding: 2px 6px;
          border-radius: 999px;
          border: 1px solid var(--ink);
          white-space: nowrap;
        }
        .l3b-signal-tag--flagged { background: var(--red); color: var(--paper); }
        .l3b-signal-tag--elevated { background: var(--amber); color: var(--paper); }
        .l3b-signal-tag--nominal { background: var(--paper); }
        .l3b-signal-value { font-size: 12.5px; font-weight: 700; margin-bottom: 2px; }
        .l3b-signal-detail { font-size: 10.5px; color: var(--ink-70); line-height: 1.4; }

        @keyframes l3b-enter { to { opacity: 1; transform: translateX(0); } }

        /* --- dossier --- */
        .l3b-dossier { border-top: 2px solid var(--ink); padding: 18px 22px 22px; position: relative; z-index: 2; }
        .l3b-dossier-card {
          border: 1.5px solid var(--ink);
          background: var(--paper-2);
          padding: 14px 16px;
          font-size: 11.5px;
          line-height: 1.6;
          min-height: 90px;
          opacity: 0;
          transform: translateY(6px);
        }
        .l3b-dossier-card.is-revealed {
          animation: l3b-dossier-enter 0.4s ease forwards;
        }
        @keyframes l3b-dossier-enter { to { opacity: 1; transform: translateY(0); } }
        .l3b-dossier-row {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 10px;
          margin-bottom: 8px;
          flex-wrap: wrap;
        }
        .l3b-dossier-layer { font-weight: 700; font-size: 12.5px; }
        .l3b-dossier-note { color: var(--ink-70); }
        .l3b-dossier-formula {
          margin-top: 8px;
          font-size: 10.5px;
          color: var(--ink-45);
          border-top: 1px dashed var(--line);
          padding-top: 8px;
        }

        .l3b-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
          padding: 16px 22px;
          border-top: 2px solid var(--ink);
          background: var(--paper-2);
        }
        .l3b-metrics { display: flex; gap: 22px; font-size: 11px; letter-spacing: 0.05em; flex-wrap: wrap; }
        .l3b-metrics b { display: block; font-size: 14px; letter-spacing: 0; margin-top: 2px; }
        .l3b-verdict-tag { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; }
        .l3b-verdict-tag--FLAGGED { color: var(--red); }
        .l3b-verdict-tag--CLEARED { color: var(--blue); }

        .l3b-controls { display: flex; gap: 8px; flex-wrap: wrap; }
        .l3b-btn {
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
        .l3b-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .l3b-btn--ghost { background: var(--paper); }
        .l3b-btn--red { background: var(--red); color: var(--paper); }
        .l3b-btn:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }
      `}</style>

            <div className="l3b-ticker" aria-hidden="true">
                <span className="l3b-ticker-track">
                    REFERENCE-FREE WEIGHT FORENSICS&nbsp;&nbsp;*&nbsp;&nbsp;SVD SPECTRAL COLLAPSE {"\u03C3"}1/{"\u03C3"}2&nbsp;&nbsp;*&nbsp;&nbsp;NEURON L2 NORM RATIO&nbsp;&nbsp;*&nbsp;&nbsp;BIMODAL EXCESS KURTOSIS {"\u03B3"}2&nbsp;&nbsp;*&nbsp;&nbsp;MULTI-SIGNAL CONSENSUS ENGINE&nbsp;&nbsp;*&nbsp;&nbsp;REFERENCE-FREE WEIGHT FORENSICS&nbsp;&nbsp;*&nbsp;&nbsp;SVD SPECTRAL COLLAPSE {"\u03C3"}1/{"\u03C3"}2&nbsp;&nbsp;*&nbsp;&nbsp;NEURON L2 NORM RATIO&nbsp;&nbsp;*&nbsp;&nbsp;
                </span>
            </div>

            <div className="l3b-header">
                <div>
                    <Eyebrow tone="yellow">LAYER 03b — SPECTRAL &amp; WEIGHT ANOMALY DETECTOR</Eyebrow>
                    <h2 className="l3b-title-serif">
                        No reference model.<br /><em>Just</em> geometry.
                    </h2>
                    <p className="l3b-sub">
                        Three geometric invariants are computed directly from raw float32 tensors {"\u2014"}
                        no clean baseline, no source training data. Each layer is judged against a fixed
                        physical threshold; two or more violations on the same tensor trigger consensus scoring.
                    </p>
                </div>
                <div className="l3b-meta">
                    CASE FILE<br />{s ? s.file : "awaiting state_dict"}
                </div>
            </div>

            <div className="l3b-stage">
                <div className="l3b-three-mount" ref={mountRef} />

                <div className="l3b-panel">
                    <div className="l3b-panel-label">
                        <Radar size={11} style={{ verticalAlign: "-2px", marginRight: 5 }} />
                        SVD SPECTRAL DECAY ({"\u03C3"}1 {"\u2192"} {"\u03C3"}10)
                    </div>
                    {s ? (
                        <SvdChart scenario={s} revealed={chartsRevealed} />
                    ) : (
                        <p className="l3b-placeholder">
                            Run a sample below to reshape its 2D weight tensor and take the SVD spectrum.
                            Clean layers decay smoothly; a rank-1 trojan injection collapses it.
                        </p>
                    )}

                    <div className="l3b-panel-label">
                        <Activity size={11} style={{ verticalAlign: "-2px", marginRight: 5 }} />
                        NEURON L2 ENERGY PROFILE
                    </div>
                    {s ? (
                        <NeuronChart scenario={s} revealed={chartsRevealed} />
                    ) : (
                        <p className="l3b-placeholder">
                            Per-neuron L2 norms across the flagged layer, relative to the layer median.
                        </p>
                    )}
                </div>

                <div className="l3b-panel">
                    <div className="l3b-panel-label">3-SIGNAL GEOMETRIC BREAKDOWN</div>
                    {!s && (
                        <p className="l3b-placeholder">
                            All three signals always run {"\u2014"} there's no reference model to gate against.
                            Consensus (2+ signals on one tensor) is what actually moves the score.
                        </p>
                    )}
                    {s && (
                        <div className="l3b-signal-list">
                            {signalsRevealed.map((sig) => (
                                <SignalCard key={sig.key} signal={sig} active={sig.state === "flagged"} />
                            ))}
                        </div>
                    )}

                    <div className="l3b-panel-label">EXCESS KURTOSIS ({"\u03B3"}2) GAUGE</div>
                    {s ? (
                        <KurtosisGauge scenario={s} revealed={kurtRevealed} />
                    ) : (
                        <p className="l3b-placeholder">
                            Fourth-moment distribution shape. Classification heads are excluded from this signal.
                        </p>
                    )}
                </div>
            </div>

            <div className="l3b-dossier">
                <div className="l3b-panel-label">
                    <FileText size={11} style={{ verticalAlign: "-2px", marginRight: 5 }} />
                    FLAGGED LAYER DOSSIER
                </div>
                <div className={`l3b-dossier-card ${dossierRevealed ? "is-revealed" : ""}`}>
                    {!s && <span className="l3b-dim">Awaiting a sample. A dossier compiles once signals resolve.</span>}
                    {s && dossierRevealed && (
                        <>
                            <div className="l3b-dossier-row">
                                <span className="l3b-dossier-layer">
                                    {s.targetLayer ? s.targetLayer : "\u2014 no layer named \u2014"}
                                </span>
                                <Eyebrow tone={s.verdictBool ? "flagged" : "cleared"}>{s.consensusLabel}</Eyebrow>
                            </div>
                            <div className="l3b-dossier-note">{s.dossierNote}</div>
                            <div className="l3b-dossier-formula">$ {s.scoreFormula}</div>
                        </>
                    )}
                </div>
            </div>

            <div className="l3b-footer">
                <div className="l3b-metrics">
                    <div>
                        LATENCY
                        <b>{latency.toFixed(1)}ms</b>
                    </div>
                    <div>
                        RISK SCORE
                        <b style={{ color: s && bandRevealed ? s.bandColor : "var(--ink)" }}>
                            {s ? scoreVal.toFixed(3) : "\u2014"}
                        </b>
                    </div>
                    <div>
                        VERDICT
                        <b>
                            {verdict === "FLAGGED" && (
                                <span className="l3b-verdict-tag l3b-verdict-tag--FLAGGED">
                                    <ShieldAlert size={14} /> TROJAN DETECTED
                                </span>
                            )}
                            {verdict === "CLEARED" && (
                                <span className="l3b-verdict-tag l3b-verdict-tag--CLEARED">
                                    <ShieldCheck size={14} /> CLEARED
                                </span>
                            )}
                            {!verdict && "\u2014"}
                        </b>
                    </div>
                </div>

                <div className="l3b-controls">
                    <button className="l3b-btn l3b-btn--red" disabled={running} onClick={() => run("trojan")}>
                        <Play size={12} /> TROJAN
                    </button>
                    <button className="l3b-btn" disabled={running} onClick={() => run("drift")}>
                        <AlertTriangle size={12} /> DRIFT
                    </button>
                    <button className="l3b-btn l3b-btn--ghost" disabled={running} onClick={() => run("clean")}>
                        <Play size={12} /> CLEAN
                    </button>
                    <button className="l3b-btn l3b-btn--ghost" onClick={reset} aria-label="Reset">
                        <RotateCcw size={13} />
                    </button>
                </div>
            </div>
        </div>
    );
}