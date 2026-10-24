import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { ShieldAlert, ShieldCheck, Play, RotateCcw, ArrowRight } from "lucide-react";

/**
 * LAYER 2 — STATISTICAL N-GRAM CLASSIFIER (XGBOOST)
 * -----------------------------------------------------------------------
 * Same case-file system as Layer 1 (cream paper, warm-black ink, Fraunces
 * + JetBrains Mono, flat red/blue/yellow accents). Layer 1 was a binary
 * verdict so it got one stamp. Layer 2 is a continuous probability from
 * an ensemble, so the signature here is the "jury of 100 trees" — a
 * literal grid of tiny decision-tree glyphs voting — plus a physical
 * gauge needle. It still closes on the same stamp motif as Layer 1 so
 * the two layers read as one series.
 */

// ---------------------------------------------------------------------
// Evidence data
// ---------------------------------------------------------------------
const MALICIOUS_BIGRAMS = [
  { id: "b1", from: "STACK_GLOBAL", to: "REDUCE", pct: "88.7%", risk: "high" },
  { id: "b2", from: "GLOBAL", to: "REDUCE", pct: "94.2%", risk: "high" },
  { id: "b3", from: "MARK", to: "GLOBAL", pct: "76.1%", risk: "high" },
  { id: "b4", from: "TUPLE", to: "REDUCE", pct: "82.0%", risk: "high" },
  { id: "b5", kind: "metric", label: "EVAL_RISK_RATIO", value: "0.184", tag: "CRITICAL", tone: "red" },
];

const CLEAN_BIGRAMS = [
  { id: "c1", from: "EMPTY_DICT", to: "SETITEMS", pct: "—", risk: "benign" },
  { id: "c2", from: "PROTO", to: "EMPTY_DICT", pct: "—", risk: "benign" },
  { id: "c3", kind: "metric", label: "EVAL_RISK_RATIO", value: "<0.001", tag: "NOMINAL", tone: "blue" },
];

const MALICIOUS_VOTES = (() => {
  const v = new Array(100).fill(true); // true = guilty / malicious
  [6, 19, 33, 58, 71, 84].forEach((i) => (v[i] = false)); // 6 dissenting trees -> 94/100
  return v;
})();

const CLEAN_VOTES = (() => {
  const v = new Array(100).fill(false);
  [12, 47].forEach((i) => (v[i] = true)); // 2 false-positive trees -> 2/100
  return v;
})();

function buildBars(scenario) {
  const maliciousSpikes = new Set([3, 7, 12, 18, 24, 27, 29, 31, 40, 41, 42]);
  const cleanSpikes = new Set([1, 4, 9, 15, 20, 26, 33, 40]);
  const bars = [];
  for (let i = 0; i < 43; i++) {
    const group = i < 26 ? "unigram" : i < 40 ? "bigram" : "structural";
    let value;
    let tone;
    if (scenario === "malicious") {
      value = maliciousSpikes.has(i) ? 0.78 + ((i * 37) % 20) / 100 : 0.06 + ((i * 13) % 14) / 100;
      tone = value > 0.55 ? "red" : "muted";
    } else {
      value = cleanSpikes.has(i) ? 0.32 + ((i * 19) % 10) / 100 : 0.04 + ((i * 11) % 10) / 100;
      tone = value > 0.28 ? "blue" : "muted";
    }
    bars.push({ i, group, value: Math.min(value, 0.98), tone });
  }
  return bars;
}

const BIGRAM_STEP_MS = 420;
const JURY_DURATION_MS = 1100;
const BARS_DELAY_MS = 250;
const BARS_SETTLE_MS = 550;
const GAUGE_DELAY_MS = 300;
const GAUGE_DURATION_MS = 850;
const STAMP_DELAY_MS = 250;

// ---------------------------------------------------------------------
// Three.js layer: paper grain + rubber-stamp slam (same mechanism as
// Layer 1, so the two components share a visual family).
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
      ctx.font = "800 78px 'JetBrains Mono', 'Courier New', monospace";
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
    const stampMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
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
// Gauge geometry — single formula reused for the arc, the threshold
// tick, and the needle rotation so all three stay aligned.
// ---------------------------------------------------------------------
const GAUGE_CX = 130;
const GAUGE_CY = 120;
const GAUGE_R = 96;
const angleForP = (p) => -90 + p * 180;
function polarToCartesian(cx, cy, r, angleDeg) {
  const a = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}
function describeArc(a0, a1) {
  const start = polarToCartesian(GAUGE_CX, GAUGE_CY, GAUGE_R, a1);
  const end = polarToCartesian(GAUGE_CX, GAUGE_CY, GAUGE_R, a0);
  const largeArc = a1 - a0 <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${GAUGE_R} ${GAUGE_R} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

function Gauge() {
  const tick0 = polarToCartesian(GAUGE_CX, GAUGE_CY, GAUGE_R - 10, angleForP(0.5));
  const tick1 = polarToCartesian(GAUGE_CX, GAUGE_CY, GAUGE_R + 10, angleForP(0.5));
  const label = polarToCartesian(GAUGE_CX, GAUGE_CY, GAUGE_R + 22, angleForP(0.5));
  return (
    <svg viewBox="0 0 260 140" width="260" height="140" className="l2-gauge-svg" aria-hidden="true">
      <path d={describeArc(-90, -36)} className="l2-arc l2-arc--low" />
      <path d={describeArc(-36, 18)} className="l2-arc l2-arc--mid" />
      <path d={describeArc(18, 90)} className="l2-arc l2-arc--high" />
      <line x1={tick0.x} y1={tick0.y} x2={tick1.x} y2={tick1.y} className="l2-threshold-tick" />
      <text x={label.x} y={label.y} textAnchor="middle" className="l2-threshold-label">
        0.50
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------
// Small presentational bits
// ---------------------------------------------------------------------
function Eyebrow({ children, tone = "yellow" }) {
  return <span className={`l2-pill l2-pill--${tone}`}>{children}</span>;
}

function BigramRow({ row }) {
  if (row.kind === "metric") {
    return (
      <div className={`l2-metricrow l2-metricrow--${row.tone}`}>
        <span className="l2-metriclabel">{row.label}</span>
        <span className="l2-metricvalue">{row.value}</span>
        <span className="l2-metrictag">{row.tag}</span>
      </div>
    );
  }
  return (
    <div className={`l2-bigram l2-bigram--${row.risk}`}>
      <span className="l2-bg-node">{row.from}</span>
      <ArrowRight size={12} className="l2-bg-arrow" />
      <span className="l2-bg-node">{row.to}</span>
      <span className="l2-bg-pct">{row.pct}</span>
    </div>
  );
}

function TreeVote({ active, guilty, delay }) {
  const tone = !active ? "idle" : guilty ? "red" : "blue";
  return (
    <svg
      viewBox="0 0 20 20"
      width="18"
      height="18"
      className={`l2-tree l2-tree--${tone} ${active ? "l2-tree--pop" : ""}`}
      style={active ? { animationDelay: `${delay}ms` } : undefined}
      aria-hidden="true"
    >
      <circle cx="10" cy="4" r="2.1" />
      <line x1="10" y1="6" x2="5" y2="12" />
      <line x1="10" y1="6" x2="15" y2="12" />
      <circle cx="5" cy="14" r="2.1" />
      <circle cx="15" cy="14" r="2.1" />
    </svg>
  );
}

export default function Layer2ScanAnimation() {
  const mountRef = useRef(null);
  const stamp = useStampScene(mountRef);

  const [scenario, setScenario] = useState(null);
  const [bigrams, setBigrams] = useState([]);
  const [juryPhase, setJuryPhase] = useState(false);
  const [juryVotes, setJuryVotes] = useState(() => new Array(100).fill(false));
  const [juryDisplay, setJuryDisplay] = useState({ voted: 0, guilty: 0, clear: 0 });
  const [barsPhase, setBarsPhase] = useState(false);
  const [bars, setBars] = useState(() => buildBars(null));
  const [needleP, setNeedleP] = useState(0);
  const [probability, setProbability] = useState(null);
  const [latency, setLatency] = useState(0);
  const [verdict, setVerdict] = useState(null);
  const [tags, setTags] = useState([]);
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

      setScenario(which);
      setBigrams([]);
      setJuryPhase(false);
      setJuryVotes(which === "malicious" ? MALICIOUS_VOTES : CLEAN_VOTES);
      setJuryDisplay({ voted: 0, guilty: 0, clear: 0 });
      setBarsPhase(false);
      setBars(buildBars(which));
      setNeedleP(0);
      setProbability(null);
      setLatency(0);
      setVerdict(null);
      setTags([]);
      setRunning(true);

      const rows = which === "malicious" ? MALICIOUS_BIGRAMS : CLEAN_BIGRAMS;
      rows.forEach((row, i) => {
        const t = setTimeout(() => setBigrams((prev) => [...prev, row]), i * BIGRAM_STEP_MS);
        timers.current.push(t);
      });
      const afterBigrams = rows.length * BIGRAM_STEP_MS + 200;

      timers.current.push(
        setTimeout(() => {
          setJuryPhase(true);
          const votes = which === "malicious" ? MALICIOUS_VOTES : CLEAN_VOTES;
          const finalGuilty = votes.filter(Boolean).length;
          const finalClear = 100 - finalGuilty;
          const start = performance.now();
          function tick() {
            const p = Math.min(1, (performance.now() - start) / JURY_DURATION_MS);
            setJuryDisplay({
              voted: Math.round(100 * p),
              guilty: Math.round(finalGuilty * p),
              clear: Math.round(finalClear * p),
            });
            if (p < 1) requestAnimationFrame(tick);
          }
          requestAnimationFrame(tick);
        }, afterBigrams)
      );

      const barsStart = afterBigrams + JURY_DURATION_MS + BARS_DELAY_MS;
      timers.current.push(setTimeout(() => setBarsPhase(true), barsStart));

      const gaugeStart = barsStart + BARS_SETTLE_MS + GAUGE_DELAY_MS;
      timers.current.push(
        setTimeout(() => {
          const p = which === "malicious" ? 0.97 : 0.014;
          setProbability(p);
          setNeedleP(p);
          runLatencyCount(which === "malicious" ? 8.4 : 6.1);
        }, gaugeStart)
      );

      const stampStart = gaugeStart + GAUGE_DURATION_MS + STAMP_DELAY_MS;
      timers.current.push(
        setTimeout(() => {
          if (which === "malicious") {
            setVerdict("malicious");
            stamp.current && stamp.current.fire("MALICIOUS", "#C4402E");
            setTags([
              { label: "bigram_STACK_GLOBAL→REDUCE", value: "+0.412" },
              { label: "unigram_REDUCE", value: "+0.285" },
              { label: "eval_risk_ratio", value: "+0.180" },
            ]);
          } else {
            setVerdict("cleared");
            stamp.current && stamp.current.fire("CLEARED", "#2F4FE0");
            setTags([]);
          }
          setRunning(false);
        }, stampStart)
      );
    },
    [clearTimers, runLatencyCount, stamp]
  );

  const reset = useCallback(() => {
    clearTimers();
    stamp.current && stamp.current.reset();
    setScenario(null);
    setBigrams([]);
    setJuryPhase(false);
    setJuryVotes(new Array(100).fill(false));
    setJuryDisplay({ voted: 0, guilty: 0, clear: 0 });
    setBarsPhase(false);
    setBars(buildBars(null));
    setNeedleP(0);
    setProbability(null);
    setLatency(0);
    setVerdict(null);
    setTags([]);
    setRunning(false);
  }, [clearTimers, stamp]);

  useEffect(() => () => clearTimers(), [clearTimers]);

  const groups = ["unigram", "bigram", "structural"];
  const groupMeta = {
    unigram: "UNIGRAMS · 26",
    bigram: "BIGRAMS · 14",
    structural: "STRUCTURAL · 3",
  };

  return (
    <div className="l2-wrap">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,900;1,9..144,500&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

        .l2-wrap {
          --paper: #F2ECDC;
          --paper-2: #E8DFC6;
          --ink: #17130E;
          --ink-70: rgba(23,19,14,0.72);
          --ink-45: rgba(23,19,14,0.45);
          --line: rgba(23,19,14,0.16);
          --red: #C4402E;
          --blue: #2F4FE0;
          --yellow: #EFC23A;
          font-family: 'JetBrains Mono', monospace;
          color: var(--ink);
          background: var(--paper);
          border: 2px solid var(--ink);
          max-width: 900px;
          margin: 0 auto;
          position: relative;
          overflow: hidden;
        }
        .l2-wrap * { box-sizing: border-box; }

        .l2-ticker {
          background: var(--ink);
          color: var(--paper);
          font-size: 11px;
          letter-spacing: 0.14em;
          padding: 7px 0;
          overflow: hidden;
          white-space: nowrap;
          border-bottom: 2px solid var(--ink);
        }
        .l2-ticker-track {
          display: inline-block;
          padding-left: 100%;
          animation: l2-scroll 20s linear infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .l2-ticker-track { animation: none; padding-left: 2%; }
        }
        @keyframes l2-scroll {
          from { transform: translateX(0); }
          to { transform: translateX(-100%); }
        }

        .l2-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          padding: 22px 24px 16px;
          border-bottom: 2px solid var(--ink);
        }
        .l2-title-serif {
          font-family: 'Fraunces', serif;
          font-weight: 900;
          font-size: 30px;
          line-height: 1.05;
          margin: 10px 0 4px;
          letter-spacing: -0.01em;
        }
        .l2-title-serif em { font-style: italic; font-weight: 500; }
        .l2-sub {
          font-size: 12px;
          color: var(--ink-70);
          max-width: 50ch;
          line-height: 1.5;
        }
        .l2-meta {
          text-align: right;
          font-size: 11px;
          color: var(--ink-45);
          letter-spacing: 0.06em;
          white-space: nowrap;
        }

        .l2-pill {
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
        .l2-pill--yellow { background: var(--yellow); }

        .l2-stage {
          position: relative;
          display: grid;
          grid-template-columns: 1.15fr 1fr;
        }
        @media (max-width: 640px) {
          .l2-stage { grid-template-columns: 1fr; }
        }

        .l2-three-mount { position: absolute; inset: 0; z-index: 5; pointer-events: none; }

        .l2-panel { padding: 18px 22px 20px; position: relative; z-index: 2; min-height: 300px; }
        .l2-panel + .l2-panel { border-left: 2px solid var(--ink); }
        @media (max-width: 640px) {
          .l2-panel { min-height: 260px; }
          .l2-panel + .l2-panel { border-left: none; border-top: 2px solid var(--ink); }
        }
        .l2-panel-label {
          font-size: 10.5px;
          font-weight: 700;
          letter-spacing: 0.14em;
          color: var(--ink-45);
          margin-bottom: 12px;
        }

        .l2-bigram, .l2-metricrow {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          padding: 7px 10px;
          border: 1.5px solid var(--line);
          border-left-width: 4px;
          background: var(--paper);
          margin-bottom: 6px;
          opacity: 0;
          transform: translateX(-8px);
          animation: l2-enter 0.35s ease forwards;
        }
        @keyframes l2-enter { to { opacity: 1; transform: translateX(0); } }
        .l2-bigram--high { border-left-color: var(--red); background: #F7DFDA; }
        .l2-bigram--benign { border-left-color: var(--blue); }
        .l2-bg-node { font-weight: 700; }
        .l2-bg-arrow { color: var(--ink-45); flex-shrink: 0; }
        .l2-bg-pct { margin-left: auto; color: var(--ink-70); font-size: 11px; }

        .l2-metricrow { border-left-width: 6px; font-weight: 700; }
        .l2-metricrow--red { border-left-color: var(--red); background: #F7DFDA; }
        .l2-metricrow--blue { border-left-color: var(--blue); background: #E4EAFB; }
        .l2-metriclabel { letter-spacing: 0.04em; }
        .l2-metricvalue { margin-left: auto; }
        .l2-metrictag {
          font-size: 9.5px;
          letter-spacing: 0.08em;
          padding: 2px 7px;
          border: 1px solid var(--ink);
          border-radius: 999px;
        }

        .l2-jury-grid {
          display: grid;
          grid-template-columns: repeat(10, 1fr);
          gap: 4px;
          max-width: 230px;
        }
        .l2-tree circle { fill: currentColor; }
        .l2-tree line { stroke: currentColor; fill: none; stroke-width: 1.6; }
        .l2-tree--idle { color: var(--line); }
        .l2-tree--red { color: var(--red); }
        .l2-tree--blue { color: var(--blue); }
        .l2-tree--pop { animation: l2-vote-in 0.3s ease forwards; }
        @keyframes l2-vote-in { from { opacity: 0.3; transform: scale(0.5); } to { opacity: 1; transform: scale(1); } }

        .l2-jury-tally { margin-top: 12px; font-size: 11px; color: var(--ink-70); }
        .l2-jury-tally b { color: var(--ink); }
        .l2-tally-red { color: var(--red); font-weight: 700; }
        .l2-tally-blue { color: var(--blue); font-weight: 700; }

        .l2-strip {
          display: flex;
          gap: 22px;
          padding: 16px 22px 20px;
          align-items: flex-end;
          border-top: 2px solid var(--ink);
          overflow-x: auto;
          position: relative;
          z-index: 2;
        }
        .l2-strip-group { display: flex; flex-direction: column; gap: 6px; }
        .l2-strip-caption { font-size: 9.5px; letter-spacing: 0.1em; color: var(--ink-45); }
        .l2-strip-bars { display: flex; align-items: flex-end; gap: 2px; height: 64px; }
        .l2-bar {
          width: 4px;
          background: var(--line);
          transition: height 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        .l2-bar--red { background: var(--red); }
        .l2-bar--blue { background: var(--blue); }
        .l2-bar--muted { background: var(--ink-45); opacity: 0.5; }

        .l2-results {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 20px;
          flex-wrap: wrap;
          padding: 16px 22px 4px;
          background: var(--paper-2);
          border-top: 2px solid var(--ink);
        }
        .l2-metrics { display: flex; gap: 22px; font-size: 11px; letter-spacing: 0.05em; }
        .l2-metrics b { display: block; font-size: 14px; letter-spacing: 0; margin-top: 2px; }
        .l2-verdict-tag { display: inline-flex; align-items: center; gap: 6px; }
        .l2-verdict-tag--malicious { color: var(--red); }
        .l2-verdict-tag--cleared { color: var(--blue); }

        .l2-gauge-wrap { position: relative; width: 260px; height: 200px; flex-shrink: 0; }
        .l2-arc { fill: none; stroke-width: 10; stroke-linecap: round; }
        .l2-arc--low { stroke: var(--blue); opacity: 0.85; }
        .l2-arc--mid { stroke: var(--yellow); opacity: 0.9; }
        .l2-arc--high { stroke: var(--red); opacity: 0.85; }
        .l2-threshold-tick { stroke: var(--ink); stroke-width: 2; }
        .l2-threshold-label {
          font-size: 9px;
          fill: var(--ink-45);
          font-family: 'JetBrains Mono', monospace;
        }
        .l2-needle {
          position: absolute;
          left: 130px;
          bottom: 80px;
          width: 3px;
          height: 88px;
          background: var(--ink);
          border-radius: 2px;
          transform-origin: 50% 100%;
          transition: transform 0.85s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .l2-needle-pivot {
          position: absolute;
          left: 130px;
          bottom: 80px;
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background: var(--ink);
          transform: translate(-50%, 50%);
        }
        .l2-gauge-readout {
          position: absolute;
          left: 50%;
          bottom: 4px;
          transform: translateX(-50%);
          text-align: center;
        }
        .l2-gauge-p { display: block; font-size: 22px; font-weight: 700; }
        .l2-gauge-p-label { font-size: 9.5px; letter-spacing: 0.1em; color: var(--ink-45); }

        .l2-tags {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          padding: 12px 22px;
          background: var(--paper-2);
          min-height: 44px;
        }
        .l2-tag {
          font-size: 10.5px;
          padding: 5px 10px;
          border: 1.5px solid var(--ink);
          border-radius: 999px;
          background: var(--paper);
          display: inline-flex;
          gap: 6px;
        }
        .l2-tag b { color: var(--red); }

        .l2-footer {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 10px;
          padding: 14px 22px 18px;
          background: var(--paper-2);
        }
        .l2-btn {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.08em;
          border: 2px solid var(--ink);
          background: var(--yellow);
          padding: 9px 16px;
          border-radius: 999px;
          display: inline-flex;
          align-items: center;
          gap: 7px;
          cursor: pointer;
        }
        .l2-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .l2-btn--ghost { background: var(--paper); }
        .l2-btn:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }
      `}</style>

      <div className="l2-ticker" aria-hidden="true">
        <span className="l2-ticker-track">
          43-DIMENSIONAL FEATURE VECTOR&nbsp;&nbsp;*&nbsp;&nbsp;XGBOOST ENSEMBLE, 100 TREES&nbsp;&nbsp;*&nbsp;&nbsp;N-GRAM GRAMMAR ANALYSIS&nbsp;&nbsp;*&nbsp;&nbsp;SUB-10MS VERDICT&nbsp;&nbsp;*&nbsp;&nbsp;43-DIMENSIONAL FEATURE VECTOR&nbsp;&nbsp;*&nbsp;&nbsp;XGBOOST ENSEMBLE, 100 TREES&nbsp;&nbsp;*&nbsp;&nbsp;
        </span>
      </div>

      <div className="l2-header">
        <div>
          <Eyebrow>LAYER 02 — STATISTICAL SCAN</Eyebrow>
          <h2 className="l2-title-serif">
            Doesn't read the words.<br /><em>Reads</em> the accent.
          </h2>
          <p className="l2-sub">
            A clean model's opcodes are almost entirely numbers. An exploit has a grammar
            of its own — load a callable, then call it, then hide it. Layer 2 measures
            that accent across 43 statistical features and an ensemble of 100 trees.
          </p>
        </div>
        <div className="l2-meta">CASE FILE<br />model.pt</div>
      </div>

      <div className="l2-stage">
        <div className="l2-three-mount" ref={mountRef} />

        <div className="l2-panel">
          <div className="l2-panel-label">BIGRAM EVIDENCE BOARD</div>
          {bigrams.length === 0 && (
            <p style={{ fontSize: 12, color: "var(--ink-45)" }}>
              Run a sample to surface its highest-risk opcode transitions.
            </p>
          )}
          {bigrams.map((row) => (
            <BigramRow key={row.id} row={row} />
          ))}
        </div>

        <div className="l2-panel">
          <div className="l2-panel-label">JURY OF 100 TREES</div>
          <div className="l2-jury-grid">
            {juryVotes.map((guilty, i) => (
              <TreeVote key={i} active={juryPhase} guilty={guilty} delay={i * 9} />
            ))}
          </div>
          <div className="l2-jury-tally">
            <b>{juryDisplay.voted}</b>/100 VOTED &nbsp;·&nbsp;
            <span className="l2-tally-red">{juryDisplay.guilty} GUILTY</span> &nbsp;·&nbsp;
            <span className="l2-tally-blue">{juryDisplay.clear} CLEAR</span>
          </div>
        </div>
      </div>

      <div className="l2-strip">
        {groups.map((g) => (
          <div className="l2-strip-group" key={g}>
            <span className="l2-strip-caption">{groupMeta[g]}</span>
            <div className="l2-strip-bars">
              {bars
                .filter((b) => b.group === g)
                .map((b) => (
                  <span
                    key={b.i}
                    className={`l2-bar l2-bar--${b.tone}`}
                    style={{
                      height: barsPhase ? `${b.value * 100}%` : "0%",
                      transitionDelay: `${b.i * 4}ms`,
                    }}
                  />
                ))}
            </div>
          </div>
        ))}
      </div>

      <div className="l2-results">
        <div className="l2-metrics">
          <div>
            LATENCY
            <b>{latency.toFixed(1)}ms</b>
          </div>
          <div>
            VERDICT
            <b>
              {verdict === "malicious" && (
                <span className="l2-verdict-tag l2-verdict-tag--malicious">
                  <ShieldAlert size={14} /> MALICIOUS
                </span>
              )}
              {verdict === "cleared" && (
                <span className="l2-verdict-tag l2-verdict-tag--cleared">
                  <ShieldCheck size={14} /> CLEARED
                </span>
              )}
              {!verdict && "—"}
            </b>
          </div>
        </div>

        <div className="l2-gauge-wrap">
          <Gauge />
          <div className="l2-needle" style={{ transform: `translateX(-50%) rotate(${angleForP(needleP)}deg)` }} />
          <div className="l2-needle-pivot" />
          <div className="l2-gauge-readout">
            <span className="l2-gauge-p">{probability === null ? "—" : probability.toFixed(3)}</span>
            <span className="l2-gauge-p-label">P(MALICIOUS)</span>
          </div>
        </div>
      </div>

      <div className="l2-tags">
        {tags.map((t) => (
          <span className="l2-tag" key={t.label}>
            {t.label} <b>{t.value}</b>
          </span>
        ))}
      </div>

      <div className="l2-footer">
        <button className="l2-btn" disabled={running} onClick={() => run("malicious")}>
          <Play size={13} /> RUN MALICIOUS SAMPLE
        </button>
        <button className="l2-btn l2-btn--ghost" disabled={running} onClick={() => run("clean")}>
          <Play size={13} /> RUN CLEAN SAMPLE
        </button>
        <button className="l2-btn l2-btn--ghost" onClick={reset} aria-label="Reset">
          <RotateCcw size={13} />
        </button>
      </div>
    </div>
  );
}
