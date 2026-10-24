import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { ShieldAlert, ShieldCheck, Play, RotateCcw } from "lucide-react";

/**
 * LAYER 1 — STATIC OPCODE SCAN
 * -----------------------------------------------------------------------
 * Design direction: "Field Notes" editorial / case-file dossier.
 * Cream paper, warm near-black ink, three flat accent colors, a fat
 * high-contrast serif for voice, JetBrains Mono for every technical
 * surface. No gradients, no glass, no glow — the one bold move is the
 * physical rubber-stamp verdict that slams onto the case file.
 *
 * Three.js is scoped to exactly two jobs: a faint animated paper-grain
 * field, and the 3D stamp-slam (perspective drop + shadow + camera
 * jitter). Everything readable — type, data, labels — stays in the DOM
 * so it renders crisp at any size.
 */

// ---------------------------------------------------------------------
// Opcode scripts for the two demo scenarios described in the Layer 1
// spec: a malicious pickle resolving os.system, and a clean model file
// that never resolves a callable.
// ---------------------------------------------------------------------
const MALICIOUS_OPS = [
  { id: "m1", code: "PROTO", arg: "4", kind: "meta" },
  { id: "m2", code: "SHORT_BINUNICODE", arg: "'os'", kind: "push", toStack: true },
  { id: "m3", code: "SHORT_BINUNICODE", arg: "'system'", kind: "push", toStack: true },
  { id: "m4", code: "STACK_GLOBAL", arg: "", kind: "resolve" },
  { id: "m5", code: "REDUCE", arg: "", kind: "execute" },
];

const CLEAN_OPS = [
  { id: "c1", code: "PROTO", arg: "4", kind: "meta" },
  { id: "c2", code: "BINFLOAT", arg: "0.0341", kind: "push" },
  { id: "c3", code: "EMPTY_TUPLE", arg: "", kind: "push" },
  { id: "c4", code: "MARK", arg: "", kind: "meta" },
  { id: "c5", code: "BUILD", arg: "__setstate__", kind: "benign" },
  { id: "c6", code: "STOP", arg: "", kind: "meta" },
];

const STEP_MS = 480;

// ---------------------------------------------------------------------
// Three.js layer: paper grain + rubber-stamp slam.
// Isolated in a hook so the main component stays declarative.
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

      // distressed double ring
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
      ctx.font = "800 88px 'JetBrains Mono', 'Courier New', monospace";
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

    // soft contact shadow beneath the stamp — a radial-gradient texture so
    // the edges fade out instead of cutting off as a hard rectangle
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
    let anim = null; // { start, duration, from:{}, to:{} }
    let shake = null; // { start, duration }

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
        const ease = 1 - Math.pow(1 - p, 3); // easeOutCubic
        const overshoot = p < 1 ? Math.sin(p * Math.PI) * 0.06 : 0;
        stamp.position.y = anim.fromY + (anim.toY - anim.fromY) * ease;
        stamp.position.z = anim.fromZ + (anim.toZ - anim.fromZ) * ease;
        const s = anim.fromScale + (anim.toScale - anim.fromScale) * ease + overshoot;
        stamp.scale.set(s, s, 1);
        stampMat.opacity = ease;
        shadow.material.opacity = ease * 0.28;
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
  return (
    <span className={`l1-pill l1-pill--${tone}`}>
      {children}
    </span>
  );
}

function OpRow({ op, active }) {
  return (
    <div className={`l1-oprow l1-oprow--${op.kind} ${active ? "is-active" : ""}`}>
      <span className="l1-opcode">{op.code}</span>
      {op.arg ? <span className="l1-oparg">{op.arg}</span> : null}
    </div>
  );
}

export default function Layer1ScanAnimation() {
  const mountRef = useRef(null);
  const stamp = useStampScene(mountRef);

  const [scenario, setScenario] = useState(null); // 'malicious' | 'clean' | null
  const [revealed, setRevealed] = useState([]);
  const [stackItems, setStackItems] = useState([]);
  const [merged, setMerged] = useState(false);
  const [verdict, setVerdict] = useState(null); // 'blocked' | 'cleared' | null
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
      setScenario(which);
      setRevealed([]);
      setStackItems([]);
      setMerged(false);
      setVerdict(null);
      setLatency(0);
      setRunning(true);

      const ops = which === "malicious" ? MALICIOUS_OPS : CLEAN_OPS;

      ops.forEach((op, i) => {
        const t = setTimeout(() => {
          setRevealed((prev) => [...prev, op]);

          if (op.toStack) {
            setStackItems((prev) => [...prev, op.arg]);
          }
          if (op.kind === "resolve") {
            setTimeout(() => setMerged(true), 180);
          }
          if (op.kind === "execute") {
            setTimeout(() => {
              setVerdict("blocked");
              stamp.current && stamp.current.fire("BLOCKED", "#C4402E");
              runLatencyCount(2.4);
              setRunning(false);
            }, 420);
          }
          if (op.code === "STOP" && which === "clean") {
            setTimeout(() => {
              setVerdict("cleared");
              stamp.current && stamp.current.fire("CLEARED", "#2F4FE0");
              runLatencyCount(3.1);
              setRunning(false);
            }, 260);
          }
        }, i * STEP_MS);
        timers.current.push(t);
      });
    },
    [clearTimers, runLatencyCount, stamp]
  );

  const reset = useCallback(() => {
    clearTimers();
    stamp.current && stamp.current.reset();
    setScenario(null);
    setRevealed([]);
    setStackItems([]);
    setMerged(false);
    setVerdict(null);
    setLatency(0);
    setRunning(false);
  }, [clearTimers, stamp]);

  useEffect(() => () => clearTimers(), [clearTimers]);

  const trayLabel = scenario === "clean" ? "SAFE PASS GATE" : "VIRTUAL STACK";

  return (
    <div className="l1-wrap">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,900;1,9..144,500&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

        .l1-wrap {
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
          max-width: 880px;
          margin: 0 auto;
          position: relative;
          overflow: hidden;
        }
        .l1-wrap * { box-sizing: border-box; }

        .l1-ticker {
          background: var(--ink);
          color: var(--paper);
          font-size: 11px;
          letter-spacing: 0.14em;
          padding: 7px 0;
          overflow: hidden;
          white-space: nowrap;
          border-bottom: 2px solid var(--ink);
        }
        .l1-ticker-track {
          display: inline-block;
          padding-left: 100%;
          animation: l1-scroll 18s linear infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .l1-ticker-track { animation: none; padding-left: 2%; }
        }
        @keyframes l1-scroll {
          from { transform: translateX(0); }
          to { transform: translateX(-100%); }
        }

        .l1-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          padding: 22px 24px 16px;
          border-bottom: 2px solid var(--ink);
        }
        .l1-title-serif {
          font-family: 'Fraunces', serif;
          font-weight: 900;
          font-size: 30px;
          line-height: 1.05;
          margin: 10px 0 4px;
          letter-spacing: -0.01em;
        }
        .l1-title-serif em {
          font-style: italic;
          font-weight: 500;
        }
        .l1-sub {
          font-size: 12px;
          color: var(--ink-70);
          max-width: 46ch;
          line-height: 1.5;
        }
        .l1-meta {
          text-align: right;
          font-size: 11px;
          color: var(--ink-45);
          letter-spacing: 0.06em;
          white-space: nowrap;
        }

        .l1-pill {
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
        .l1-pill--yellow { background: var(--yellow); }
        .l1-pill--blue { background: var(--blue); color: var(--paper); }
        .l1-pill--red { background: var(--red); color: var(--paper); }
        .l1-pill--paper { background: var(--paper); }

        .l1-stage {
          position: relative;
          display: grid;
          grid-template-columns: 1.3fr 1fr;
          gap: 0;
          min-height: 340px;
        }
        @media (max-width: 620px) {
          .l1-stage { grid-template-columns: 1fr; }
        }

        .l1-three-mount {
          position: absolute;
          inset: 0;
          z-index: 5;
          pointer-events: none;
        }

        .l1-panel {
          padding: 18px 22px 22px;
          position: relative;
          z-index: 2;
        }
        .l1-panel + .l1-panel {
          border-left: 2px solid var(--ink);
        }
        @media (max-width: 620px) {
          .l1-panel + .l1-panel { border-left: none; border-top: 2px solid var(--ink); }
        }

        .l1-panel-label {
          font-size: 10.5px;
          font-weight: 700;
          letter-spacing: 0.14em;
          color: var(--ink-45);
          margin-bottom: 12px;
        }

        .l1-transcript {
          display: flex;
          flex-direction: column;
          gap: 6px;
          min-height: 210px;
        }
        .l1-oprow {
          display: flex;
          align-items: baseline;
          gap: 10px;
          font-size: 12.5px;
          padding: 6px 10px;
          border: 1.5px solid var(--line);
          border-left-width: 4px;
          background: var(--paper);
          opacity: 0;
          transform: translateX(-8px);
          animation: l1-enter 0.35s ease forwards;
        }
        @keyframes l1-enter {
          to { opacity: 1; transform: translateX(0); }
        }
        .l1-oprow--meta { border-left-color: var(--ink-45); }
        .l1-oprow--push { border-left-color: var(--ink-70); }
        .l1-oprow--benign { border-left-color: var(--blue); }
        .l1-oprow--resolve { border-left-color: var(--yellow); background: #FBF2D6; }
        .l1-oprow--execute { border-left-color: var(--red); background: #F7DFDA; }
        .l1-opcode { font-weight: 700; letter-spacing: 0.03em; }
        .l1-oparg { color: var(--ink-70); }

        .l1-tray {
          border: 2px dashed var(--line);
          min-height: 140px;
          padding: 14px;
          display: flex;
          flex-direction: column-reverse;
          align-items: flex-start;
          gap: 8px;
          position: relative;
        }
        .l1-tray-item {
          font-size: 12.5px;
          font-weight: 600;
          padding: 6px 12px;
          background: var(--paper-2);
          border: 1.5px solid var(--ink);
          animation: l1-drop 0.32s ease forwards;
        }
        @keyframes l1-drop {
          from { opacity: 0; transform: translateY(-14px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .l1-tray-merged {
          font-size: 13px;
          font-weight: 700;
          padding: 8px 14px;
          background: var(--yellow);
          border: 2px solid var(--ink);
          animation: l1-merge 0.4s ease forwards;
        }
        @keyframes l1-merge {
          from { transform: scale(0.85); opacity: 0.4; }
          to { transform: scale(1); opacity: 1; }
        }

        .l1-gate {
          min-height: 140px;
          border: 2px dashed var(--line);
          display: flex;
          align-items: center;
          justify-content: center;
          flex-direction: column;
          gap: 8px;
          text-align: center;
          font-size: 11.5px;
          color: var(--ink-70);
        }

        .l1-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
          padding: 16px 22px;
          border-top: 2px solid var(--ink);
          background: var(--paper-2);
        }
        .l1-metrics {
          display: flex;
          gap: 22px;
          font-size: 11px;
          letter-spacing: 0.05em;
        }
        .l1-metrics b { display: block; font-size: 14px; letter-spacing: 0; margin-top: 2px; }
        .l1-verdict-tag {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-weight: 700;
        }
        .l1-verdict-tag--blocked { color: var(--red); }
        .l1-verdict-tag--cleared { color: var(--blue); }

        .l1-controls { display: flex; gap: 10px; }
        .l1-btn {
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
        .l1-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .l1-btn--ghost { background: var(--paper); }
        .l1-btn:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }
      `}</style>

      <div className="l1-ticker" aria-hidden="true">
        <span className="l1-ticker-track">
          NO DESERIALIZATION&nbsp;&nbsp;*&nbsp;&nbsp;ZERO HOST EXECUTION&nbsp;&nbsp;*&nbsp;&nbsp;SUB-4MS VERDICT&nbsp;&nbsp;*&nbsp;&nbsp;VIRTUAL STACK MACHINE&nbsp;&nbsp;*&nbsp;&nbsp;NO DESERIALIZATION&nbsp;&nbsp;*&nbsp;&nbsp;ZERO HOST EXECUTION&nbsp;&nbsp;*&nbsp;&nbsp;SUB-4MS VERDICT&nbsp;&nbsp;*&nbsp;&nbsp;
        </span>
      </div>

      <div className="l1-header">
        <div>
          <Eyebrow tone="yellow">LAYER 01 — STATIC SCAN</Eyebrow>
          <h2 className="l1-title-serif">
            Reads the bytecode.<br /><em>Never</em> runs it.
          </h2>
          <p className="l1-sub">
            Every opcode is disassembled from the raw byte stream and simulated on a
            virtual stack. Nothing is deserialized, so nothing on the host can execute.
          </p>
        </div>
        <div className="l1-meta">
          CASE FILE<br />model.pt
        </div>
      </div>

      <div className="l1-stage">
        <div className="l1-three-mount" ref={mountRef} />

        <div className="l1-panel">
          <div className="l1-panel-label">OPCODE TRANSCRIPT</div>
          <div className="l1-transcript">
            {revealed.length === 0 && (
              <p style={{ fontSize: 12, color: "var(--ink-45)" }}>
                Run a sample below to disassemble it opcode by opcode.
              </p>
            )}
            {revealed.map((op) => (
              <OpRow key={op.id} op={op} active={op.kind === "execute" || op.kind === "resolve"} />
            ))}
          </div>
        </div>

        <div className="l1-panel">
          <div className="l1-panel-label">{trayLabel}</div>

          {scenario !== "clean" ? (
            <div className="l1-tray">
              {!merged &&
                stackItems.map((item, i) => (
                  <span className="l1-tray-item" key={`${item}-${i}`}>
                    {item}
                  </span>
                ))}
              {merged && <span className="l1-tray-merged">os.system</span>}
              {scenario === null && (
                <span style={{ fontSize: 11.5, color: "var(--ink-45)" }}>
                  Pushed strings land here before a callable is resolved.
                </span>
              )}
            </div>
          ) : (
            <div className="l1-gate">
              {verdict === "cleared" ? (
                <ShieldCheck size={26} color="#2F4FE0" />
              ) : (
                <ShieldAlert size={26} color="var(--ink-45)" />
              )}
              <span>
                No GLOBAL or REDUCE opcode resolves a callable —<br />opcodes pass straight through.
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="l1-footer">
        <div className="l1-metrics">
          <div>
            LATENCY
            <b>{latency.toFixed(1)}ms</b>
          </div>
          <div>
            VERDICT
            <b>
              {verdict === "blocked" && (
                <span className="l1-verdict-tag l1-verdict-tag--blocked">
                  <ShieldAlert size={14} /> BLOCKED
                </span>
              )}
              {verdict === "cleared" && (
                <span className="l1-verdict-tag l1-verdict-tag--cleared">
                  <ShieldCheck size={14} /> CLEARED
                </span>
              )}
              {!verdict && "—"}
            </b>
          </div>
        </div>

        <div className="l1-controls">
          <button className="l1-btn" disabled={running} onClick={() => run("malicious")}>
            <Play size={13} /> RUN MALICIOUS SAMPLE
          </button>
          <button className="l1-btn l1-btn--ghost" disabled={running} onClick={() => run("clean")}>
            <Play size={13} /> RUN CLEAN SAMPLE
          </button>
          <button className="l1-btn l1-btn--ghost" onClick={reset} aria-label="Reset">
            <RotateCcw size={13} />
          </button>
        </div>
      </div>
    </div>
  );
}
