import React, { useEffect, useRef, useState } from "react";

/**
 * HexScanPanel — signature animation for the empty hero block.
 *
 * Concept: a live grid of hex byte-pairs (what Layer 1's opcode scan
 * actually looks at), swept by a scanline. As the scanline crosses a row,
 * a few bytes flash "flagged" (blue outline, matches the header pill
 * color) before clearing — visually dramatizing "inspect the model
 * before it inspects you" instead of decorating the space with a
 * generic gradient/3D blob.
 */

const COLS = 12;
const ROWS = 16;
const HEX_CHARS = "0123456789ABCDEF";

function randomByte() {
  return (
    HEX_CHARS[Math.floor(Math.random() * 16)] +
    HEX_CHARS[Math.floor(Math.random() * 16)]
  );
}

function makeGrid() {
  return Array.from({ length: ROWS }, () =>
    Array.from({ length: COLS }, randomByte)
  );
}

export default function HexScanPanel() {
  const [grid, setGrid] = useState(makeGrid);
  const [flagged, setFlagged] = useState(() => new Set());
  const reducedMotion = useRef(
    typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
  );

  useEffect(() => {
    if (reducedMotion.current) {
      // Static, single frozen "finding" — no motion, still communicates the idea.
      setFlagged(new Set(["3-5", "7-2", "11-9"]));
      return;
    }

    // Re-roll a handful of bytes every tick, for a "live stream" feel.
    const churnId = setInterval(() => {
      setGrid((g) => {
        const next = g.map((row) => row.slice());
        for (let i = 0; i < 6; i++) {
          const r = Math.floor(Math.random() * ROWS);
          const c = Math.floor(Math.random() * COLS);
          next[r][c] = randomByte();
        }
        return next;
      });
    }, 400);

    // Scanline sweep drives which row gets flagged bytes, timed to the
    // CSS animation duration (4.8s) via row = elapsed / (duration/ROWS).
    const DURATION_MS = 4800;
    const start = performance.now();
    let rafId;
    const tick = (now) => {
      const elapsed = (now - start) % DURATION_MS;
      const row = Math.floor((elapsed / DURATION_MS) * ROWS);
      if (Math.random() < 0.35) {
        const col = Math.floor(Math.random() * COLS);
        const key = `${row}-${col}`;
        setFlagged((prev) => {
          const next = new Set(prev);
          next.add(key);
          return next;
        });
        setTimeout(() => {
          setFlagged((prev) => {
            const next = new Set(prev);
            next.delete(key);
            return next;
          });
        }, 900);
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      clearInterval(churnId);
      cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <div className="hex-scan-panel">
      <div className="hex-scan-grid">
        {grid.map((row, r) => (
          <div className="hex-scan-row" key={r}>
            {row.map((byte, c) => {
              const key = `${r}-${c}`;
              return (
                <span
                  key={key}
                  className={flagged.has(key) ? "byte flagged" : "byte"}
                >
                  {byte}
                </span>
              );
            })}
          </div>
        ))}
      </div>

      {!reducedMotion.current && <div className="hex-scan-line" />}

      <div className="hex-scan-label">CHECKPOINT.SAFETENSORS — LIVE READ</div>

      <style>{`
        .hex-scan-panel {
          --hsp-paper: var(--paper, #F2ECDC);
          --hsp-ink: #1a1712;
          --hsp-blue: #3548E0;
          --hsp-yellow: var(--yellow, #E8BA4C);
          position: relative;
          height: 100%;
          width: 100%;
          overflow: hidden;
          background: var(--hsp-yellow);
          font-family: "JetBrains Mono", ui-monospace, monospace;
        }

        .hex-scan-grid {
          display: flex;
          flex-direction: column;
          height: 100%;
          width: 100%;
          padding: 24px;
          box-sizing: border-box;
          justify-content: space-between;
        }

        .hex-scan-row {
          display: flex;
          justify-content: space-between;
        }

        .byte {
          color: var(--hsp-ink);
          opacity: 0.32;
          font-size: 13px;
          letter-spacing: 0.02em;
          transition: opacity 0.15s ease, color 0.15s ease;
        }

        .byte.flagged {
          opacity: 1;
          color: var(--hsp-blue);
          background: var(--hsp-paper);
          outline: 1.5px solid var(--hsp-blue);
          border-radius: 2px;
          padding: 0 2px;
        }

        .hex-scan-line {
          position: absolute;
          left: 0;
          right: 0;
          height: 2px;
          background: var(--hsp-ink);
          box-shadow: 0 0 12px 2px rgba(26, 23, 18, 0.35);
          animation: hsp-sweep 4.8s linear infinite;
        }

        @keyframes hsp-sweep {
          0%   { top: 0%; }
          100% { top: 100%; }
        }

        .hex-scan-label {
          position: absolute;
          left: 24px;
          bottom: 14px;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 0.08em;
          color: var(--hsp-ink);
          opacity: 0.6;
        }

        @media (prefers-reduced-motion: reduce) {
          .hex-scan-line { display: none; }
        }
      `}</style>
    </div>
  );
}
