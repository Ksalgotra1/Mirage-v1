import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, useReducedMotion } from 'framer-motion';
import Header from '@/components/Header.jsx';
import Footer from '@/components/Footer.jsx';

const MarqueeStrip = () => {
  const items = [
    "LAYER 1: ZERO-HOST OPCODE DISASSEMBLY",
    "LAYER 2: 43-D XGBOOST N-GRAM CLASSIFIER",
    "LAYER 3A: ADAPTIVE MULTI-RATE MANTISSA STEGANALYSIS · 1–16 BITS",
    "LAYER 3B: PRUNING-SAFE SPECTRAL BACKDOOR DETECTOR",
    "5-SIGNAL FORENSIC BREAKDOWN · TVD · FFT · PARITY COMPENSATION",
    "AUTOMATED FOUR-PHASE MATHEMATICAL DISARMAMENT",
    "SUB-50MS MULTI-LAYER LATENCY"
  ];
  const duplicated = [...items, ...items];
  return (
    <div className="border-b-2 border-ink bg-[#17130E] overflow-hidden select-none">
      <div className="marquee-track flex w-max gap-10 whitespace-nowrap py-2 font-mono text-[10.5px] uppercase tracking-[0.2em] text-[#F2ECDC]">
        {duplicated.map((item, idx) => (
          <span key={`${item}-${idx}`} className="flex items-center gap-10">
            <span>{item}</span>
            <span className="text-[#EFC23A] font-bold" aria-hidden="true">✦</span>
          </span>
        ))}
      </div>
    </div>
  );
};

const SiteLayout = () => {
  const location = useLocation();
  const reduceMotion = useReducedMotion();
  return (
    <div className="min-h-screen bg-[#F2ECDC] text-[#17130E]">
      <a href="#main-content" className="skip-link">Skip to content</a>
      <MarqueeStrip />
      <Header />
      <motion.main key={location.pathname} id="main-content" initial={{ opacity: 0, y: reduceMotion ? 0 : 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: reduceMotion ? 0 : 0.28 }} className="mx-auto max-w-[1400px] px-4 py-8 sm:px-6 lg:px-8">
        <Outlet />
      </motion.main>
      <Footer />
    </div>
  );
};

export default SiteLayout;
