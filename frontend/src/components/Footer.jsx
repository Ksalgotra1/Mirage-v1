import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheck, Cpu, Terminal, Github, ArrowUpRight } from 'lucide-react';

const Footer = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="border-t-2 border-ink bg-[#17130E] text-[#F2ECDC]">
            <div className="mx-auto max-w-[1400px] px-4 py-12 sm:px-6 lg:px-8">
                <div className="grid gap-10 md:grid-cols-12">
                    <div className="md:col-span-5">
                        <div className="flex items-center gap-3">
                            <div className="flex h-11 w-11 items-center justify-center rounded-sm border-2 border-[#F2ECDC] bg-[#EFC23A] text-[#17130E]">
                                <span className="font-mono text-lg font-extrabold leading-none">MR</span>
                            </div>
                            <div>
                                <p className="font-serif text-2xl font-black tracking-tight text-[#F2ECDC]">MIRAGE</p>
                                <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#F2ECDC]/70">
                                    ADVANCED ML BACKDOOR & MALWARE DEFENSE
                                </p>
                            </div>
                        </div>

                        <p className="mt-5 font-serif text-lg leading-relaxed text-[#F2ECDC]/90 md:max-w-md">
                            4-Layer Deep Inspection & Automated Disarmament across PyTorch (.pt), SafeTensors (.safetensors), Pickles (.pkl), and HDF5 (.h5).
                        </p>

                        <div className="mt-6 flex items-center gap-3">
                            <span className="inline-flex items-center gap-1.5 rounded-full border border-[#2F4FE0] bg-[#2F4FE0]/20 px-3 py-1 font-mono text-[10px] font-bold text-[#8ba2ff]">
                                <span className="h-1.5 w-1.5 rounded-full bg-[#2F4FE0] animate-pulse"></span>
                                FASTAPI BACKEND :8000
                            </span>
                            <span className="inline-flex items-center gap-1.5 rounded-full border border-[#EFC23A] bg-[#EFC23A]/20 px-3 py-1 font-mono text-[10px] font-bold text-[#EFC23A]">
                                4 DEFENSE LAYERS ARMED
                            </span>
                        </div>
                    </div>

                    <div className="md:col-span-3">
                        <p className="font-mono text-[11px] font-bold uppercase tracking-[0.2em] text-[#EFC23A]">NAVIGATION</p>
                        <ul className="mt-4 space-y-2.5 font-mono text-xs tracking-wider">
                            <li>
                                <Link to="/" className="text-[#F2ECDC]/80 transition hover:text-[#EFC23A]">
                                    SCANNER & DISARM
                                </Link>
                            </li>
                            <li>
                                <Link to="/layers" className="text-[#F2ECDC]/80 transition hover:text-[#EFC23A]">
                                    LAYER SANDBOX (L1 - L3B)
                                </Link>
                            </li>
                            <li>
                                <Link to="/benchmarks" className="text-[#F2ECDC]/80 transition hover:text-[#EFC23A]">
                                    EVALUATION EVIDENCE
                                </Link>
                            </li>
                            <li>
                                <Link to="/disarm" className="text-[#F2ECDC]/80 transition hover:text-[#EFC23A]">
                                    DISARMAMENT CHAMBER
                                </Link>
                            </li>
                            <li>
                                <Link to="/docs" className="text-[#F2ECDC]/80 transition hover:text-[#EFC23A]">
                                    MATHEMATICAL SPECS
                                </Link>
                            </li>
                        </ul>
                    </div>

                    <div className="md:col-span-4">
                        <p className="font-mono text-[11px] font-bold uppercase tracking-[0.2em] text-[#EFC23A]">LAYER ARCHITECTURE</p>
                        <ul className="mt-4 space-y-2 font-mono text-[11px] text-[#F2ECDC]/75">
                            <li className="flex items-center justify-between border-b border-[#F2ECDC]/10 pb-1.5">
                                <span>L1: STATIC OPCODE SCAN</span>
                                <span className="text-[#8ba2ff]">ZERO-DESERIALIZE</span>
                            </li>
                            <li className="flex items-center justify-between border-b border-[#F2ECDC]/10 pb-1.5">
                                <span>L2: STATISTICAL XGBOOST</span>
                                <span className="text-[#8ba2ff]">43-D N-GRAMS</span>
                            </li>
                            <li className="flex items-center justify-between border-b border-[#F2ECDC]/10 pb-1.5">
                                <span>L3A: GALOIS FIELD STEGO</span>
                                <span className="text-[#8ba2ff]">512×512 GF + FORENSICS</span>
                            </li>
                            <li className="flex items-center justify-between pb-1.5">
                                <span>L3B: SPECTRAL INVARIANTS</span>
                                <span className="text-[#8ba2ff]">SVD + L2 + KURTOSIS</span>
                            </li>
                        </ul>
                    </div>
                </div>

                <div className="mt-12 flex flex-col items-start justify-between gap-4 border-t border-[#F2ECDC]/15 pt-6 font-mono text-[10.5px] uppercase tracking-[0.16em] text-[#F2ECDC]/60 md:flex-row md:items-center">
                    <p>{`(c) ${currentYear} MIRAGE DEFENSE SYSTEM. ALL RIGHTS RESERVED.`}</p>
                    <p className="flex items-center gap-2">
                        <span>MODEL VULNERABILITY MITIGATION SUITE</span>
                    </p>
                </div>
            </div>
        </footer>
    );
};

export default Footer;