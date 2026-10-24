import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, ShieldAlert } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const NAV_ITEMS = [
    { path: '/scanner', label: 'SCANNER' },
    { path: '/disarm', label: 'DISARM' },
    { path: '/benchmarks', label: 'BENCHMARKS' },
    { path: '/docs', label: 'DOCUMENTATION' },
];

const Header = () => {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const location = useLocation();

    const isActive = (path) => {
        return location.pathname.startsWith(path);
    };

    return (
        <header className="sticky top-0 z-50 border-b-2 border-ink bg-[#F2ECDC]/95 backdrop-blur-sm">
            <div className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-8">
                <div className="flex h-20 items-center justify-between gap-6">
                    <Link to="/" className="group flex items-center gap-3">
                        <div className="flex h-11 w-11 items-center justify-center rounded-sm border-2 border-ink bg-[#17130E] text-[#F2ECDC] transition-transform duration-200 group-hover:-rotate-6">
                            <span className="font-mono text-base font-extrabold leading-none tracking-tighter">MR</span>
                        </div>
                        <div className="leading-tight">
                            <span className="block font-serif text-lg font-black tracking-tight text-[#17130E]">MIRAGE</span>
                            <span className="block font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-[#17130E]/70">
                                4-LAYER ML THREAT DEFENSE
                            </span>
                        </div>
                    </Link>

                    <nav className="hidden items-center gap-1 lg:flex">
                        {NAV_ITEMS.map((item) => (
                            <Link
                                key={item.path}
                                to={item.path}
                                className={`relative rounded-full px-4 py-2 font-mono text-xs font-bold uppercase tracking-[0.16em] transition-colors ${isActive(item.path)
                                        ? 'bg-[#17130E] text-[#F2ECDC]'
                                        : 'text-[#17130E] hover:bg-[#17130E]/10'
                                    }`}
                            >
                                {item.label}
                            </Link>
                        ))}
                    </nav>

                    <div className="hidden lg:block">
                        <Link
                            to="/scanner"
                            className="inline-flex items-center gap-2 rounded-full border-2 border-ink bg-[#EFC23A] px-5 py-2 font-mono text-xs font-bold uppercase tracking-[0.18em] text-[#17130E] transition-transform hover:-translate-y-0.5 hover:shadow-[2px_2px_0px_#17130E]"
                        >
                            <ShieldAlert className="h-3.5 w-3.5 text-[#C4402E]" />
                            SCAN MODEL
                        </Link>
                    </div>

                    <button
                        onClick={() => setMobileMenuOpen((value) => !value)}
                        className="rounded-full border-2 border-ink p-2 text-[#17130E] transition hover:bg-[#17130E] hover:text-[#F2ECDC] lg:hidden"
                        aria-label="Toggle menu"
                        aria-expanded={mobileMenuOpen}
                        aria-controls="mobile-navigation"
                    >
                        {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
                    </button>
                </div>
            </div>

            <AnimatePresence>
                {mobileMenuOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.2 }}
                        className="border-t-2 border-ink bg-[#F2ECDC] lg:hidden"
                    >
                        <nav id="mobile-navigation" className="space-y-2 px-4 py-6">
                            {NAV_ITEMS.map((item) => (
                                <Link
                                    key={item.path}
                                    to={item.path}
                                    onClick={() => setMobileMenuOpen(false)}
                                    className={`block rounded-md px-3 py-2.5 font-mono text-xs font-bold uppercase tracking-[0.16em] ${isActive(item.path) ? 'bg-[#17130E] text-[#F2ECDC]' : 'text-[#17130E] hover:bg-[#17130E]/10'
                                        }`}
                                >
                                    {item.label}
                                </Link>
                            ))}
                            <Link
                                to="/scanner"
                                onClick={() => setMobileMenuOpen(false)}
                                className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-full border-2 border-ink bg-[#EFC23A] px-5 py-3 font-mono text-xs font-bold uppercase tracking-[0.18em] text-[#17130E]"
                            >
                                <ShieldAlert className="h-3.5 w-3.5 text-[#C4402E]" />
                                SCAN MODEL
                            </Link>
                        </nav>
                    </motion.div>
                )}
            </AnimatePresence>
        </header>
    );
};

export default Header;
