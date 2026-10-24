import React from 'react';

const accentClass = {
    sun: 'accent-sun',
    electric: 'accent-electric',
    magenta: 'accent-magenta',
    mint: 'accent-mint',
    ink: 'bg-ink',
};

const tiltClass = {
    left: 'rotate-tilt-left',
    right: 'rotate-tilt-right',
    none: '',
};

const StickerTag = ({ children, accent = 'sun', tilt = 'left', className = '' }) => {
    return (
        <span
            className={`inline-flex items-center gap-2 rounded-full border-2 border-ink px-4 py-1.5 font-mono text-[11px] font-semibold uppercase tracking-[0.22em] shadow-ink-sm ${accentClass[accent] || accentClass.sun} ${tiltClass[tilt] || ''} ${className}`}
        >
            {children}
        </span>
    );
};

export default StickerTag;