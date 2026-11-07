import { C } from '../lib/colors'

// Machine-direction schematic of a fourdrinier paper machine. Sensor positions
// are illustrative: the dataset's 59 signals are anonymised.
const topX = [540, 600, 660, 720, 780]
const botX = [570, 630, 690, 750]
const TY = 78, BY = 150, R = 24

function Drawing({ narrow, id }: { narrow: boolean; id: string }) {
  const lab = narrow ? 17 : 14
  // serpentine sheet path through the dryer cans
  let dry = `M 505 ${TY - R}`
  topX.forEach((x, i) => {
    dry += ` L ${x} ${TY - R} A ${R} ${R} 0 0 1 ${x + R * 0.7} ${TY + R * 0.7}`
    if (i < botX.length) dry += ` L ${botX[i] - R * 0.7} ${BY - R * 0.7} A ${R} ${R} 0 0 0 ${botX[i] + R * 0.7} ${BY - R * 0.7}`
  })
  return (
    <g fontFamily='"Barlow Condensed", "Arial Narrow", sans-serif' fontWeight={600} letterSpacing="0.06em">
      {/* frame line */}
      <line x1="0" y1="206" x2="1000" y2="206" stroke={C.gridMajor} />
      {/* headbox */}
      <path d="M 14 70 H 70 L 86 112 L 92 118 H 70 L 14 104 Z" fill="none" stroke={C.ink} strokeWidth="1.6" />
      <line x1="24" y1="84" x2="64" y2="84" stroke={C.ink2} strokeWidth="1" />
      <line x1="24" y1="94" x2="66" y2="94" stroke={C.ink2} strokeWidth="1" />
      {/* forming wire loop */}
      <path d="M 112 120 H 318 A 18 18 0 0 1 318 156 H 112 A 18 18 0 0 1 112 120 Z" fill="none" stroke={C.ink2} strokeWidth="1.2" strokeDasharray="1 3" />
      <circle cx="112" cy="138" r="14" fill={C.paper} stroke={C.ink} strokeWidth="1.6" />
      <circle cx="318" cy="138" r="14" fill={C.paper} stroke={C.ink} strokeWidth="1.6" />
      {[150, 180, 210, 240, 270].map((x) => <rect key={x} x={x - 9} y={121} width={18} height={4} fill={C.ink2} />)}
      {[175, 235, 290].map((x) => <circle key={x} cx={x} cy={163} r={5} fill="none" stroke={C.ink2} strokeWidth="1" />)}
      {/* press nips */}
      {[380, 448].map((x) => (
        <g key={x}>
          <circle cx={x} cy={98} r={20} fill={C.paper} stroke={C.ink} strokeWidth="1.6" />
          <circle cx={x} cy={140} r={20} fill={C.paper} stroke={C.ink} strokeWidth="1.6" />
          <circle cx={x} cy={98} r={3} fill={C.ink} /><circle cx={x} cy={140} r={3} fill={C.ink} />
        </g>
      ))}
      {/* dryer cans */}
      {topX.map((x) => <g key={'t' + x}><circle cx={x} cy={TY} r={R} fill={C.paper} stroke={C.ink} strokeWidth="1.6" /><circle cx={x} cy={TY} r={3} fill={C.ink} /></g>)}
      {botX.map((x) => <g key={'b' + x}><circle cx={x} cy={BY} r={R} fill={C.paper} stroke={C.ink} strokeWidth="1.6" /><circle cx={x} cy={BY} r={3} fill={C.ink} /></g>)}
      <rect x="512" y="36" width="300" height="152" fill="none" stroke={C.gridMajor} strokeDasharray="5 4" />
      {/* calender stack */}
      {[70, 100, 130].map((y) => <circle key={y} cx={862} cy={y + 4} r={14} fill={C.paper} stroke={C.ink} strokeWidth="1.6" />)}
      {/* reel */}
      <circle cx={950} cy={120} r={36} fill={C.paper} stroke={C.ink} strokeWidth="1.6" />
      <circle cx={950} cy={120} r={26} fill="none" stroke={C.ink2} strokeWidth="0.8" />
      <circle cx={950} cy={120} r={18} fill="none" stroke={C.ink2} strokeWidth="0.8" />
      <circle cx={950} cy={120} r={6} fill={C.ink} />

      {/* the sheet */}
      <g fill="none" stroke={C.trace} strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M 92 118 L 112 120 H 330 L 380 119 L 448 119 L 505 54" />
        <path d={dry} clipPath={`url(#bb-${id})`} />
        <path d={dry} clipPath={`url(#ab-${id})`} />
        <path d={`M ${780 + R * 0.7} ${TY + R * 0.7} L 848 104 L 876 90 L 862 74 L 940 86`} />
      </g>
      <defs>
        <clipPath id={`bb-${id}`}><rect x="500" y="0" width="142" height="230" /></clipPath>
        <clipPath id={`ab-${id}`}><rect x="664" y="0" width="300" height="230" /></clipPath>
      </defs>
      {/* break marker between cans */}
      <g stroke={C.brk} strokeWidth="2" fill="none" strokeLinecap="round">
        <path d="M 640 108 l 5 -6 l 3 7 l 4 -5" />
        <path d="M 656 116 l 4 -5 l 3 7 l 4 -6" />
      </g>
      <g>
        <line x1="652" y1="104" x2="652" y2="28" stroke={C.brk} strokeWidth="1" />
        <text x={657} y={30} fontSize={lab} fill={C.brk}>SHEET BREAK</text>
      </g>

      {/* sensor taps (illustrative) */}
      {[60, 220, 414, 540, 720, 862, 950].map((x, i) => (
        <g key={x} stroke={C.ink2} strokeWidth="1">
          <line x1={x} y1={i === 6 ? 84 : i === 5 ? 56 : i === 3 || i === 4 ? 54 : i === 2 ? 78 : 66} x2={x} y2={i === 3 || i === 4 ? 44 : 30} />
          <rect x={x - 5} y={i === 3 || i === 4 ? 36 : 22} width="10" height="8" fill={C.paper} />
        </g>
      ))}

      {/* section labels */}
      <g fontSize={lab} fill={C.ink}>
        <text x={50} y={196} textAnchor="middle">HEADBOX</text>
        <text x={215} y={196} textAnchor="middle">FORMING WIRE</text>
        <text x={414} y={196} textAnchor="middle">PRESS</text>
        <text x={662} y={204} textAnchor="middle">DRYER SECTION</text>
        <text x={862} y={196} textAnchor="middle">CALENDER</text>
        <text x={950} y={196} textAnchor="middle">REEL</text>
      </g>
    </g>
  )
}

export default function Schematic() {
  return (
    <figure className="schematic">
      <svg className="sch-wide" viewBox="0 0 1000 224" role="img" aria-label="Machine-direction schematic: headbox, forming wire, press, dryer section, calender and reel, with a sheet break in the dryer section">
        <Drawing narrow={false} id="w" />
      </svg>
      <div className="sch-narrow" role="img" aria-label="Machine-direction schematic in two halves: wet end then dry end">
        <svg viewBox="0 0 500 224"><Drawing narrow id="n1" /></svg>
        <svg viewBox="500 0 500 224"><Drawing narrow id="n2" /></svg>
      </div>
      <figcaption className="mono">
        <span>MACHINE DIRECTION</span>
        <svg width="48" height="10" viewBox="0 0 48 10" aria-hidden="true"><path d="M0 5h44M39 1l5 4-5 4" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
        <span className="cap-note">Sensor positions illustrative; the 59 signals are anonymised.</span>
      </figcaption>
    </figure>
  )
}
