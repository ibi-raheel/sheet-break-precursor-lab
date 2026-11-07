import { useMemo, useState } from 'react'
import { C } from '../lib/colors'
import { onsetMin, isFirst, episodes, hazard, hazMax, minutes, N, clock, rowAt } from '../lib/data'
import { useWidth } from '../lib/hooks'

const DAYS = 28
const DAY = 1440

export default function DaySheets() {
  const [ref, w] = useWidth<HTMLDivElement>(860)
  const narrow = w < 560
  const left = narrow ? 34 : 54
  const rowH = narrow ? 15 : 17
  const top = 22
  const plotW = Math.max(100, w - left - 4)
  const X = (m: number) => left + ((m % DAY) / DAY) * plotW
  const height = top + DAYS * rowH + 8
  const [tip, setTip] = useState<{ k: number; x: number; y: number } | null>(null)

  const hazPaths = useMemo(() => {
    const step = Math.max(2, Math.round(DAY / plotW) * 2) // minutes per sample
    const out: string[] = []
    for (let d = 0; d < DAYS; d++) {
      const y0 = top + d * rowH + rowH - 1
      let p = `M ${left} ${y0}`
      for (let m = 0; m <= DAY; m += step) {
        const i = rowAt(d * DAY + m)
        const v = i < N ? hazard[i] : 0
        const x = left + (m / DAY) * plotW
        p += ` L ${x.toFixed(1)} ${(y0 - (v / hazMax) * (rowH - 3)).toFixed(1)}`
      }
      p += ` L ${left + plotW} ${y0} Z`
      out.push(p)
    }
    return out
  }, [plotW, rowH, left])

  const bands = useMemo(() => {
    const b: { x: number; y: number; w: number }[] = []
    episodes.filter((e) => e.breaks.length > 1).forEach((e) => {
      let s = e.start
      while (s < e.end) {
        const d = Math.floor(s / DAY)
        const segEnd = Math.min(e.end, (d + 1) * DAY - 0.01)
        b.push({ x: X(s), y: top + d * rowH + rowH - 3, w: Math.max(2, X(segEnd) - X(s)) })
        s = (d + 1) * DAY
      }
    })
    return b
  }, [plotW, rowH, left])

  const hours = narrow ? [0, 6, 12, 18, 24] : [0, 3, 6, 9, 12, 15, 18, 21, 24]
  return (
    <div className="daysheets" ref={ref}>
      <svg width={w} height={height} role="img" aria-label="Twenty-eight daily chart sheets stacked, one row per day, with every break marked; breaks bunch into episodes and the hazard rises after each break">
        {hours.map((h) => {
          const x = left + (h / 24) * plotW
          return (
            <g key={h}>
              <line x1={x} x2={x} y1={top - 4} y2={top + DAYS * rowH} stroke={h % 12 === 0 ? C.gridMajor : C.gridMinor} />
              <text x={x} y={top - 9} textAnchor={h === 0 ? 'start' : h === 24 ? 'end' : 'middle'} className="ax">{String(h % 24).padStart(2, '0')}:00</text>
            </g>
          )
        })}
        {Array.from({ length: DAYS }, (_, d) => (
          <g key={d}>
            <line x1={left} x2={left + plotW} y1={top + (d + 1) * rowH - 0.5} y2={top + (d + 1) * rowH - 0.5} stroke={C.gridMinor} />
            {(!narrow || d % 2 === 0) && <text x={left - 8} y={top + d * rowH + rowH - 4} textAnchor="end" className="ax">{narrow ? d + 1 : `${d + 1} May`}</text>}
            <path d={hazPaths[d]} fill={C.hazard} fillOpacity={0.3} stroke="none" />
          </g>
        ))}
        {bands.map((b, i) => <rect key={i} x={b.x} y={b.y} width={b.w} height={2} fill={C.ink} opacity={0.75} />)}
        {onsetMin.map((m, k) => {
          const d = Math.floor(m / DAY); if (d >= DAYS) return null
          const x = X(m), y = top + d * rowH
          const first = isFirst[k]
          return (
            <g key={k}>
              <line x1={x} x2={x} y1={first ? y + 1 : y + 5} y2={y + rowH - 1} stroke={C.brk} strokeWidth={first ? 1.8 : 1.3} />
              <rect
                x={x - 5} y={y} width={10} height={rowH} fill="transparent"
                onPointerEnter={() => setTip({ k, x, y })} onPointerLeave={() => setTip(null)}
              />
            </g>
          )
        })}
        {tip && <line x1={tip.x} x2={tip.x} y1={top - 4} y2={top + DAYS * rowH} stroke={C.ink} strokeOpacity={0.35} pointerEvents="none" />}
      </svg>
      {tip && (
        <div className="tip mono" style={{ left: Math.min(tip.x + 10, w - 200), top: tip.y + rowH + 4 }}>
          <div>{clock(onsetMin[tip.k])}</div>
          <div>{isFirst[tip.k] ? 'first break of an episode' : `follow-on, +${onsetMin[tip.k] - onsetMin[tip.k - 1]} min`}</div>
        </div>
      )}
      <ul className="legend mono">
        <li><i className="lg lg-brk" />first break</li>
        <li><i className="lg lg-brk short" />follow-on (&lt; 2 h after the last)</li>
        <li><i className="lg lg-ep" />episode</li>
        <li><i className="lg lg-haz" />hazard from break history</li>
      </ul>
    </div>
  )
}

// silence unused warning for minutes (kept for parity with strip helpers)
void minutes
