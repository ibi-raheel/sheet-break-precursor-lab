import { useState } from 'react'
import { C } from '../lib/colors'
import { r, f3, pv } from '../lib/data'
import { useWidth } from '../lib/hooks'

// Step 3: the trained sensor model and the hybrid next to the history-only
// hazard, on identical forward-chaining folds (hybrid.horizons in the JSON).
type Sub = 'first' | 'follow' | 'all'
const SUBS: { id: Sub; label: string; n: (h: Hz) => number }[] = [
  { id: 'first', label: 'First breaks', n: (h) => h.test_first_breaks },
  { id: 'follow', label: 'Follow-on breaks', n: (h) => h.test_follow_breaks },
  { id: 'all', label: 'All breaks', n: (h) => h.test_breaks },
]
const MODELS = [
  { id: 'hawkes', name: 'History only', sub: 'break hazard, 3 parameters', col: C.hazard },
  { id: 'sensor_gbm', name: 'AI sensor model', sub: 'gradient boosting, 531 features', col: C.sensor },
  { id: 'hybrid_gbm', name: 'Hybrid', sub: 'AI boosted from the hazard', col: C.hybrid },
] as const
type Hz = (typeof r.hybrid.horizons)['20']
const HZ = ['20', '60'] as const

export default function ModelPanel() {
  const [sub, setSub] = useState<Sub>('first')
  const [hi, setHi] = useState(0)
  const [ref, w] = useWidth<HTMLDivElement>(820)
  const h = r.hybrid.horizons[HZ[hi]] as Hz
  const ap = h.ap as Record<string, Record<string, number>>
  const chance = h.ap_chance.models as Record<string, { first: { obs: number; null_mean: number; null_p95: number; p: number } }>
  const prev = ap.hawkes[`prev_${sub}`]
  const narrow = w < 560
  const labelW = narrow ? 12 : 210
  const padR = 64
  const plotW = Math.max(120, w - labelW - padR)
  let vmax = prev
  MODELS.forEach((m) => { vmax = Math.max(vmax, ap[m.id][sub]); if (sub === 'first') vmax = Math.max(vmax, chance[m.id].first.null_p95) })
  const xmax = Math.ceil((vmax * 1.12) / 0.02) * 0.02
  const X = (v: number) => labelW + (v / xmax) * plotW
  const rowH = narrow ? 66 : 54
  const top = 26
  const height = top + MODELS.length * rowH + 30
  // Value label: right of the dot when above base; left when there is room, else past the chance band.
  const valPos = (v: number, right: number) => {
    if (v >= prev) return { x: X(v) + 12, textAnchor: 'start' as const }
    if (X(v) - 12 - 110 > 0) return { x: X(v) - 12, textAnchor: 'end' as const }
    return { x: Math.max(X(prev), X(right)) + 10, textAnchor: 'start' as const }
  }
  const lc = r.hybrid.lead_check.models as Record<string, Record<string, { p: number }>>
  const verdict = (id: string, p: number) => {
    if (id === 'hawkes') return p < 0.05 ? 'above chance' : 'no skill'
    const p2 = hi === 0 ? lc[id]?.['1']?.p : undefined
    if (p < 0.05 && (p2 === undefined || p2 < 0.05)) return 'above chance'
    if (p < 0.05) return `borderline (p = ${pv(p2!)} in a second null run)`
    return 'not above chance'
  }
  const ticks: number[] = []
  const step = xmax > 0.3 ? 0.1 : xmax > 0.12 ? 0.04 : 0.02
  for (let t = 0; t <= xmax + 1e-9; t += step) ticks.push(+t.toFixed(3))

  return (
    <div className="panel" ref={ref}>
      <div className="controls">
        <div className="seg" role="radiogroup" aria-label="Which breaks to score">
          {SUBS.map((s) => (
            <button key={s.id} role="radio" aria-checked={sub === s.id} className={sub === s.id ? 'on' : ''} onClick={() => setSub(s.id)}>
              {s.label} <span className="mono n">{s.n(h)}</span>
            </button>
          ))}
        </div>
        <label className="slider">
          <span>Look-ahead</span>
          <input type="range" min={0} max={1} step={1} value={hi} onChange={(e) => setHi(+e.target.value)} aria-valuetext={`${HZ[hi]} minutes`} />
          <span className="mono">{HZ[hi]} min</span>
        </label>
      </div>
      <svg width={w} height={height} role="img" aria-label={`PR-AUC on ${sub} breaks at a ${HZ[hi]}-minute look-ahead, three models against the base rate`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={top - 6} y2={height - 24} stroke={C.gridMinor} />
            <text x={X(t)} y={height - 8} textAnchor="middle" className="ax">{t.toFixed(2)}</text>
          </g>
        ))}
        <line x1={X(prev)} x2={X(prev)} y1={top - 14} y2={height - 24} stroke={C.ink} strokeDasharray="3 3" />
        <text x={X(prev)} y={top - 17} textAnchor={X(prev) > w - 120 ? 'end' : 'middle'} className="ax strong">base rate {f3(prev)}</text>
        {MODELS.map((m, i) => {
          const v = ap[m.id][sub]
          const y = top + i * rowH + (narrow ? 34 : rowH / 2)
          const c = chance[m.id].first
          const above = v > prev
          return (
            <g key={m.id}>
              {narrow ? (
                <text x={0} y={y - 18} className="rowlab"><tspan fontWeight={600}>{m.name}</tspan><tspan className="rowsub" dx={6}>{m.sub}</tspan></text>
              ) : (
                <>
                  <text x={0} y={y - 2} className="rowlab" fontWeight={600}>{m.name}</text>
                  <text x={0} y={y + 14} className="rowsub">{m.sub}</text>
                </>
              )}
              {sub === 'first' && (
                <g aria-hidden="true">
                  <rect x={X(c.null_mean)} y={y - 7} width={Math.max(2, X(c.null_p95) - X(c.null_mean))} height={14} fill={C.mask} />
                  <line x1={X(c.null_mean)} x2={X(c.null_mean)} y1={y - 9} y2={y + 9} stroke={C.muted} strokeWidth={1.5} />
                </g>
              )}
              <line x1={X(prev)} x2={X(v)} y1={y} y2={y} stroke={m.col} strokeWidth={2} opacity={0.55} />
              <circle cx={X(v)} cy={y} r={6} fill={m.col} stroke={C.paper} strokeWidth={2} />
              <text {...valPos(v, sub === 'first' ? c.null_p95 : prev)} y={y + 4} className="val">
                {f3(v)}{' '}
                <tspan className="rowsub">{above ? `${(v / prev).toFixed(1)}×` : 'below base'}</tspan>
              </text>
            </g>
          )
        })}
      </svg>
      <div className="readout mono" aria-live="polite">
        {sub === 'first' ? (
          <>
            <span><i className="lg lg-null" />chance band: score rotated in time, 1,000 draws (mean to 95th pct)</span>
            {MODELS.map((m) => {
              const c = chance[m.id].first
              return <span key={m.id} style={{ color: C.ink }}><i className="dot" style={{ background: m.col }} />{m.name}: p = {pv(c.p)}, {verdict(m.id, c.p)}</span>
            })}
          </>
        ) : (
          <span>Chance null computed for first breaks and quiet rows only; see details for CIs.</span>
        )}
      </div>
    </div>
  )
}
