import { useState } from 'react'
import { C } from '../lib/colors'
import { r, f2, f3 } from '../lib/data'
import { useWidth } from '../lib/hooks'

// Step 4: what the model leans on (mean |SHAP|) against what actually comes
// before breaks (best single-signal lift). One dot per sensor, 59 in all.
type P = (typeof r.scatter)[number]

export default function ShapScatter() {
  const [ref, w] = useWidth<HTMLDivElement>(820)
  const [hov, setHov] = useState<P | null>(null)
  const narrow = w < 560
  const pad = { l: narrow ? 34 : 46, r: 14, t: 18, b: 40 }
  const height = narrow ? 300 : 340
  const pts = r.scatter as P[]
  const sMax = Math.max(...pts.map((p) => p.shap)) * 1.06
  const lMax = Math.ceil(Math.max(...pts.map((p) => p.best_lift ?? 0)) + 0.5)
  const X = (v: number) => pad.l + (v / sMax) * (w - pad.l - pad.r)
  const Y = (v: number) => pad.t + (1 - v / lMax) * (height - pad.t - pad.b)
  const yt = Array.from({ length: Math.floor(lMax / 2) + 1 }, (_, i) => i * 2)
  const xt = [0, 0.2, 0.4, 0.6]
  const top10 = new Set(pts.filter((p) => p.shap_rank <= 10).map((p) => p.signal))
  const surv = pts.find((p) => p.survives)
  const lead = pts.find((p) => p.shap_rank === 1)!

  return (
    <div className="panel" ref={ref}>
      <svg width={w} height={height} role="img" aria-label={`Scatter of 59 sensors: SHAP importance against best lift. Spearman correlation ${f2(r.shap_lift_spearman)}. None of the top 10 by SHAP survives the permutation test.`}>
        {yt.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={Y(t)} y2={Y(t)} stroke={t === 0 ? C.gridMajor : C.gridMinor} />
            <text x={pad.l - 8} y={Y(t) + 4} textAnchor="end" className="ax">{t}×</text>
          </g>
        ))}
        {xt.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={pad.t} y2={height - pad.b} stroke={C.gridMinor} />
            <text x={X(t)} y={height - pad.b + 16} textAnchor="middle" className="ax">{t.toFixed(1)}</text>
          </g>
        ))}
        <line x1={pad.l} x2={w - pad.r} y1={Y(1)} y2={Y(1)} stroke={C.ink} strokeDasharray="3 3" />
        <text x={w - pad.r} y={Y(1) + 15} textAnchor="end" className="ax strong halo">{narrow ? 'lift 1× = chance' : 'lift 1× = no better than chance'}</text>
        <text x={pad.l} y={height - 6} className="ax">model importance, mean |SHAP| →</text>
        <text x={pad.l + 4} y={pad.t + 10} className="ax">↑ best lift before a break</text>
        {pts.map((p) => {
          const t = top10.has(p.signal)
          return (
            <circle
              key={p.signal}
              cx={X(p.shap)} cy={Y(p.best_lift ?? 0)} r={t ? 5.5 : 4}
              fill={t ? C.sensor : C.paper}
              stroke={t ? C.paper : C.ink2}
              strokeWidth={t ? 2 : 1.3}
            />
          )
        })}
        {surv && (
          <g>
            <circle cx={X(surv.shap)} cy={Y(surv.best_lift!)} r={9} fill="none" stroke={C.brk} strokeWidth={1.5} />
            <text x={X(surv.shap) + 14} y={Y(surv.best_lift!) + 4} className="anno halo" fill={C.brk}>{surv.signal}: highest lift, SHAP rank {surv.shap_rank}</text>
          </g>
        )}
        <text x={X(lead.shap) - 10} y={Y(lead.best_lift!) - 12} textAnchor="end" className="anno halo">{lead.signal}: SHAP rank 1, lift {lead.best_lift!.toFixed(1)}×</text>
        {/* hit targets */}
        {pts.map((p) => (
          <circle key={'h' + p.signal} cx={X(p.shap)} cy={Y(p.best_lift ?? 0)} r={11} fill="transparent"
            onPointerEnter={() => setHov(p)} onPointerLeave={() => setHov(null)} />
        ))}
      </svg>
      {hov && (
        <div className="tip mono" style={{ left: Math.min(X(hov.shap) + 14, w - 210), top: Math.max(0, Y(hov.best_lift ?? 0) - 64) }}>
          <div>{hov.signal} · SHAP rank {hov.shap_rank}</div>
          <div>best rule {hov.best_rule}, lift {f2(hov.best_lift ?? 0)}×</div>
          <div>q = {f3(hov.best_q ?? 1)} · {hov.survives ? 'survives (break level)' : 'does not survive'}</div>
        </div>
      )}
      <ul className="legend mono">
        <li><i className="lg lg-dot on" />top 10 by SHAP</li>
        <li><i className="lg lg-dot" />other sensors</li>
        <li><i className="lg lg-ring" />survives the break-level test</li>
      </ul>
    </div>
  )
}
