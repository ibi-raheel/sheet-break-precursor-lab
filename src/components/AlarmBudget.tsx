import { useState } from 'react'
import { C } from '../lib/colors'
import { r, f1, pv as pv0 } from '../lib/data'

const pv = (p: number) => (p === 0 ? '< 0.002' : pv0(p))

// Step 5: every score gets the same alarm-on time; count warned breaks
// (alarm on 20 to 10 min before) and compare with the score's own rotated copy.
const H20 = r.hybrid.horizons['20']
const BUDGETS = ['0.02', '0.05', '0.1', '0.2'] as const
const ROWS = [
  { id: 'hawkes', name: 'History only', col: C.hazard },
  { id: 'sensor_gbm', name: 'AI sensor model', col: C.sensor },
  { id: 'hybrid_gbm', name: 'Hybrid', col: C.hybrid },
] as const
type B = { warned_first: number; warned_follow: number; fa_per_shift: number }
type Ch = { first_obs: number; first_null_mean: number; first_p: number; follow_null_mean: number; follow_p: number }

function Units({ n, k, col, mark, label }: { n: number; k: number; col: string; mark?: number; label: string }) {
  return (
    <div className="units" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }} role="img" aria-label={`${label}: ${k} of ${n} warned${mark !== undefined ? `, chance about ${f1(mark)}` : ''}`}>
      {Array.from({ length: n }, (_, i) => <i key={i} className={i < k ? 'u on' : 'u'} style={i < k ? { background: col } : undefined} />)}
      {mark !== undefined && <b className="u-null" style={{ left: `${(mark / n) * 100}%` }} aria-hidden="true" />}
    </div>
  )
}

export default function AlarmBudget() {
  const [bi, setBi] = useState(2)
  const b = BUDGETS[bi]
  const bud = H20.budgets[b] as unknown as Record<string, B>
  const ch = H20.chance.budgets[b] as unknown as Record<string, Ch> & { random_iid: { first_mean: number } }
  const pctOn = Math.round(+b * 100)
  return (
    <div className="panel">
      <div className="controls">
        <label className="slider wide">
          <span>Alarm on</span>
          <input type="range" min={0} max={3} step={1} value={bi} onChange={(e) => setBi(+e.target.value)} aria-valuetext={`${pctOn}% of operating time`} />
          <span className="mono">{pctOn}%</span>
        </label>
      </div>
      <div className="annun" role="table" aria-label={`Breaks warned at least 10 minutes ahead with the alarm on ${pctOn}% of the time`}>
        <div className="an-row an-head" role="row">
          <span role="columnheader">Score</span>
          <span role="columnheader">First breaks warned, of {H20.test_first_breaks}</span>
          <span role="columnheader">Follow-on warned, of {H20.test_follow_breaks}</span>
          <span role="columnheader" className="num">False alarms per shift</span>
        </div>
        {ROWS.map((m) => {
          const x = bud[m.id], c = ch[m.id]
          return (
            <div className="an-row" role="row" key={m.id}>
              <span role="cell" className="an-name"><i className="dot" style={{ background: m.col }} />{m.name}</span>
              <span role="cell" className="an-cell">
                <Units n={H20.test_first_breaks} k={x.warned_first} col={m.col} mark={c.first_null_mean} label={`${m.name}, first breaks`} />
                <span className="mono an-k">{x.warned_first} <em>chance {f1(c.first_null_mean)} · p {pv(c.first_p)}</em></span>
              </span>
              <span role="cell" className="an-cell">
                <Units n={H20.test_follow_breaks} k={x.warned_follow} col={m.col} mark={c.follow_null_mean} label={`${m.name}, follow-on breaks`} />
                <span className="mono an-k">{x.warned_follow} <em>chance {f1(c.follow_null_mean)} · p {pv(c.follow_p)}</em></span>
              </span>
              <span role="cell" className="num mono an-fa">{f1(x.fa_per_shift)}</span>
            </div>
          )
        })}
      </div>
      <p className="panel-note mono">
        <i className="lg lg-nullmark" />chance = the score's own trace rotated in time (500 draws), same on-time and flicker.
        A random score at this budget would warn about {Math.round(ch.random_iid.first_mean)} first breaks because it flickers. Filled squares are counts, not specific breaks.
      </p>
    </div>
  )
}
