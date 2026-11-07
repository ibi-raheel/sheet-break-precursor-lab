import raw from '../data/results.json'

// Everything on the page is read from the precomputed pipeline output.
export const r = raw

const T = r.timeline
export const minutes: number[] = T.minutes
export const values: number[] = T.values
export const onsets: number[] = T.onsets
export const masks: number[][] = T.masks
export const N = minutes.length
export const START = Date.UTC(1999, 4, 1, 0, 0) // meta.start, 1999-05-01 00:00
export const TOTAL_MIN = minutes[N - 1]
export const onsetMin = onsets.map((i) => minutes[i])

// Episodes: breaks closer than meta gap_min (120 min) are chained.
const GAP = r.episodes.gap_min
export const isFirst = onsetMin.map((m, k) => k === 0 || m - onsetMin[k - 1] >= GAP)
export type Episode = { start: number; end: number; breaks: number[] }
export const episodes: Episode[] = []
onsetMin.forEach((m, k) => {
  if (isFirst[k]) episodes.push({ start: m, end: m, breaks: [k] })
  else { const e = episodes[episodes.length - 1]; e.end = m; e.breaks.push(k) }
})

// Break-history hazard, recomputed in the browser from the three fitted
// Hawkes parameters (train weeks) and past break times only (METHOD.md s.9):
// p(t) = 1 - exp(-[mu*h + (alpha/beta) E(t) (1 - e^(-beta h))]),  h = 20 min
const { mu, a, b } = r.reliability.hawkes
const H = 20
export const hazard = new Float32Array(N)
{
  let E = 0, last = minutes[0], k = 0
  const tail = (a / b) * (1 - Math.exp(-b * H))
  for (let i = 0; i < N; i++) {
    const m = minutes[i]
    E *= Math.exp(-b * (m - last)); last = m
    while (k < onsets.length && onsets[k] <= i) { E += Math.exp(-b * (m - minutes[onsets[k]])); k++ }
    hazard[i] = 1 - Math.exp(-(mu * H + tail * E))
  }
}
export const eligible = new Uint8Array(N).fill(1)
masks.forEach(([s, e]) => { for (let i = s; i <= e && i < N; i++) eligible[i] = 0 })
// Illustrative alarm: on for the top 10% of eligible rows by hazard.
export const ALARM_SHARE = 0.1
export const hazardAlarm = new Uint8Array(N)
{
  const v: number[] = []
  for (let i = 0; i < N; i++) if (eligible[i]) v.push(hazard[i])
  v.sort((x, y) => x - y)
  const thr = v[Math.floor(v.length * (1 - ALARM_SHARE))]
  for (let i = 0; i < N; i++) hazardAlarm[i] = eligible[i] && hazard[i] >= thr ? 1 : 0
}
export let hazMax = 0
for (let i = 0; i < N; i++) if (hazard[i] > hazMax) hazMax = hazard[i]

// Robust y-range for the trace (0.5 to 99.5 percentile).
const sorted = [...values].sort((x, y) => x - y)
export const yLo = sorted[Math.floor(N * 0.005)]
export const yHi = sorted[Math.floor(N * 0.995)]

const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export function clock(min: number, withDay = true) {
  const d = new Date(START + min * 60000)
  const hh = String(d.getUTCHours()).padStart(2, '0'), mm = String(d.getUTCMinutes()).padStart(2, '0')
  return withDay ? `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${hh}:${mm}` : `${hh}:${mm}`
}
export function rowAt(min: number) {
  let lo = 0, hi = N - 1
  while (lo < hi) { const mid = (lo + hi) >> 1; if (minutes[mid] < min) lo = mid + 1; else hi = mid }
  return lo
}
export const f3 = (x: number) => x.toFixed(3)
export const f2 = (x: number) => x.toFixed(2)
export const f1 = (x: number) => x.toFixed(1)
export const pct = (x: number, d = 0) => `${(100 * x).toFixed(d)}%`
export const pv = (p: number) => (p >= 0.995 ? '1.00' : p < 0.1 ? p.toFixed(3) : p.toFixed(2))

// The one break-level survivor: x6 below a fixed threshold (detectors[0]).
// The trace on the strip is x6 itself, so the rule can be drawn exactly.
export const RULE = r.detectors.find((d) => d.id === 'x6:level')!
export const ruleAlarm = new Uint8Array(N)
for (let i = 0; i < N; i++) ruleAlarm[i] = eligible[i] && values[i] < RULE.threshold ? 1 : 0

// Rising edges of an alarm channel, as row indices.
export function edges(a: Uint8Array, i0 = 0, i1 = N - 1) {
  const out: number[] = []
  for (let i = Math.max(1, i0); i <= i1; i++) if (a[i] && !a[i - 1]) out.push(i)
  return out
}
