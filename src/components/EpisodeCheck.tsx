import { C } from '../lib/colors'
import { r, pct, f3 } from '../lib/data'

// Step 6: the one rule that survives at break level, counted two ways.
const top = r.episode_level.top_break_level
const nB = r.meta.eval_breaks
const nE = r.episode_level.eval_episodes
const hitsB = Math.round(top.coverage * nB)
const hitsE = top.covered_ep
// From README / METHOD s.6 (verified by hand): 7 of the rule's 8 break-level hits fall in one 10-hour episode.
const SAME_EVENING = 7

function Row({ n, k, label, verdict, ok, group }: { n: number; k: number; label: string; verdict: string; ok: boolean; group?: number }) {
  return (
    <div className="ep-row">
      <div className="ep-lab"><b>{label}</b><span className="mono">{k} of {n} covered · {verdict}</span></div>
      <div className="units tall" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }} role="img" aria-label={`${k} of ${n} covered`}>
        {Array.from({ length: n }, (_, i) => (
          <i key={i} className={i < k ? 'u on' : 'u'} style={i < k ? { background: ok ? C.rule : C.brk } : undefined} />
        ))}
        {group && <span className="ep-brace" style={{ width: `${(group / n) * 100}%` }} />}
      </div>
    </div>
  )
}

export default function EpisodeCheck() {
  return (
    <div className="panel">
      <p className="ep-title mono">RULE {top.id.replace(':', ' ').toUpperCase()} · alarm when {r.detectors[0].signal} drops below {r.detectors[0].threshold.toFixed(1)}</p>
      <Row n={nB} k={hitsB} label="Counted per break" verdict={`q = ${f3(top.q)}, survives`} ok group={SAME_EVENING} />
      <p className="ep-note mono"><span className="ep-tick" style={{ width: `${(SAME_EVENING / nB) * 100}%` }} />{SAME_EVENING} of these {hitsB} hits fall in one evening</p>
      <Row n={nE} k={hitsE} label="Counted per episode" verdict={`q = ${f3(top.q_ep)}, fails`} ok={false} />
      <dl className="ledger mono">
        <div><dt>Rules tested</dt><dd>{r.detectors.length}, thresholds set on earlier weeks only</dd></div>
        <div><dt>Permutation null</dt><dd>{r.meta.permutations.toLocaleString()} draws, Benjamini-Hochberg at q &lt; {r.meta.alpha}</dd></div>
        <div><dt>Survive per break</dt><dd>{r.union.n_detectors} of {r.detectors.length} (coverage {pct(top.coverage, 1)})</dd></div>
        <div><dt>Survive per episode</dt><dd>{r.episode_level.survivors} of {r.detectors.length}</dd></div>
        <div><dt>Fake break times</dt><dd>{r.validation.null_calibration.survivors_per_run.filter((x) => x === 0).length} of {r.validation.null_calibration.runs} runs give 0 survivors</dd></div>
        <div><dt>Labels shifted 1 step</dt><dd>PR-AUC {f3(r.validation.leakage.pr_auc)} → {f3(r.validation.leakage.pr_auc_shifted)}</dd></div>
      </dl>
    </div>
  )
}
