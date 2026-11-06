import r from './data/results.json'

type Det = { id: string; rule: string; direction: string; coverage: number; lift: number | null; q: number; false_alarms_per_day: number; survives: boolean }
const pct = (x: number) => `${(100 * x).toFixed(1)}%`

export default function App() {
  const m = r.meta
  const dets = (r.detectors as Det[]).slice(0, 12)
  const top10 = [...r.scatter].sort((a, b) => a.shap_rank - b.shap_rank).slice(0, 10)
  return (
    <main>
      <header>
        <h1>Sheet-break precursor lab</h1>
        <p>{m.breaks} real paper-machine sheet breaks over {m.start.slice(0, 10)} to {m.end.slice(0, 10)}, {m.signals} sensors at {m.sampling_min}-minute sampling. Which sensor behaviours actually come before a break?</p>
        <p className="note">Interim results page. The full guided view is in progress; the method is in METHOD.md.</p>
      </header>

      <section>
        <h2>1 · How far ahead are breaks predictable?</h2>
        <p>PR-AUC on the held-out last week, each model next to its baselines.</p>
        <table>
          <thead><tr><th>Horizon</th><th>Gradient boosting</th><th>Logistic</th><th>Persistence</th><th>Prevalence</th></tr></thead>
          <tbody>{r.horizons.map((h) => (
            <tr key={h.h_min}><td>{h.h_min} min</td><td>{h.gbm.toFixed(3)}</td><td>{h.logreg.toFixed(3)}</td><td>{h.persistence.toFixed(3)}</td><td>{h.prevalence.toFixed(3)}</td></tr>
          ))}</tbody>
        </table>
      </section>

      <section>
        <h2>2 · Which detectors survive the permutation null?</h2>
        <p>{r.detectors.length} single-signal rules, thresholds set on earlier weeks only, scored on {m.eval_breaks} breaks with {m.permutations.toLocaleString()} permutations and Benjamini-Hochberg. {r.union.n_detectors} survive{r.union.n_detectors === 1 ? 's' : ''} at q &lt; {m.alpha}. Breaks cluster, so each rule is also scored once per episode ({r.episode_level.eval_episodes} episodes): {r.episode_level.survivors} survive at that level.</p>
        <table>
          <thead><tr><th>Detector</th><th>Fires</th><th>Coverage</th><th>Lift</th><th>q</th><th>False alarms / day</th></tr></thead>
          <tbody>{dets.map((d) => (
            <tr key={d.id} className={d.survives ? 'keep' : ''}><td>{d.id}</td><td>{d.direction}</td><td>{pct(d.coverage)}</td><td>{d.lift?.toFixed(2)}</td><td>{d.q.toFixed(3)}</td><td>{d.false_alarms_per_day.toFixed(1)}</td></tr>
          ))}</tbody>
        </table>
      </section>

      <section>
        <h2>3 · Importance is not lift</h2>
        <p>Survivors among the 10 highest-SHAP signals: {r.top10_shap_survivors}. Spearman correlation of SHAP importance and best lift: {r.shap_lift_spearman.toFixed(2)}.</p>
        <table>
          <thead><tr><th>SHAP rank</th><th>Signal</th><th>Best rule</th><th>Best lift</th><th>Best q</th></tr></thead>
          <tbody>{top10.map((p) => (
            <tr key={p.signal}><td>{p.shap_rank}</td><td>{p.signal}</td><td>{p.best_rule}</td><td>{p.best_lift?.toFixed(2)}</td><td>{p.best_q?.toFixed(3)}</td></tr>
          ))}</tbody>
        </table>
      </section>

      <footer>Data: Ranjan et al., <a href="https://arxiv.org/abs/1809.10717">Rare Event Classification in Multivariate Time Series</a> (ProcessMiner), CC BY-NC-SA 4.0. Built by <a href="https://ibiraheel.com">Ibi Raheel</a>.</footer>
    </main>
  )
}
