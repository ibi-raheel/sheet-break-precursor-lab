import { useEffect, useState, type ReactNode } from 'react'
import { r, f1, f2, f3, pct, pv } from './lib/data'
import Schematic from './components/Schematic'
import TrendStrip from './components/TrendStrip'
import DaySheets from './components/DaySheets'
import ModelPanel from './components/ModelPanel'
import ShapScatter from './components/ShapScatter'
import AlarmBudget from './components/AlarmBudget'
import EpisodeCheck from './components/EpisodeCheck'

// Every number below is read from src/data/results.json (pipeline output).
const m = r.meta
const rel = r.reliability
const H20 = r.hybrid.horizons['20']
const H60 = r.hybrid.horizons['60']
const AP = H20.ap as Record<string, Record<string, number>>
const APC = H20.ap_chance.models as Record<string, { first: { obs: number; null_mean: number; p: number }; quiet: { obs: number; null_mean: number; p: number } }>
const CI = H20.ap_ci as Record<string, Record<string, { point: number; lo: number; hi: number }>>
const LC = r.hybrid.lead_check.models as Record<string, Record<string, { obs: number; prev: number; p: number }>>
const B10 = H20.budgets['0.1'] as unknown as Record<string, { warned_first: number; warned_follow: number; fa_per_shift: number }>
const EV = r.events.targets['1.0'] as unknown as Record<string, { warned_breaks: number; recall_first: number; recall_follow?: number; fa_per_shift_test: number; on_share_test: number; lead_min_median: number }>
const ECON = r.economics.at_1_fa_per_shift as Record<string, { breakeven_e: number }>
const sgn = (x: number) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(3)
const ci = (c: { point: number; lo: number; hi: number }) => `${sgn(c.point)} (${sgn(c.lo)} to ${sgn(c.hi)})`
const ord = (n: number) => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] ?? 'th')
const firstRatio = AP.sensor_gbm.first / AP.sensor_gbm.prev_first

const STEPS = [
  { id: 'machine', short: 'The machine' },
  { id: 'clusters', short: 'Breaks cluster' },
  { id: 'ai', short: 'Can AI see it?' },
  { id: 'importance', short: 'Importance vs cause' },
  { id: 'alarm', short: 'Operator alarm' },
  { id: 'proof', short: 'How we know' },
]

function useActiveStep() {
  const [active, setActive] = useState(0)
  const [progress, setProgress] = useState(0)
  useEffect(() => {
    const els = STEPS.map((s) => document.getElementById(s.id)!)
    const on = () => {
      const mid = window.innerHeight * 0.4
      let a = 0
      els.forEach((el, i) => { if (el.getBoundingClientRect().top < mid) a = i })
      setActive(a)
      const doc = document.documentElement
      setProgress(Math.min(1, Math.max(0, window.scrollY / Math.max(1, doc.scrollHeight - window.innerHeight))))
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    window.addEventListener('resize', on)
    return () => { window.removeEventListener('scroll', on); window.removeEventListener('resize', on) }
  }, [])
  return { active, progress }
}

function Step({ n, id, title, lede, children, details }: { n: number; id: string; title: string; lede: ReactNode; children: ReactNode; details: ReactNode }) {
  return (
    <section className="step" id={id} aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}><span className="step-n mono" aria-hidden="true">{n}</span>{title}</h2>
      <p className="lede">{lede}</p>
      <div className="visual">{children}</div>
      <details className="more">
        <summary><span>Show details</span><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5 6 7.5l3-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg></summary>
        <div className="more-body">{details}</div>
      </details>
    </section>
  )
}

export default function App() {
  const { active, progress } = useActiveStep()
  return (
    <>
      <div className="mbar" aria-hidden="true">
        <div className="mbar-fill" style={{ transform: `scaleX(${progress})` }} />
        <span className="mono">{active + 1}/{STEPS.length} · {STEPS[active].short}</span>
      </div>
      <div className="frame">
        <nav className="rail" aria-label="Steps">
          <ol>
            {STEPS.map((s, i) => (
              <li key={s.id} className={i === active ? 'on' : i < active ? 'past' : ''}>
                <a href={`#${s.id}`} aria-current={i === active ? 'step' : undefined}>
                  <span className="mono">{i + 1}</span>{s.short}
                </a>
              </li>
            ))}
          </ol>
        </nav>
        <main>
          <header className="mast">
            <h1>Sheet-break precursor lab</h1>
            <p className="mast-lede">Can a paper machine's sensors warn of a sheet break before it happens? Six steps through {m.breaks} real breaks, with the trained model next to the simple baseline at every turn.</p>
            <p className="mast-meta mono">
              <span>RECORD 1 to 29 May</span><span>{m.signals} sensors</span><span>{m.sampling_min}-min sampling</span><span>{m.rows.toLocaleString()} rows</span>
            </p>
            <Schematic />
          </header>

          <Step n={1} id="machine" title={`One machine, 28 days, ${m.breaks} breaks`}
            lede={<>The chart recorder for one of the {m.signals} sensors. Each red mark is a sheet break, and the line is down about {Math.round(r.downtime.mean_min_per_break)} minutes after it. The purple pen is the chance of a break in the next 20 minutes, from break history alone.</>}
            details={<>
              <ul>
                <li>Downtime is inferred from timestamp gaps right after a break ({r.downtime.gaps_within_2_rows_after_break} of {r.downtime.gaps_total} gaps): {f1(r.downtime.total_hours)} line-hours over 28 days, {f1(r.downtime.hours_per_day)} h a day. The dataset does not label it.</li>
                <li>The trace is sensor {r.timeline.signal}, the one rule that survives the break-level test (step 6). The {m.mask_rows}-row grey band after each break is the restart mask: rows dropped so a model does not learn the restart itself.</li>
                <li>The hazard pen is recomputed in your browser from the three fitted parameters and past break times only. The hazard alarm is drawn for the top 10% of rows, an illustrative threshold. The sensor rule alarm is {r.timeline.signal} below {r.detectors[0].threshold.toFixed(1)}.</li>
                <li>Per-row scores of the AI and hybrid models are not shipped with the page; their alarms are measured in steps 3 and 5.</li>
              </ul>
            </>}>
            <TrendStrip />
          </Step>

          <Step n={2} id="clusters" title="Breaks come in clusters"
            lede={<>{m.breaks} breaks fall into {r.episodes.all_episodes} episodes. In the 2 hours after a break the chance of another is {pct(rel.p_break_within_2h_after_break)}, against {pct(rel.p_break_in_random_2h)} for any 2-hour window.</>}
            details={<>
              <p>Fitted on the {rel.train_breaks} training breaks. A self-exciting (Hawkes) model fits best: a break raises the rate of the next one, and the boost halves every {Math.round(rel.hawkes.excitation_halflife_min)} minutes.</p>
              <div className="tbl"><table>
                <thead><tr><th>Model of break arrival</th><th>Parameters</th><th>AIC</th><th>Reading</th></tr></thead>
                <tbody>
                  <tr><td>Exponential (memoryless)</td><td>1</td><td>{f1(rel.exponential.aic)}</td><td>rejected by KS, p = {f3(rel.exponential.ks_p)}</td></tr>
                  <tr><td>Weibull renewal</td><td>2</td><td>{f1(rel.weibull.aic)}</td><td>shape k = {f2(rel.weibull.k)} (CI {f2(rel.weibull.k_ci[0])} to {f2(rel.weibull.k_ci[1])}): risk highest right after a break</td></tr>
                  <tr className="hl"><td>Hawkes, self-exciting</td><td>3</td><td>{f1(rel.hawkes.aic)}</td><td>branching ratio {f2(rel.hawkes.branching)}: about {pct(rel.hawkes.branching)} of breaks triggered by an earlier one (no CI)</td></tr>
                </tbody>
              </table></div>
              <p>Follow-on breaks come a median {rel.followon_since_prev_min.q50} min after the previous one; 90% within {rel.followon_since_prev_min.q90} min. A watch window of about 105 minutes after a restart covers 90% of them, with no machine learning at all.</p>
            </>}>
            <DaySheets />
          </Step>

          <Step n={3} id="ai" title="Can the sensors see a first break coming?"
            lede={<>Break history cannot: on first breaks it scores below the base rate. The AI sensor model scores {firstRatio.toFixed(1)}× the base rate, above chance (p = {pv(APC.sensor_gbm.first.p)}). The hybrid is borderline, and by 60 minutes ahead the signal is gone.</>}
            details={<>
              <p>PR-AUC on {H20.test_breaks} out-of-sample breaks ({H20.test_first_breaks} first, {H20.test_follow_breaks} follow-on), forward-chaining folds that never split an episode, {H20.h_min}-minute look-ahead. Prevalence: all {f3(AP.hawkes.prev_all)}, first {f3(AP.hawkes.prev_first)}, follow-on {f3(AP.hawkes.prev_follow)}.</p>
              <div className="tbl"><table>
                <thead><tr><th>Score</th><th>All</th><th>First</th><th>Follow-on</th><th>Quiet rows</th></tr></thead>
                <tbody>
                  {(['hawkes', 'sensor_gbm', 'hybrid_gbm', 'hybrid_superpose', 'hybrid_logistic'] as const).map((k) => (
                    <tr key={k}><td>{({ hawkes: 'History-only hazard', sensor_gbm: 'AI sensor model', hybrid_gbm: 'Hybrid (boosted from hazard)', hybrid_superpose: 'Hybrid, superposed (negative result)', hybrid_logistic: 'Hybrid, logistic (fails)' })[k]}</td>
                      <td>{f3(AP[k].all)}</td><td>{f3(AP[k].first)}</td><td>{f3(AP[k].follow)}</td><td>{f3((H20.ap_quiet as Record<string, number>)[k])}</td></tr>
                  ))}
                </tbody>
              </table></div>
              <div className="tbl"><table>
                <thead><tr><th>Paired difference, 95% CI</th><th>All</th><th>First</th><th>Follow-on</th></tr></thead>
                <tbody>
                  <tr><td>Hybrid − hazard</td><td>{ci(CI.hybrid_gbm_minus_hawkes.ap_all)}</td><td>{ci(CI.hybrid_gbm_minus_hawkes.ap_first)}</td><td>{ci(CI.hybrid_gbm_minus_hawkes.ap_follow)}</td></tr>
                  <tr><td>Hybrid − AI sensor model</td><td>{ci(CI.hybrid_gbm_minus_sensor_gbm.ap_all)}</td><td>{ci(CI.hybrid_gbm_minus_sensor_gbm.ap_first)}</td><td>{ci(CI.hybrid_gbm_minus_sensor_gbm.ap_follow)}</td></tr>
                </tbody>
              </table></div>
              <div className="tbl"><table>
                <thead><tr><th>First-break PR-AUC with lead time</th><th>≥ 2 min</th><th>≥ 6 min</th><th>≥ 10 min</th></tr></thead>
                <tbody>
                  <tr><td>Base rate</td>{['1', '3', '5'].map((k) => <td key={k}>{f3(LC.sensor_gbm[k].prev)}</td>)}</tr>
                  {(['hawkes', 'sensor_gbm', 'hybrid_gbm', 'grade_only'] as const).map((s) => (
                    <tr key={s}><td>{({ hawkes: 'History-only hazard', sensor_gbm: 'AI sensor model', hybrid_gbm: 'Hybrid', grade_only: 'Product grade only' })[s]}</td>
                      {['1', '3', '5'].map((k) => <td key={k}>{f3(LC[s][k].obs)} <span className="dim">p {pv(LC[s][k].p)}</span></td>)}</tr>
                  ))}
                </tbody>
              </table></div>
              <ul>
                <li>The hazard's lack of first-break skill is structural: it is highest right after a break, and first breaks come at least 2 hours after one.</li>
                <li>Above chance is not useful yet: {firstRatio.toFixed(1)}× a {pct(AP.hawkes.prev_first, 1)} base rate is a screening signal for a process engineer, not an operator alarm.</li>
                <li>At 60 minutes the hazard wins overall (all-break PR-AUC {f3((H60.ap as Record<string, Record<string, number>>).hawkes.all)} vs hybrid {f3((H60.ap as Record<string, Record<string, number>>).hybrid_gbm.all)}) and no score beats chance on first breaks.</li>
                <li>Model: LightGBM, {r.hybrid.gbm_params.n_estimators} trees, {r.hybrid.gbm_params.num_leaves} leaves, untuned. The hybrid starts each fit from the hazard's log-odds, so its trees learn only the sensor correction.</li>
              </ul>
            </>}>
            <ModelPanel />
          </Step>

          <Step n={4} id="importance" title="What the model leans on is not what precedes a break"
            lede={<>None of the 10 sensors the model relies on most passes the permutation test, and importance barely tracks lift (Spearman {f2(r.shap_lift_spearman)}). The sensor with the highest lift ranks {ord(r.scatter.find((p) => p.survives)?.shap_rank ?? 0)} by SHAP.</>}
            details={<>
              <p>SHAP measures what the model uses to fit; lift measures how much more often a rule fires in the 20 minutes before a break than at other times. A model can lean on a sensor that tracks product grade or line speed without that sensor ever warning of a break.</p>
              <div className="tbl"><table>
                <thead><tr><th>SHAP rank</th><th>Sensor</th><th>Best rule</th><th>Best lift</th><th>q</th></tr></thead>
                <tbody>{[...r.scatter].sort((a, b) => a.shap_rank - b.shap_rank).slice(0, 10).map((p) => (
                  <tr key={p.signal}><td>{p.shap_rank}</td><td>{p.signal}</td><td>{p.best_rule}</td><td>{f2(p.best_lift ?? 0)}×</td><td>{f3(p.best_q ?? 1)}</td></tr>
                ))}</tbody>
              </table></div>
            </>}>
            <ShapScatter />
          </Step>

          <Step n={5} id="alarm" title="What an operator alarm would look like"
            lede={<>Give every score the same alarm-on time and count the breaks it warns at least 10 minutes ahead. The hazard catches follow-ons quietly; the AI catches some first breaks but flickers, about {Math.round(B10.sensor_gbm.fa_per_shift)} false alarms a shift at 10%.</>}
            details={<>
              <p>The other fair comparison fixes the false-alarm count instead (about 1 per 8-hour shift, thresholds set on training weeks, {r.events.test_breaks} test breaks). It flips the picture:</p>
              <div className="tbl"><table>
                <thead><tr><th>Score at ~1 false alarm per shift</th><th>Breaks warned</th><th>Alarm on</th><th>False alarms / shift</th></tr></thead>
                <tbody>
                  <tr><td>AI sensor model</td><td>{EV.gbm.warned_breaks} of {r.events.test_breaks}</td><td>{pct(EV.gbm.on_share_test, 1)}</td><td>{f1(EV.gbm.fa_per_shift_test)}</td></tr>
                  <tr><td>History-only hazard</td><td>{EV.hawkes.warned_breaks} of {r.events.test_breaks}</td><td>{pct(EV.hawkes.on_share_test, 1)}</td><td>{f1(EV.hawkes.fa_per_shift_test)}</td></tr>
                </tbody>
              </table></div>
              <ul>
                <li>Cost, at 5 minutes per alarm check and 1 false alarm per shift: the hazard alarm pays only if operators prevent at least {pct(ECON.hawkes.breakeven_e)} of the breaks they are warned about. The AI model would need to prevent {pct(ECON.gbm.breakeven_e)}, so it never pays as an alarm.</li>
                <li>Neither matching is fair alone: count-matching flatters the long, blocky hazard alarm; time-matching flatters the flickering sensor alarms. Both are reported.</li>
                <li>Of the 8 first-break comparisons (2 learned models × 4 budgets), 3 clear the chance null at p &lt; 0.05, with no multiplicity correction.</li>
              </ul>
            </>}>
            <AlarmBudget />
          </Step>

          <Step n={6} id="proof" title="How we know"
            lede={<>The one rule that passes a {m.permutations.toLocaleString()}-draw permutation test does it on a single bad evening. Count each episode once and {r.episode_level.survivors} of {r.detectors.length} rules survive.</>}
            details={<>
              <ul>
                <li>No leakage: features look back only; folds move forward in time, never split an episode, and purge the last {H20.h_min} minutes before each test block. Hazard parameters are refitted on each training prefix.</li>
                <li>Chance nulls keep each score's own shape: it is rotated in time inside each block, so on-time, flicker and autocorrelation stay, and only alignment with breaks is removed.</li>
                <li>Uncertainty: paired bootstrap over day blocks for PR-AUC, over whole episodes for recall, 2,000 resamples. Seeds 11, 12, 13 give hybrid all-break PR-AUC {r.hybrid.horizons['20'].seed_robustness.map((s) => f3(s.ap_all_hybrid)).join(' and ')} next to {f3(AP.hybrid_gbm.all)}.</li>
                <li>Hand checks: lift, downtime, Hawkes likelihood, break-even and a matched-budget alarm are recalculated by hand in 36 tests; the Hawkes and Weibull fitters recover known parameters on simulated data.</li>
                <li>Limits: {m.eval_breaks} scored breaks is a small sample; many comparisons are reported without correction; downtime is an interpretation of timestamp gaps; sensor names are anonymised, so no rule can be tied to a physical cause here.</li>
              </ul>
            </>}>
            <EpisodeCheck />
          </Step>

          <footer className="foot">
            <p>Data: Ranjan, Reddy, Mustonen, Paynabar and Pourak (2018), <a href="https://arxiv.org/abs/1809.10717">Rare Event Classification in Multivariate Time Series</a>, arXiv:1809.10717. Licence CC BY-NC-SA 4.0; derived results are shared under the same licence and the raw data is not redistributed. Results are precomputed by the pipeline in this repository; nothing here calls a live model.</p>
            <p className="foot-sign">More work at <a href="https://ibiraheel.com">ibiraheel.com</a></p>
          </footer>
        </main>
      </div>
    </>
  )
}
