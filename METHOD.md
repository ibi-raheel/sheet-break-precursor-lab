# Method

Rare-event precursor analysis on a real paper-machine sheet-break dataset. All numbers below are produced by `pipeline/run.py` and exported to `src/data/results.json`; nothing is hand-entered.

## 1. Data

- **Source:** Ranjan, C., Reddy, M., Mustonen, M., Paynabar, K., Pourak, K. (2018). *Dataset: Rare Event Classification in Multivariate Time Series.* arXiv:1809.10717. Data from a pulp-and-paper mill, released by ProcessMiner Inc.
- **Licence:** CC BY-NC-SA 4.0. This repo is non-commercial; the raw CSV is not redistributed here (see `DATA.md` for how to obtain it). Derived results in `src/data/` are shared under the same licence.
- **Shape (measured):** 18,398 rows, 1999-05-01 00:00 to 1999-05-29 00:06, nominal 2-minute sampling, 59 continuous anonymised sensors, one 8-level categorical (`x28`, treated as a grade code and one-hot encoded) and one binary (`x61`).
- **Events:** 124 rows with `y = 1`. Every one is an isolated row, so 124 break events (consecutive `y = 1` rows would be merged into one event). Break rows are 0.674% of all rows.
- **Gaps:** 107 timestamps are more than 2 minutes after the previous one. 103 of them start within 2 rows after a break: they are post-break downtime the publishers cut out (measured in section 10). Rolling features are defined in rows, not clock time; this is an assumption.
- The whole dataset is used. No subsetting.

## 2. Labels and restart masking

For a lookahead horizon of *h* rows, row *t* is positive when a break onset *o* satisfies *t < o ≤ t + h*:

    label_h(t) = 1  if  ∃ o : o − h ≤ t < o

The break row itself and the *k* rows after each break are **ineligible**: they are not used for training, scoring or as nominal windows. Restart rows would otherwise teach the model "a break just happened", which leaks.

*k* is measured, not guessed. The transient score is the mean absolute row-to-row change of the standardised signals,

    s(t) = (1/59) Σ_j | z_j(t) − z_j(t−1) |

For each training-period break, the calm limit is mean + 2·sd of *s* over the 15 rows before onset. Recovery is the first post-break row that starts 3 consecutive rows below the limit. Over training breaks the median recovery is 2 rows; the 90th percentile is **7 rows (14 min)**, which is used as *k*.

(First version: recovery was defined as returning to the pre-break *level* of every signal. That never happened inside 30 rows for most breaks because the line often settles at a new operating point after a break, so the mask ran into its cap. The change-rate criterion above replaced it before any detector results were looked at.)

## 3. Features (past-only)

For every continuous signal *x* and windows *w* ∈ {5, 15} rows (10 and 30 min): rolling mean, rolling std, deviation from rolling median `x(t) − median(x(t−w+1..t))`, and least-squares slope

    slope_w(t) = Σ_i (i − ī)(x_i − x̄) / Σ_i (i − ī)²,   i over the last w rows

Every feature at row *t* uses rows ≤ *t* only. `tests/test_pipeline.py::test_features_use_only_past_rows` changes the future and asserts the past features do not move.

## 4. Models across horizons

- **Split:** time-blocked, no shuffling. First 75% of rows (to 1999-05-22 06:02) train, last 25% (about 6.8 days, 33 breaks) test.
- **Gradient boosting:** LightGBM, 250 trees, 15 leaves, class weight = negatives / positives, 2 threads.
- **Logistic regression:** standardised features, L2 (C = 0.05), balanced classes.
- **Baselines:** prevalence (PR-AUC of a random score equals the positive rate), persistence (score = minus rows since the last break, i.e. "breaks cluster"), and two fitted break-hazard models that use no sensor data at all (section 9): a Weibull renewal model and a self-exciting (Hawkes) model, both fitted on train weeks only and scored from past break times only.
- **Metric:** PR-AUC (average precision) on eligible test rows.

| Horizon | GBM (531 sensor features) | Logistic | Hawkes hazard (no sensors) | Weibull hazard | Persistence | Prevalence |
|---|---|---|---|---|---|---|
| 2 min | 0.013 | 0.014 | **0.017** | 0.010 | 0.010 | 0.008 |
| 4 min | 0.019 | 0.019 | **0.033** | 0.019 | 0.020 | 0.015 |
| 6 min | 0.038 | 0.026 | **0.049** | 0.029 | 0.030 | 0.023 |
| 10 min | 0.053 | 0.041 | **0.082** | 0.050 | 0.050 | 0.038 |
| 20 min | 0.134 | 0.083 | **0.166** | 0.105 | 0.106 | 0.076 |
| 30 min | 0.191 | 0.123 | **0.253** | 0.163 | 0.170 | 0.112 |

**Uncertainty (moving-block bootstrap, 2,000 resamples of 12-hour blocks of test rows, 95% CI):**

| Difference in PR-AUC | Point | 95% CI | Share of resamples > 0 |
|---|---|---|---|
| GBM − persistence, 20 min | +0.027 | −0.000 to +0.104 | 97% |
| GBM − persistence, 30 min | +0.021 | −0.016 to +0.118 | 85% |
| GBM − Hawkes, 20 min | −0.033 | −0.068 to +0.045 | 38% |
| GBM − Hawkes, 30 min | −0.062 | −0.109 to +0.036 | 25% |

Reading: skill over prevalence is small in absolute terms at every horizon. The GBM's edge over "breaks cluster" at 20 minutes sits on the edge of noise (the CI touches 0), and at 30 minutes it is inside the noise. A three-parameter self-exciting hazard model that only knows *when past breaks happened* scores higher than the 531-feature GBM at every horizon; the difference is not significant either way, but the sensor model never wins. The Weibull hazard is rank-equivalent to persistence (with k < 1 its risk falls monotonically with time since the last break), so its PR-AUC matches persistence; its value is calibration, not ranking. On this line, sensor-model scores alone would not make a usable alarm.

**First break vs follow-on (20 min).** Every test break is labelled first-in-episode (no break in the previous 2 h) or follow-on. PR-AUC with the other class's positive rows removed:

| Score | First-in-episode (prev. 0.046) | Follow-on (prev. 0.033) |
|---|---|---|
| GBM | **0.062** | 0.091 |
| GBM + time-since-break feature | 0.055 | 0.117 |
| Hawkes hazard | 0.036 | **0.233** |
| Persistence | 0.031 | 0.137 |

Follow-on breaks are a restart-stability problem that break timing alone predicts well (Hawkes 7× prevalence). First breaks of an episode are the hard, valuable problem: hazard models are below prevalence there by construction, and the only score above prevalence is the sensor GBM, at 1.36×. That is where sensor data has to earn its place. Adding time-since-break to the GBM does not help first breaks.

**Importance:** TreeSHAP contributions (LightGBM `pred_contrib`) on the 20-minute model, mean |SHAP| over test rows, summed over all features derived from each raw signal.

## 5. Candidate detectors, lift and the permutation null

Each of 59 signals gets three operator-readable rules: **level** (raw value), **slope** (10-min slope) and **deviation** (deviation from 30-min rolling median). 175 detectors had enough data. Detector window *W* = 10 rows (20 min).

**Rolling origin.** The data is cut into 4 blocks of about 7 days. For blocks 2, 3 and 4, a rule's direction and threshold come only from rows *before* that block: direction = the side where the pre-break median sits relative to the nominal median; threshold = the 99th (or 1st) percentile of nominal rows. Scoring pools the 74 breaks in blocks 2 to 4 whose 20-minute window is fully eligible (20.9 days).

For a detector with firing indicator *f(t)*, define `A(t) = 1 if f fires anywhere in (t − W, t]`.

- **Coverage** (pre-break fire rate): `P_pre = mean over breaks of A(o − 1)`
- **Nominal fire rate:** `P_nom = mean of A(t)` over all 11,740 nominal window ends (window fully eligible, no positive labels)
- **Lift:** `L = P_pre / P_nom`
- **Window precision:** `fire_pre / (fire_pre + fire_nom)` in window counts
- **Alarm precision** (exported as `precision`): share of alarm rising edges that fall inside a 20-min pre-break window; **false alarms/day** = the other rising edges ÷ days scored.

**Hand calculation (test `test_lift_and_precision_match_hand_calculation`)** for the top detector, `x6` level below threshold:

    fire_pre = 8 of 74 pre-break windows      P_pre = 8/74      = 0.10811
    fire_nom = 130 of 11,740 nominal windows   P_nom = 130/11740 = 0.011073
    lift = 0.10811 / 0.011073 = 9.763
    window precision = 8 / (8 + 130) = 0.0580

The exported lift (9.763) equals the hand calculation; `test_exported_top_detector_matches_hand_calculation` re-derives it from the exported JSON.

**Permutation test.** For each of B = 5,000 draws, every real break is replaced by a random nominal window end from its own block; the null statistic is the mean of *A* at those ends. Then

    p = (1 + #{null_b ≥ observed}) / (1 + B)

**Episode-aware test.** Breaks cluster (section 9), so the break-level test above treats 7 breaks from one bad evening as 7 independent hits. Episodes are built by chaining breaks less than 2 hours apart (124 breaks → 69 episodes). The episode statistic is the mean of *A*(first break of each episode − 1); the null draws one nominal window end per *episode* from its own block. BH is applied again across the 175 detectors.

**Multiple testing.** Benjamini-Hochberg across all 175 detectors:

    q_(i) = min_{j ≥ i} ( p_(j) · m / j ),   m = 175

A detector **survives** when q < 0.05 and lift > 1.

## 6. Results

- **Break level: 1 of 175 detectors survives:** `x6` level below its 1st-percentile threshold. Lift 9.76, coverage 8/74 breaks (10.8%), q = 0.035, 0.8 false alarms/day, alarm precision 0.23, first firing 14 to 20 minutes before the break.
- **Episode level: 0 of 175 survive.** Scoring each episode once, by whether the rule fired in the 20 minutes before its *first* break (50 episodes in the evaluation blocks), `x6` covers 2 of 50 episodes (p = 0.11, q = 0.58). Its break-level survival came from 7 follow-on breaks inside one 10-hour episode. The best episode-level rule is `x51` level (4 of 50 episodes, p = 0.001, q = 0.17), which does not survive BH either.
- **Bootstrap on the `x6` lift** (resample episodes for the pre-break rate, 12-hour blocks for the nominal rate): 95% CI 0 to 54; 6% of resamples give lift ≤ 1. The point estimate of 9.8 is not a stable number.
- **Importance is not lift:** the top 10 SHAP signals contain **0** survivors. `x6` is SHAP rank 33 of 59. The highest-SHAP signal (`x15`) has best lift 3.2 and q = 0.24. Spearman correlation between a signal's SHAP importance and its best detector lift: **0.08**.
- **Families:** average-linkage hierarchical clustering on 1 − |Spearman ρ| (training rows), cut at 0.3 (|ρ| ≥ 0.7), gives 44 families over 59 signals. Most sensors are not redundant copies of each other.
- **Holdout-only view** (score block 4 only, 32 breaks, thresholds from blocks 1 to 3): 4 survivors (`x6` level, `x34` level, `x14` deviation and slope). The pooled rolling-origin result is stricter because a detector has to hold across three weeks.

## 7. Validation

| Check | Expected | Measured |
|---|---|---|
| Lift / precision hand calculation | equal | 9.763 = 9.763 (unit test) |
| Null calibration: independent fake break times (20 runs, real nominal windows, same detectors) | ~0 survivors; ≤5% of p below 0.05 | 0 survivors in all 20 runs; 3.1% of p < 0.05 |
| **Clustered** null calibration: shift whole real episodes, spacing kept, to random nominal positions in their block (20 runs) | an honest test gives ~0 survivors | break-level (independent) test: **7 false survivors in 1 of 20 runs**, 4.7% of p < 0.05. Episode test: 0 survivors in 20 of 20 runs, 3.3% of p < 0.05 |
| Weibull fit vs empirical CDF of inter-break times | KS not rejected | KS D = 0.105, p = 0.29 (exponential: D = 0.156, p = 0.03, rejected) |
| Hawkes stationarity | branching ratio α/β < 1 | 0.40 |
| Downtime extraction | equals hand count | 130, 24, 2 min on the first 3 breaks (unit test against the raw CSV) |
| Event-level alarm logic, break-even, Hawkes log-likelihood, Weibull MLE | hand calculations / simulated recovery | unit tests (`tests/test_reliability.py`); Weibull MLE recovers k = 0.8 on 20,000 draws, Hawkes fit recovers branching 0.5 on a simulated process |
| Leakage check: shift the 20-min label window one row further from the break | PR-AUC must not jump | 0.134 → 0.110 (drops, no jump) |
| Baselines | every model number next to prevalence and persistence | table in section 4 |
| BH implementation | equals `scipy.stats.false_discovery_control` | unit test |
| Past-only features and labels | future edits do not change past rows | unit tests |

`pytest -q tests`: 25 passed.

## 8. Assumptions and limitations

- Signals are anonymised, so no physical meaning is claimed for `x6` or any family. A plant engineer would name them.
- The `x6` survivor is one episode: 7 of its 8 covered breaks fall in one 10-hour window (24 May 14:54 to 25 May 00:16), and it fails the episode-level test (section 6). It is a real precursor during that episode, not a general mechanism.
- Inter-break times are in clock minutes and so include the post-break downtime; the hazard models describe calendar time, not running time.
- The Hawkes and Weibull likelihoods treat the last train interval as right-censored and condition on the first break.
- 74 scored breaks is a small sample. A coverage of 10.8% is honest but low; most breaks here have no single-sensor precursor at 20 minutes.
- Rolling features are in rows; 107 timestamp gaps make a few windows span more than their nominal duration.
- Thresholds are fixed percentiles (1% / 99%), chosen before results. They were not tuned.
- Only single-signal rules are tested. Multi-signal rules and agent-written hypotheses are out of scope.

## 9. Break clustering: reliability models (train weeks only)

On the 86 training breaks (85 inter-break times τ, clock minutes; the interval after the last one is right-censored at the split):

- coefficient of variation of τ = **1.15** (a Poisson process gives 1); 20% of gaps are under 1 h and 42% under 2 h
- **Exponential** (Poisson, memoryless): `S(t) = exp(−t/λ)`, λ = 354 min
- **Weibull renewal:** `h(t) = (k/λ)(t/λ)^(k−1)`, `S(t) = exp(−(t/λ)^k)`. MLE k = **0.87** (bootstrap 95% CI 0.78 to 0.99, 1,000 resamples), λ = 330 min. k < 1 means the risk of a break is highest right after a break and falls off.
- **Hawkes (self-exciting):** `λ(t) = μ + Σ_{t_i < t} α·e^(−β(t − t_i))`. Log-likelihood on (t₁, T]:

      ℓ = Σ_{i≥2} log λ(t_i) − μ(T − t₁) − (α/β) Σ_i (1 − e^(−β(T − t_i)))

  MLE μ = 0.0017/min (one background break per 10 h), branching ratio α/β = **0.40** (40% of breaks are triggered by an earlier break), excitation half-life ln2/β = 204 min.
- **AIC:** exponential 1169.7, Weibull 1168.9, Hawkes **1160.3**. Self-excitation is the best description of how breaks arrive.

**Scores.** At row *t*, P(break in (t, t+h]) is `1 − S(s+h)/S(s)` for the Weibull model (s = minutes since the last break) and `1 − exp(−[μh + (α/β) E(t)(1 − e^(−βh))])` with `E(t) = Σ_{t_j ≤ t} e^(−β(t − t_j))` for Hawkes. Both use past break times only.

**Operational rule (no ML).** After a break, P(another break within 2 h) = **42%** against 29% for a random 2-hour window. Follow-on breaks come a median 65 minutes after the previous one; 80% within 94 min and 90% within 103 min. A **105-minute post-restart watch window** covers 90% of follow-on breaks on the training weeks.

## 10. Downtime and alarm economics

**Measured downtime.** For each break, downtime = the first timestamp gap within 2 rows after the onset, minus the nominal 2 minutes. 103 of 124 breaks are followed by such a gap (103 of the 107 gaps in the file). Median 14 min when present (IQR 8 to 30), mean **20.3 min per break** over all breaks, **41.9 hours in total** over 28 days: 1.5 line-hours a day, 6.2% of calendar time. First breaks of an episode cost more (23.2 min) than follow-ons (16.7 min).

**Event-level alarms.** Each score's threshold is set on the train weeks to give 0.5, 1 or 2 false alarms per 8-hour operating shift (GBM thresholds from blocked 3-fold out-of-fold scores, so the threshold is not set on in-sample fits). An alarm is a rising edge of score > threshold; it is true if a break follows within 20 minutes. A break counts as **warned** if the alarm is on between 20 and 10 minutes before it (≥ 10 minutes of lead time). Results on the 38 test breaks (19 episodes), target 1 false alarm per shift:

| Score | Breaks warned | First-in-episode | Follow-on | Realised false alarms / shift (test) | Median lead |
|---|---|---|---|---|---|
| GBM | 5 / 38 (13%) | 2 / 19 | 3 / 19 | 1.5 | 14 min |
| GBM + time since break | 4 / 38 (11%) | 1 / 19 | 3 / 19 | 1.6 | 18 min |
| Hawkes hazard (no sensors) | **21 / 38 (55%)** | 3 / 19 | **18 / 19** | 1.6 | 20 min |
| `x6` level rule (fixed threshold) | 9 / 38 (24%) | 1 / 19 | 8 / 19 | 0.8 | 20 min |

At 2 false alarms per shift the Hawkes score warns 28 of 38 (74%), the GBM 8 of 38. Thresholds drift: a target of 1 per shift on train weeks became 1.5 to 1.6 per shift on test weeks.

**Break-even.** With *e* = the share of warned breaks an operator actually prevents (not measurable from this data):

    line-minutes saved per day = recall × breaks/day × downtime per break × e − false alarms/day × check time
    break-even e = false alarms/day × check time / (recall × breaks/day × downtime per break)

Inputs: 5.6 breaks/day (test weeks), 20.3 min per break (measured), 5 min per alarm check (assumed). At 1 false alarm per shift:

| Score | Break-even e | Net line-hours / month at e = 10% | 25% | 50% |
|---|---|---|---|---|
| GBM | 1.50 (never pays) | −10.5 | −9.4 | −7.5 |
| Hawkes watch window | 0.38 | −9.0 | −4.2 | **+3.7** |
| `x6` level rule | 0.43 | −4.5 | −2.5 | +0.9 |

No score pays for itself unless operators can stop a large share of the breaks they are warned about. The sensor GBM never does at this alarm budget. Line-hours convert to money at the plant's own cost per hour; no dollar figure is assumed here.

## 11. Week 1 on a new line

What these numbers say a forward-deployed data scientist should do first:

1. **Day 1: measure the cost and the clustering from the historian.** Post-break downtime per break (here 20 min, 1.5 h/day) and the break hazard (here k = 0.87, branching 0.40). This sizes the prize before any model.
2. **Deploy the post-restart watch window as a zero-ML rule.** Here: after any break, run with extra attention for 105 minutes. On test weeks a hazard-based alarm caught 18 of 19 follow-on breaks at about 1.6 false alarms per shift.
3. **Screen detectors on first-in-episode breaks only, with an episode-aware null.** Break-level screening produced a survivor that is one bad evening; the clustered calibration shows the break-level test can produce 7 false survivors in a run.
4. **Bring the top episode-level candidates (here `x51` level, `x26`/`x16`/`x13` deviation) to the process engineer** to name the signals and confirm on the floor, then trial one with a stated false-alarm budget and a measured *e*.

## Reproduce

```bash
uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python
export OMP_NUM_THREADS=2
.venv/bin/python -I pipeline/run.py data/raw/processminer.csv src/data/results.json   # ~55 s
.venv/bin/python -I -m pytest -q tests
```
