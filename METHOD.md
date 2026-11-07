# Method

Rare-event precursor analysis on a real paper-machine sheet-break dataset. All numbers below are produced by `pipeline/run.py` and `pipeline/hybrid_run.py` (section 12) and exported to `src/data/results.json`; nothing is hand-entered.

## 1. Data

- **Source:** Ranjan, C., Reddy, M., Mustonen, M., Paynabar, K., Pourak, K. (2018). *Dataset: Rare Event Classification in Multivariate Time Series.* arXiv:1809.10717. Data from a pulp-and-paper mill, released by ProcessMiner Inc.
- **Licence:** CC BY-NC-SA 4.0. This repo is non-commercial; the raw CSV is not redistributed here (see `DATA.md` for how to obtain it). Derived results in `src/data/` are shared under the same licence.
- **Shape (measured):** 18,398 rows, 1999-05-01 00:00 to 1999-05-29 00:06, nominal 2-minute sampling, 59 continuous anonymised sensors, one 8-level categorical (`x28`, treated as a grade code and one-hot encoded) and one binary (`x61`).
- **Events:** 124 rows with `y = 1`. Every one is an isolated row, so 124 break events (consecutive `y = 1` rows would be merged into one event). Break rows are 0.674% of all rows.
- **Gaps:** 107 timestamps are more than 2 minutes after the previous one. 103 of them start within 2 rows after a break. They are read here as post-break downtime removed from the record (section 10); this is an interpretation, since the dataset does not label the gaps. Rolling features are defined in rows, not clock time; this is an assumption.
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

- **Split:** time-blocked, no shuffling. First 75% of rows (to 1999-05-22 06:02) train, last 25% (about 6.8 days, 38 breaks) test. 5 of the 38 fall inside the restart mask of the previous break, so PR-AUC at the 2-minute horizon scores 33 of them.
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

Follow-on breaks are a restart-stability problem that break timing alone predicts well (Hawkes 7× prevalence). First breaks of an episode are the hard, valuable problem: hazard models are below prevalence there by construction, and the only score above prevalence is the sensor GBM, at 1.36×. This rests on 19 first-in-episode test breaks and has no confidence interval here; section 12 re-tests it on 53 first breaks with forward-chaining folds and bootstrap CIs, where it holds. Adding time-since-break to the GBM does not help first breaks.

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
| **Clustered** null calibration: shift whole real episodes, spacing kept, to random nominal positions in their block (20 runs) | an honest test gives ~0 survivors | break-level (independent) test: **7 false survivors in 1 of 20 runs**, 4.7% of p < 0.05. Episode test: 0 survivors in 20 of 20 runs, 3.3% of p < 0.05. Note: 1 run in 20 with a false survivor is within what BH at 0.05 allows under a complete null, so this does not show the break-level test is miscalibrated; it shows that when it fails under clustering it fails in bulk. 20 runs cannot separate the two tests' error rates. |
| Weibull fit vs empirical CDF of inter-break times | KS not rejected | KS D = 0.105, p = 0.29 (exponential: D = 0.156, p = 0.03, rejected). Parameters are fitted on the same intervals, so these KS p-values are optimistic (Lilliefors effect): the exponential rejection holds a fortiori, the Weibull "not rejected" is weaker than p = 0.29 suggests |
| Hawkes stationarity | branching ratio α/β < 1 | 0.40 |
| Downtime extraction | equals hand count | 130, 24, 2 min on the first 3 breaks (unit test against the raw CSV) |
| Event-level alarm logic, break-even, Hawkes log-likelihood, Weibull MLE | hand calculations / simulated recovery | unit tests (`tests/test_reliability.py`); Weibull MLE recovers k = 0.8 on 20,000 draws, Hawkes fit recovers branching 0.5 on a simulated process |
| Leakage check: shift the 20-min label window one row further from the break | PR-AUC must not jump | 0.134 → 0.110 (drops, no jump) |
| Baselines | every model number next to prevalence and persistence | table in section 4 |
| BH implementation | equals `scipy.stats.false_discovery_control` | unit test |
| Past-only features and labels | future edits do not change past rows | unit tests |

`pytest -q tests`: 36 passed (25 here plus 11 for section 12 in `tests/test_hybrid.py`, including a hand-worked matched-budget alarm, warning and false-alarm example, a hand-worked check of the chance null, a hand-worked superposition of two Poisson sources (0.2 and 0.5 give 0.6), and a check that the superposed hybrid with a constant background reproduces the full Hawkes model).

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

  MLE μ = 0.0017/min (one background break per 10 h), branching ratio α/β = **0.40** (in a stationary Hawkes process, the expected share of breaks triggered by an earlier break; point estimate on 86 breaks, no CI computed), excitation half-life ln2/β = 204 min.
- **AIC:** exponential 1169.7, Weibull 1168.9, Hawkes **1160.3**. Of these three models, self-excitation describes how breaks arrive best (ΔAIC ≈ 9).

**Scores.** At row *t*, P(break in (t, t+h]) is `1 − S(s+h)/S(s)` for the Weibull model (s = minutes since the last break) and `1 − exp(−[μh + (α/β) E(t)(1 − e^(−βh))])` with `E(t) = Σ_{t_j ≤ t} e^(−β(t − t_j))` for Hawkes. Both use past break times only.

**Operational rule (no ML).** After a break, P(another break within 2 h) = **42%** against 29% for a random 2-hour window. Follow-on breaks come a median 65 minutes after the previous one; 80% within 94 min and 90% within 103 min. A **105-minute post-restart watch window** covers 90% of follow-on breaks on the training weeks.

## 10. Downtime and alarm economics

**Downtime (estimated from timestamp gaps).** Assumption: a timestamp gap right after a break is the line being down and the rows removed from the record. The data does not label this. For each break, downtime = the first timestamp gap within 2 rows after the onset, minus the nominal 2 minutes. 103 of 124 breaks are followed by such a gap (103 of the 107 gaps in the file). Median 14 min when present (IQR 8 to 30), mean **20.3 min per break** over all breaks, **41.9 hours in total** over 28 days: 1.5 line-hours a day, 6.2% of calendar time. First breaks of an episode cost more (23.2 min) than follow-ons (16.7 min).

**Event-level alarms.** Each score's threshold is set on the train weeks to give 0.5, 1 or 2 false alarms per 8-hour operating shift (GBM thresholds from blocked 3-fold out-of-fold scores, so the threshold is not set on in-sample fits). An alarm is a rising edge of score > threshold; it is true if a break follows within 20 minutes. A break counts as **warned** if the alarm is on between 20 and 10 minutes before it (≥ 10 minutes of lead time). Results on the 38 test breaks (19 episodes), target 1 false alarm per shift:

| Score | Breaks warned | First-in-episode | Follow-on | Realised false alarms / shift (test) | Alarm on, share of operating time (test) | Median lead |
|---|---|---|---|---|---|---|
| GBM | 5 / 38 (13%) | 2 / 19 | 3 / 19 | 1.5 | 2.6% | 14 min |
| GBM + time since break | 4 / 38 (11%) | 1 / 19 | 3 / 19 | 1.6 | 3.0% | 18 min |
| Hawkes hazard (no sensors) | **21 / 38 (55%)** | 3 / 19 | **18 / 19** | 1.6 | 29.5% | 20 min |
| `x6` level rule (fixed threshold)* | 9 / 38 (24%) | 1 / 19 | 8 / 19 | 0.8 | 3.3% | 20 min |

**Read with care.** False alarms are counted as rising edges, so an alarm that stays on for hours counts once. At the same false-alarm count the Hawkes alarm is on for 29.5% of operating time and the GBM for 2.6%. Per unit of alarm-on time the GBM warns more breaks (13% recall from 2.6% on-time, against 55% from 29.5%). Which score is "better" depends on whether an operator's cost scales with alarm count or alarm duration. That is not measured here.

\* `x6` level was chosen as the top rule by rolling-origin scoring that includes the test weeks, and all 9 of its warned test breaks fall in the single 24–25 May episode (14:54 to 00:46) that made it the top rule. Its threshold is out of sample but its selection is not, so this row is optimistic.

At 2 false alarms per shift the Hawkes score warns 28 of 38 (74%), the GBM 8 of 38, but the Hawkes alarm is then on for 64% of operating time (GBM 4.4%). Thresholds drift: a target of 1 per shift on train weeks became 1.5 to 1.6 per shift on test weeks.

**Break-even.** With *e* = the share of warned breaks an operator actually prevents (not measurable from this data):

    line-minutes saved per day = recall × breaks/day × downtime per break × e − false alarms/day × check time
    break-even e = false alarms/day × check time / (recall × breaks/day × downtime per break)

Inputs: 5.6 breaks/day (test weeks), 20.3 min per break (measured), 5 min per alarm check (assumed). At 1 false alarm per shift:

| Score | Break-even e | Net line-hours / month at e = 10% | 25% | 50% |
|---|---|---|---|---|
| GBM | 1.50 (never pays) | −10.5 | −9.4 | −7.5 |
| Hawkes hazard alarm | 0.38 | −9.0 | −4.2 | **+3.7** |
| `x6` level rule* | 0.43 | −4.5 | −2.5 | +0.9 |

No score pays for itself unless operators can stop a large share of the breaks they are warned about. Under this cost model (5 minutes per alarm, regardless of how long it stays on) the sensor GBM never does at this alarm budget. The cost model favours the Hawkes alarm, which is on 11 times longer than the GBM's (see above). If check cost scales with alarm-on time, the ranking may change; that variant is not computed. The `x6` row is optimistic (see the note above). Line-hours convert to money at the plant's own cost per hour; no dollar figure is assumed here.

## 11. Week 1 on a new line

What these numbers say a forward-deployed data scientist should do first:

1. **Day 1: measure the cost and the clustering from the historian.** Post-break downtime per break (here 20 min, 1.5 h/day) and the break hazard (here k = 0.87, branching 0.40). This sizes the prize before any model.
2. **Trial a post-restart watch window as a zero-ML rule.** Here: after any break, run with extra attention for 105 minutes (the train-week 90th percentile of follow-on spacing). The fixed window itself was not scored on test weeks. The closest scored rule, the Hawkes hazard alarm, caught 18 of 19 test follow-on breaks at about 1.6 false alarms per shift, while staying on for 30% of operating time. Measure what the extra attention costs before rolling it out.
3. **Point the learned model at the breaks history cannot foresee, and judge it against chance.** Boost a sensor model from the break hazard (section 12) and score it on first-in-episode breaks with a threshold-free metric and, for alarms, a chance null at the same alarm-on time. Here the sensor signal on first breaks is small (PR-AUC 1.3× to 1.6× the base rate) and the hybrid's warned-break count beats chance at only one of four budgets.
4. **Screen detectors on first-in-episode breaks only, with an episode-aware null.** Break-level screening produced a survivor that is one bad evening; the clustered calibration shows the break-level test can produce 7 false survivors in a run.
5. **Bring the top episode-level candidates (here `x51` level, `x26`/`x16`/`x13` deviation) to the process engineer** to name the signals and confirm on the floor, then trial one with a stated false-alarm budget and a measured *e*.

## 12. Hazard + sensors: where the learned model earns its place

Sections 4 and 10 left two loose ends: the sensor model never beat the 3-parameter hazard overall, and the alarm comparison was at a matched false-alarm *count* that left the hazard alarm on for 29.5% of the time and the GBM for 2.6%. This section addresses both, and shows that matching alarm-on time brings its own bias. Code: `pipeline/hybrid.py`, `pipeline/hybrid_run.py`; tests: `tests/test_hybrid.py`.

**Hybrid model.** The hazard knows only break history. The learned part starts from it and adds sensor corrections:

    logit P(break in (t, t+h] | history, sensors) = logit p_Hawkes(t) + f(x_t)

where `p_Hawkes(t) = 1 − exp(−[μh + (α/β) E(t)(1 − e^(−βh))])` (section 9) and `f` is a LightGBM ensemble trained with the hazard log-odds as `init_score`, so every tree fits the *residual* of the hazard. With no trees `f = 0` and the hybrid equals the hazard (unit test). `x_t` is the same 531 past-only sensor features plus grade as section 4. Hyperparameters were not tuned (there is no search in the code; the order in which settings were chosen cannot be verified from the repository): 200 trees, learning rate 0.03, 7 leaves, min 60 rows per leaf, 30% of features per tree, L2 = 5, no class weighting (so the offset stays a probability).

Compared on identical folds:

| Score | What it uses |
|---|---|
| Hawkes hazard | 3 parameters, break history only |
| Sensor GBM | same LightGBM settings, sensors only, no history |
| **Hybrid GBM** | LightGBM boosted from the Hawkes log-odds |
| Hybrid superposed | Hawkes excitation (μ = 0) + a sensor-learned background intensity |
| Hybrid logistic | L2 logistic (C = 0.01) on sensors + Hawkes log-odds |

**Second hybrid form: superposed intensities.** In the Hawkes model the background rate μ is the part history cannot explain, and it is exactly what drives first breaks. The superposed hybrid replaces the constant μ with a sensor-learned background: a LightGBM `g(x)` trained only on quiet training rows (≥ 2 h since the last break, whose next break is by definition first-in-episode), added to the excitation term as a second Poisson source:

    Λ(t) = (α/β) E(t)(1 − e^(−βh)) + Λ_bg(x_t),   P = 1 − e^(−Λ) = 1 − (1 − p_exc)(1 − g(x_t))

With `g = 1 − e^(−μh)` this is the plain Hawkes model again (unit test).

**Validation design (no leakage).** 5 contiguous blocks of about 5.6 days; each inner boundary is moved to the nearest row at least 120 minutes from every break so no episode is split (here the equal-size boundaries already satisfied this). Forward chaining: blocks 2 to 5 are each scored by models trained only on earlier blocks, with the last *h* rows before the test block purged so no training label can see a test break. The Hawkes parameters are refitted on each training prefix (branching 0.38 to 0.61 across folds). Scores at test time use only break times up to *t*, which a live system would know. Scored: 14,028 eligible rows, **89 breaks: 53 first-in-episode, 36 follow-on**, 58.5 operating shifts.

**Matched alarm budgets.** Each score gets the same alarm-on time: in each test block the alarm is on for exactly the top *b* share of eligible rows by score (label-free; ties broken by row order). A break is warned if the alarm is on between 20 and 10 minutes before it. The top-*b* cut uses the test block's own score distribution (no labels), so it is not a causal threshold; a live system would set it from earlier data and its on-time would drift from *b*.

**Chance null for warned breaks.** A warning window is only 6 rows, so an alarm that flickers hits windows by luck far more often than one long block with the same on-time. For each score and budget, 500 draws rotate the score by a random offset inside each block (over its eligible rows), which keeps the alarm's on-time and on/off run structure and removes its alignment with breaks; p is the share of draws that warn at least as many breaks as observed. A row-iid random score is also reported (`hybrid.horizons.*.chance` in the JSON).

**Chance null for first-break PR-AUC.** The same per-block rotation, 1,000 draws, with PR-AUC recomputed on the first-break subset and on quiet-regime rows; p = (1 + #draws ≥ observed) / 1,001 (`hybrid.horizons.*.ap_chance`). This is the fair reference for the task history cannot do: it keeps each score's autocorrelation and the label prevalence and removes only the alignment.

**Uncertainty.** PR-AUC differences: paired bootstrap over ~1-day blocks of test rows (720 rows), 2,000 resamples. Recall differences: paired bootstrap over episodes (all breaks of an episode resampled together), 2,000 resamples.

### Results, 20-minute horizon

PR-AUC (prevalence: all 0.056, first-break subset 0.039, follow-on subset 0.019, quiet-regime rows 0.046):

| Score | All breaks | First-in-episode | Follow-on | Quiet regime (≥ 2 h since last break) |
|---|---|---|---|---|
| Hawkes hazard | 0.102 | 0.031 | **0.148** | 0.044 |
| Sensor GBM | 0.096 | **0.061** | 0.044 | **0.076** |
| **Hybrid GBM** | **0.121** | 0.050 | 0.126 | 0.073 |
| Hybrid superposed | 0.114 | 0.050 | 0.098 | 0.069 |
| Hybrid logistic | 0.049 | 0.033 | 0.017 | 0.039 |

| Paired difference (95% CI, day-block bootstrap) | All | First-in-episode | Follow-on |
|---|---|---|---|
| Hybrid − Hawkes | **+0.019 (+0.002 to +0.043)** | **+0.020 (+0.012 to +0.034)** | −0.022 (−0.054 to +0.024) |
| Sensor GBM − Hawkes | −0.006 (−0.050 to +0.050) | **+0.031 (+0.016 to +0.063)** | −0.104 (−0.164 to +0.002) |
| Hybrid − sensor GBM | +0.025 (−0.017 to +0.061) | −0.011 (−0.035 to +0.001) | **+0.082 (+0.003 to +0.135)** |
| Superposed − hybrid GBM | −0.007 (−0.023 to +0.013) | −0.000 (−0.016 to +0.011) | −0.027 (−0.064 to +0.022) |

In the quiet regime (≥ 2 h since the last break, 80% of test rows) the hybrid beats the hazard by +0.029 PR-AUC (CI +0.013 to +0.055) and the sensor-only GBM by +0.033 (CI +0.012 to +0.071).

**First-break PR-AUC against chance** (1,000 per-block rotations of each score):

| Score | First-break PR-AUC | Rotation-null mean (95th pct) | p | Quiet-regime PR-AUC | null mean | p |
|---|---|---|---|---|---|---|
| Hawkes hazard | 0.031 | 0.042 (0.054) | 1.00 | 0.044 | 0.052 | 0.80 |
| **Sensor GBM** | **0.061** | 0.039 (0.047) | **0.001** | **0.076** | 0.047 | **0.002** |
| **Hybrid GBM** | **0.050** | 0.041 (0.050) | **0.049** | **0.073** | 0.049 | **0.003** |
| Hybrid superposed | 0.050 | 0.042 (0.052) | 0.085 | 0.069 | 0.052 | 0.029 |

At 60 minutes no score is above its null on first breaks (sensor GBM 0.122 vs 0.111, p = 0.14; hybrid p = 0.50).

Breaks warned ≥ 10 min ahead at matched alarm-on time:

| Alarm on | Score | First-in-episode (of 53) | Follow-on (of 36) | All (of 89) | False alarms / shift |
|---|---|---|---|---|---|
| 5% | Hawkes | 0 | 13 | 15% | 0.4 |
| 5% | Sensor GBM | 9 | 11 | 22% | 2.9 |
| 5% | **Hybrid GBM** | **7** | **14** | **24%** | 2.9 |
| 10% | Hawkes | 1 | 20 | 24% | 0.6 |
| 10% | Sensor GBM | 16 | 19 | 39% | 4.9 |
| 10% | **Hybrid GBM** | **15** | **21** | **40%** | 5.0 |
| 20% | Hawkes | 6 | 25 | 35% | 0.9 |
| 20% | Sensor GBM | 24 | 25 | 55% | 8.2 |
| 20% | Hybrid GBM | 20 | 24 | 49% | 8.4 |

| Hybrid − Hawkes recall (episode bootstrap, 95% CI) | First-in-episode | Follow-on | All |
|---|---|---|---|
| 5% alarm-on | **+13 pts (+6 to +23)** | +3 pts (−17 to +24) | +9 pts (0 to +20) |
| 10% alarm-on | **+26 pts (+15 to +40)** | +3 pts (−18 to +29) | **+17 pts (+5 to +31)** |
| 20% alarm-on | **+26 pts (+13 to +40)** | −3 pts (−23 to +17) | **+15 pts (+4 to +27)** |

These recall gaps compare a flickering alarm with a blocky one at equal on-time; read them with the chance null:

| Alarm on | First-in-episode warned: Hawkes / sensor GBM / hybrid (of 53) | Rotation-null mean (Hawkes / sensor / hybrid) | p (sensor / hybrid) | Row-iid random score |
|---|---|---|---|---|
| 2% | 0 / 5 / 2 | 1.3 / 2.8 / 2.4 | 0.15 / 0.71 | 6.1 |
| 5% | 0 / 9 / 7 | 3.0 / 5.7 / 5.3 | 0.08 / 0.27 | 14.4 |
| 10% | 1 / 16 / 15 | 5.7 / 10.0 / 9.7 | **0.03 / 0.03** | 25.2 |
| 20% | 6 / 24 / 20 | 11.2 / 17.3 / 17.8 | **0.03** / 0.29 | 39.6 |

The Hawkes alarm warns fewer first breaks than its own rotated copy at every budget (p ≥ 0.98): it is on only right after breaks, and first breaks come at least 2 h after one, so it is structurally unable to reach them. Its 0 to 6 of 53 is not a chance-level reference, and the "hybrid − Hawkes" first-break recall CIs above measure that structural gap, not sensor skill. All three alarms are above their nulls on follow-on breaks at 5% to 20% (p ≤ 0.02).

**Reading.**
1. **This is where the learned model earns its place: first breaks, which history cannot foresee.** On first-in-episode breaks the hazard is below prevalence (0.031 vs 0.039) and below its own rotation null (p = 1.00): break history carries no information about when a quiet line will break. The learned sensor model ranks those 20-minute windows at 1.6× the base rate (0.061) and clearly above chance (rotation null 0.039, p = 0.001), and the same holds over all quiet-regime rows (0.076 vs 0.047, p = 0.002). The hybrid is above chance too, more weakly on first breaks (0.050, p = 0.049) and clearly on quiet rows (0.073, p = 0.003). For alarms, the hybrid's 15 of 53 warned first breaks at 10% alarm-on time is above its chance null (mean 9.7, p = 0.03), but its counts at 2%, 5% and 20% are not; the sensor-only GBM clears the null at 10% and 20%. The signal is real but small and short-range (gone at 60 minutes).
2. **For follow-on breaks the hazard suffices.** The hybrid ties it (difference CI spans zero), and the sensor-only GBM is far worse there (0.044 vs 0.148). Sensors add nothing measurable once the line is in an episode.
3. **The hybrid is the best single score on point estimates, not a proven winner.** It is the only score whose all-break PR-AUC beats the hazard with a CI excluding zero (+0.019, CI +0.002 to +0.043, lower end near zero), and it has the highest all-break point estimate; its edge over the sensor-only GBM on all breaks is not significant (+0.025, CI −0.017 to +0.061). It gets there because it keeps the hazard's follow-on skill (+0.082 over sensor-only on follow-ons) while adding most of the sensor lift on first breaks (it gives back 0.011 to the sensor-only GBM there, CI −0.035 to +0.001).
4. **Superposing a sensor-learned background on the Hawkes excitation does not beat boosting from the log-odds** (all-break PR-AUC 0.114 vs 0.121, difference CI −0.023 to +0.013; first breaks tie at 0.050). Training the sensor term only on quiet rows did not buy extra first-break skill over the sensor-only GBM (−0.011, CI −0.039 to +0.004). Kept as a reported negative result.
5. **Seed check:** hybrid PR-AUC on all breaks 0.121 / 0.119 / 0.127 and on first breaks 0.050 / 0.051 / 0.050 over seeds 11, 12, 13; superposed hybrid on first breaks 0.050 / 0.050 / 0.056.

**What does not win (reported, not hidden).**
- **Absolute first-break skill is low:** PR-AUC 0.050 against a 0.039 prevalence (1.3×); 0.061 (1.6×) for the sensor-only model. Above chance is not the same as useful. At 10% alarm-on time the hybrid catches 28% of first breaks but raises about 5 rising-edge false alarms per shift. That is a screening signal for a process engineer, not an operator alarm.
- **False-alarm counts favour the hazard** at every budget (0.4 to 0.9 per shift against 2.9 to 8.4): its alarm is one long block after each break, the sensor alarms flicker. Matched on-time and matched alarm count give opposite pictures; section 10 showed the other one.
- **At 60 minutes the hazard wins overall** (PR-AUC 0.276 vs hybrid 0.219 vs sensor GBM 0.168) and the first-break lift is no longer significant (hybrid − Hawkes +0.011, CI −0.007 to +0.037). The hybrid's follow-on PR-AUC falls well below the hazard's (0.220 vs 0.476): at the longer horizon the learned corrections overfit and damage the hazard. The sensor signal is short-range.
- **Matched alarm-on time favours flickering alarms.** A row-iid random score at 10% on-time warns about 25 of 53 first breaks, more than any model; the model counts only mean something against the rotation null above. Of the 8 first-break alarm comparisons (2 learned models × 4 budgets) 3 clear it at p < 0.05, with no multiplicity correction; 10% is the budget where both do.
- **Many comparisons.** 4 budgets × 2 horizons × 3 break subsets × 4 score pairs are reported; the CIs are per comparison and not adjusted.
- **The restart mask length (7 rows) was measured in section 4 on the first 75% of the data,** which overlaps test blocks 2 to 4 here. It uses break times and sensor transients, not scores, and applies equally to every model.
- **The hybrid logistic model fails** (all-break PR-AUC 0.049, below prevalence): a linear model on 531 correlated features cannot use the sensors and drowns the hazard term.
- On first breaks the sensor-only GBM is slightly ahead of the hybrid at every budget (not significant at 5% or 10%; at 20% alarm-on the gap is 4 first breaks). If a plant wants a first-break screen only, the sensor-only model is enough; the hybrid is the better single score.
- The forward-chaining test period differs from section 4's single 25% split, so absolute PR-AUCs differ (Hawkes 0.102 here vs 0.166 there); comparisons are only made within a table.

## Reproduce

```bash
uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python
export OMP_NUM_THREADS=2
.venv/bin/python -I pipeline/run.py data/raw/processminer.csv src/data/results.json   # ~55 s
.venv/bin/python -I pipeline/hybrid_run.py data/raw/processminer.csv src/data/results.json   # ~6 min, adds the `hybrid` section
.venv/bin/python -I -m pytest -q tests
```
