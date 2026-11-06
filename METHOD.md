# Method

Rare-event precursor analysis on a real paper-machine sheet-break dataset. All numbers below are produced by `pipeline/run.py` and exported to `src/data/results.json`; nothing is hand-entered.

## 1. Data

- **Source:** Ranjan, C., Reddy, M., Mustonen, M., Paynabar, K., Pourak, K. (2018). *Dataset: Rare Event Classification in Multivariate Time Series.* arXiv:1809.10717. Data from a pulp-and-paper mill, released by ProcessMiner Inc.
- **Licence:** CC BY-NC-SA 4.0. This repo is non-commercial; the raw CSV is not redistributed here (see `DATA.md` for how to obtain it). Derived results in `src/data/` are shared under the same licence.
- **Shape (measured):** 18,398 rows, 1999-05-01 00:00 to 1999-05-29 00:06, nominal 2-minute sampling, 59 continuous anonymised sensors, one 8-level categorical (`x28`, treated as a grade code and one-hot encoded) and one binary (`x61`).
- **Events:** 124 rows with `y = 1`. Every one is an isolated row, so 124 break events (consecutive `y = 1` rows would be merged into one event). Break rows are 0.674% of all rows.
- **Gaps:** 107 timestamps are more than 2 minutes after the previous one (the publishers removed some downtime). Rolling features are defined in rows, not clock time; this is an assumption.
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
- **Baselines:** prevalence (PR-AUC of a random score equals the positive rate) and persistence (score = minus rows since the last break, i.e. "breaks cluster").
- **Metric:** PR-AUC (average precision) on eligible test rows.

| Horizon | GBM | Logistic | Persistence | Prevalence |
|---|---|---|---|---|
| 2 min | 0.013 | 0.014 | 0.010 | 0.008 |
| 4 min | 0.019 | 0.019 | 0.020 | 0.015 |
| 6 min | 0.038 | 0.026 | 0.030 | 0.023 |
| 10 min | 0.053 | 0.041 | 0.050 | 0.038 |
| 20 min | 0.134 | 0.083 | 0.106 | 0.076 |
| 30 min | 0.191 | 0.123 | 0.170 | 0.112 |

Reading: skill over prevalence is small in absolute terms at every horizon. At 20 min the GBM is 1.76× prevalence and 1.26× the persistence baseline. At 2 to 4 minutes the model is no better than "breaks cluster". On this line, model scores alone would not make a usable alarm.

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

**Multiple testing.** Benjamini-Hochberg across all 175 detectors:

    q_(i) = min_{j ≥ i} ( p_(j) · m / j ),   m = 175

A detector **survives** when q < 0.05 and lift > 1.

## 6. Results

- **1 of 175 detectors survives:** `x6` level below its 1st-percentile threshold. Lift 9.76, coverage 8/74 breaks (10.8%), q = 0.035, 0.8 false alarms/day, alarm precision 0.23, first firing 14 to 20 minutes before the break.
- **Importance is not lift:** the top 10 SHAP signals contain **0** survivors. `x6` is SHAP rank 33 of 59. The highest-SHAP signal (`x15`) has best lift 3.2 and q = 0.24. Spearman correlation between a signal's SHAP importance and its best detector lift: **0.08**.
- **Families:** average-linkage hierarchical clustering on 1 − |Spearman ρ| (training rows), cut at 0.3 (|ρ| ≥ 0.7), gives 44 families over 59 signals. Most sensors are not redundant copies of each other.
- **Holdout-only view** (score block 4 only, 32 breaks, thresholds from blocks 1 to 3): 4 survivors (`x6` level, `x34` level, `x14` deviation and slope). The pooled rolling-origin result is stricter because a detector has to hold across three weeks.

## 7. Validation

| Check | Expected | Measured |
|---|---|---|
| Lift / precision hand calculation | equal | 9.763 = 9.763 (unit test) |
| Null calibration: fake break times (20 runs, real nominal windows, same detectors) | ~0 survivors; p roughly uniform; ≤5% below 0.05 | 0 survivors in all 20 runs; 3.3% of p < 0.05 (slightly conservative: ties at zero fires push mass to p = 1) |
| Leakage check: shift the 20-min label window one row further from the break | PR-AUC must not jump | 0.134 → 0.110 (drops, no jump) |
| Baselines | every model number next to prevalence and persistence | table in section 4 |
| BH implementation | equals `scipy.stats.false_discovery_control` | unit test |
| Past-only features and labels | future edits do not change past rows | unit tests |

`pytest -q tests`: 11 passed.

## 8. Assumptions and limitations

- Signals are anonymised, so no physical meaning is claimed for `x6` or any family. A plant engineer would name them.
- The `x6` survivor is driven partly by one cluster: 7 of its 8 covered breaks fall in one 10-hour episode (24 May 14:54 to 25 May 00:16). It is a real, repeatable precursor during that episode, not proof of a general mechanism. More weeks of data would be needed.
- 74 scored breaks is a small sample. A coverage of 10.8% is honest but low; most breaks here have no single-sensor precursor at 20 minutes.
- Rolling features are in rows; 107 timestamp gaps make a few windows span more than their nominal duration.
- Thresholds are fixed percentiles (1% / 99%), chosen before results. They were not tuned.
- Only single-signal rules are tested. Multi-signal rules and agent-written hypotheses are out of scope.

## Reproduce

```bash
uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python
export OMP_NUM_THREADS=2
.venv/bin/python -I pipeline/run.py data/raw/processminer.csv src/data/results.json   # ~25 s
.venv/bin/python -I -m pytest -q tests
```
