# Sheet-break precursor lab

**The question.** With a few weeks of line data and a few dozen rare failures, which alarm rules can you trust? And how do you tell a real precursor from one bad evening, or from a signal the model only leans on?

**The answer, measured on 124 real paper-machine sheet breaks:**
- The 124 breaks were really 69 episodes. Scored one break at a time, a permutation test passed 1 of 175 sensor rules, and that rule was one 10-hour evening. Counted once per episode, none of the 175 held.
- Break history alone (3 parameters) ranked follow-on breaks as well as any sensor model.
- On the first break of an episode, where history has no skill, the sensor model ranked the 20-minute windows at 1.6× the base rate (p = 0.001). That makes it a screening signal, not yet an alarm.
- None of the 10 signals the model leaned on most (SHAP) survived, and SHAP rank vs lift correlated at 0.08.

**The solution:** a five-step first-week playbook and a small tool, `pipeline/triage.py`, that runs it on one CSV: it sizes the downtime, routes every break (follow-on → hazard or watch window, first-in-episode → sensor screening) and labels every candidate detector "holds per episode", "one-episode artefact" or "noise". See **[SOLUTION.md](SOLUTION.md)**.

Underneath is a reproducible rare-event analysis: a hybrid model that boosts sensor corrections on top of a self-exciting break hazard, forward-looking labels with restart masking, models across lookahead horizons with honest baselines, single-signal detectors scored by lift against a permutation null, and a measured check of whether model importance (SHAP) agrees with lift.

**Live demo:** _URL added after deploy_

## The problem
On a continuous line, failures like sheet breaks are rare (124 in four weeks here, 0.67% of rows), expensive and clustered. A classifier trained on a few dozen events gives a weak score that operators cannot act on, and the signals a model ranks highest are not necessarily the ones that precede a failure. A plant needs a short list of operator-readable rules, each with a measured lift and a test that it is not noise.

## What it finds (measured, see METHOD.md)
- **The learned sensor model's job is the breaks history cannot foresee, and there it beats chance.** Scored on 89 out-of-sample breaks (53 first-in-episode, 36 follow-on) with forward-chaining, episode-safe folds: on first breaks the 3-parameter break hazard has no skill (PR-AUC 0.031, below the 0.038 base rate). That is structural: the hazard is a function of time since the last break, so it ranks the rows just after a break (negatives here) above the quiet rows where first breaks happen. A gradient-boosted sensor model ranks those 20-minute windows at 0.061, 1.6× the base rate and above a chance null that rotates its own score inside each block (null mean 0.039, p = 0.001; p = 0.003 on an independent set of 1,000 rotations). It holds with at least 10 minutes of lead (0.037 vs a 0.023 base rate, p = 0.014), and product grade alone does not explain it (grade-only model 0.045, p = 0.28). The hybrid (boosted from the hazard's log-odds) is borderline on first breaks: 0.050, p = 0.049 in one null run and 0.057 in another, and not above chance once the last 6 or 10 minutes before the break are left out (p = 0.069, 0.086). Over all quiet-regime rows (≥ 2 h since the last break) both are above chance (p = 0.002 and 0.003). The signal is small, short-range (gone at 60 minutes) and not yet an operator alarm.
- **"Breaks warned at matched alarm-on time" needs a chance baseline, and mostly fails it.** At 10% alarm-on time the hybrid warns 15 of 53 first breaks at least 10 minutes ahead and the hazard 1 of 53, but the hazard alarm is one block right after each break, which can never reach a first break, while the sensor alarms flicker (about 5 rising edges per shift) and hit short warning windows by luck. A random score at 10% on-time warns about 25 of 53. Against a null that rotates each model's own score inside its block (same on-time, same flicker, alignment with breaks removed), the hybrid's first-break count is above chance at 10% on-time (15 vs a null mean of 9.7, p = 0.03) but not at 2%, 5% or 20% (p = 0.71, 0.27, 0.29). The sensor-only model clears it at 10% and 20% (p = 0.03, 0.03). The +26-point gap over the hazard is real arithmetic but not a fair measure of skill.
- **For follow-on breaks the hazard is enough.** On breaks inside an episode the hybrid is within noise of the 3-parameter hazard (PR-AUC 0.126 vs 0.148, CI of the difference −0.054 to +0.024) and a sensor-only model is far worse on PR-AUC (0.044). Sensors add nothing measurable once the line is in an episode; the hybrid's value over sensor-only is that it keeps this follow-on ranking (+0.082, CI +0.003 to +0.135).
- **Put together, the hybrid has the highest all-break PR-AUC on point estimates, not a proven win.** All-break PR-AUC 0.121 vs 0.102 for the hazard alone (day-block bootstrap CI of the difference +0.002 to +0.043, close to zero; 0.119 to 0.127 across 3 seeds) and 0.096 for sensor-only (CI −0.017 to +0.061, not significant). In the quiet regime it beats the hazard by +0.029 (CI +0.013 to +0.055) but not the sensor-only model (−0.003, CI −0.026 to +0.011). A second hybrid form, a Hawkes excitation term plus a sensor-learned background intensity trained only on quiet rows, does not do better (0.114, CI of the difference −0.023 to +0.013) and is reported as a negative result.
- **What does not win:** absolute first-break skill is low (PR-AUC 0.061 sensor-only and 0.050 hybrid against a 0.038 base rate, and the hybrid's is not robustly above chance), the sensor alarms flicker (about 5 false alarms per shift at 10% alarm-on time, against 0.6 for the hazard alarm), the sensor-only model is as good as or better than the hybrid on first breaks alone, a logistic hybrid fails, and at a 60-minute horizon the hazard wins overall and the first-break lift is not significant. The sensor signal is short-range.
- **Breaks come in episodes, and that is the strongest signal in the data.** 124 breaks form 69 episodes (breaks < 2 h apart chained). Weibull shape k = 0.87 (95% CI 0.78 to 0.99): risk is highest right after a break. A self-exciting (Hawkes) model fits best by AIC; its estimated branching ratio is 0.40 (roughly 40% of breaks triggered by an earlier one; no CI computed on this estimate).
- **On its own, a sensor-only model does not beat break timing.** On a single 25% holdout, a 3-parameter hazard model with no sensor data scores at least as high as the 531-feature sensor-only gradient-boosting model at every horizon (20 min PR-AUC 0.166 vs 0.134, prevalence 0.076). This is why the hybrid starts from the hazard. The differences are within bootstrap noise (GBM − Hawkes at 20 min: −0.033, 95% CI −0.068 to +0.045), so this is "the sensor model does not beat break timing", not "timing beats sensors". The GBM's edge over a plain "breaks cluster" baseline is +0.027 with a bootstrap 95% CI of −0.000 to +0.104.
- **Matched false-alarm count vs matched alarm-on time.** At a matched false-alarm count (about 1.6 per shift) the hazard alarm warns 18 of 19 holdout follow-ons but stays on for 30% of operating time, against 2.6% for the sensor-only model. Matching alarm-on time instead (section 12 of METHOD.md) flips the picture toward the flickering sensor alarms, and a chance null shows much of that flip is luck; neither matching alone is a fair alarm comparison, and both are reported.
- **The one break-level survivor is one bad evening.** 1 of 175 single-sensor rules survives a 5,000-draw permutation test with Benjamini-Hochberg across 74 out-of-sample breaks, but 7 of its 8 hits are in one 10-hour episode. Scored per episode it covers 2 of 50 (q = 0.58) and **0 of 175 rules survive**. A clustered null calibration (whole real episodes shifted to random positions) is consistent with this: when the break-level test fails, it fails in bulk (7 false survivors in 1 of 20 runs, against 0 of 20 for the episode test). With 20 runs, that difference is not itself significant.
- **Estimated cost:** 103 of the 107 timestamp gaps start right after a break. Read as cut-out post-break downtime (an interpretation; the dataset does not label them), they add up to 20 minutes per break, 41.9 hours over 28 days (1.5 line-hours a day). With a cost of 5 minutes per alarm (rising edge) at 1 false alarm per shift, the sensor model never pays back operator checks and the Hawkes hazard alarm pays only if operators prevent at least 38% of the breaks they are warned about. That cost model ignores how long an alarm stays on, which flatters the Hawkes alarm (on 30% of the time vs 2.6% for the GBM).
- **Importance is not lift:** none of the 10 highest-SHAP signals survives; Spearman correlation between SHAP importance and best lift is 0.08.
- **Validation:** hand-calculated lift and downtime match the pipeline; fake break times give 0 survivors in 20 of 20 runs; shifting labels one step away from the break lowers PR-AUC (0.134 to 0.110); Weibull is not rejected by a KS test where the exponential is (both with fitted parameters, so the Weibull p is optimistic); Hawkes and Weibull fitters recover known parameters on simulated data.

## How it works
- `pipeline/load.py`: parse, break onsets, sampling gaps
- `pipeline/label.py`: forward labels for horizon *h*; restart mask length measured from the post-break transient (7 rows)
- `pipeline/features.py`: past-only rolling mean, std, slope and deviation from rolling median (10 and 30 min)
- `pipeline/run.py`: time-blocked LightGBM vs logistic vs baselines at 6 horizons; TreeSHAP; 175 level/slope/deviation detectors with rolling-origin thresholds; lift, break-level and episode-level permutation p-values, BH; independent and clustered null calibration; bootstrap CIs; signal families by hierarchical clustering; JSON export
- `pipeline/stats.py`: lift, precision, permutation p-value, Benjamini-Hochberg
- `pipeline/reliability.py`: episodes, Weibull and Hawkes break-hazard MLE, post-break downtime, moving-block bootstrap, event-level alarm scoring, break-even model
- `pipeline/hybrid.py`, `pipeline/hybrid_run.py`: hazard-offset LightGBM (sensor terms boosted from the Hawkes log-odds), a superposed-intensity hybrid (Hawkes excitation + sensor-learned background), a rotation chance null for first-break PR-AUC, forward-chaining episode-safe folds with purge, matched alarm-on budgets with a circular-shift chance null, first vs follow-on scoring, day-block and episode bootstrap CIs, seed check
- `pipeline/triage.py`: first-week triage CLI over the functions above (no new statistics); see SOLUTION.md
- `pipeline/lead_check.py`: robustness of the first-break result: PR-AUC and rotation null with only positives at least 2, 6 or 10 minutes ahead, and a grade-only model
- `tests/`: 38 pytest tests, including a check that the triage tool reproduces the pipeline's numbers on the real data, including hand calculations for lift, downtime, Hawkes likelihood, break-even, a matched-budget alarm and the superposed-intensity hybrid
- The web page reads the precomputed `src/data/results.json`; no API keys, no server.

## Run locally
```bash
uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/python -I pipeline/run.py data/raw/processminer.csv src/data/results.json
.venv/bin/python -I pipeline/hybrid_run.py data/raw/processminer.csv src/data/results.json
.venv/bin/python -I pipeline/lead_check.py data/raw/processminer.csv src/data/results.json   # after hybrid_run
.venv/bin/python -I pipeline/triage.py data/raw/processminer.csv --model   # the playbook card
.venv/bin/python -I -m pytest -q tests
npm install
npm run dev
```

## Data
Real data, no simulation: Ranjan, C., Reddy, M., Mustonen, M., Paynabar, K., Pourak, K. (2018). *Dataset: Rare Event Classification in Multivariate Time Series.* [arXiv:1809.10717](https://arxiv.org/abs/1809.10717) (ProcessMiner Inc.). Licence CC BY-NC-SA 4.0; this project is non-commercial and derived results are shared under the same licence. The raw CSV is not redistributed; see [DATA.md](DATA.md).

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
