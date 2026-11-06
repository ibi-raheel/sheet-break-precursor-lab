# Sheet-break precursor lab

Which sensor behaviours actually come before a paper-machine sheet break? A reproducible rare-event analysis on 124 real breaks: forward-looking labels with restart masking, models across lookahead horizons with honest baselines, single-signal detectors scored by lift against a permutation null, and a measured check of whether model importance (SHAP) agrees with lift.

**Live demo:** _URL added after deploy_

## The problem
On a continuous line, failures like sheet breaks are rare (124 in four weeks here, 0.67% of rows), expensive and clustered. A classifier trained on a few dozen events gives a weak score that operators cannot act on, and the signals a model ranks highest are not necessarily the ones that precede a failure. A plant needs a short list of operator-readable rules, each with a measured lift and a test that it is not noise.

## What it finds (measured, see METHOD.md)
- **Breaks come in episodes, and that is the strongest signal in the data.** 124 breaks form 69 episodes (breaks < 2 h apart chained). Weibull shape k = 0.87 (95% CI 0.78 to 0.99): risk is highest right after a break. A self-exciting (Hawkes) model fits best by AIC; its estimated branching ratio is 0.40 (roughly 40% of breaks triggered by an earlier one; no CI computed on this estimate).
- **A 3-parameter hazard model with no sensor data scores at least as high as the 531-feature gradient-boosting model at every horizon** (20 min PR-AUC 0.166 vs 0.134, prevalence 0.076). The differences are within bootstrap noise (GBM − Hawkes at 20 min: −0.033, 95% CI −0.068 to +0.045), so this is "the sensor model does not beat break timing", not "timing beats sensors". The GBM's edge over a plain "breaks cluster" baseline is +0.027 with a bootstrap 95% CI of −0.000 to +0.104.
- **First breaks are the hard problem.** On first-in-episode breaks hazard models fall below prevalence, and the sensor model is the only score above it, narrowly (0.062 vs 0.046, on 19 test episodes, no CI: suggestive, not established). Follow-on breaks are a restart problem: a hazard alarm warns 18 of 19 test follow-ons at least 10 minutes ahead at about 1.6 false alarms per shift, but it does so by staying on for 30% of operating time (the GBM alarm at the same false-alarm count is on 2.6% of the time).
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
- `tests/`: 25 pytest tests, including hand calculations for lift, downtime, Hawkes likelihood and break-even
- The web page reads the precomputed `src/data/results.json`; no API keys, no server.

## Run locally
```bash
uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/python -I pipeline/run.py data/raw/processminer.csv src/data/results.json
.venv/bin/python -I -m pytest -q tests
npm install
npm run dev
```

## Data
Real data, no simulation: Ranjan, C., Reddy, M., Mustonen, M., Paynabar, K., Pourak, K. (2018). *Dataset: Rare Event Classification in Multivariate Time Series.* [arXiv:1809.10717](https://arxiv.org/abs/1809.10717) (ProcessMiner Inc.). Licence CC BY-NC-SA 4.0; this project is non-commercial and derived results are shared under the same licence. The raw CSV is not redistributed; see [DATA.md](DATA.md).

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
