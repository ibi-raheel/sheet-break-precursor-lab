# Sheet-break precursor lab

Which sensor behaviours actually come before a paper-machine sheet break? A reproducible rare-event analysis on 124 real breaks: forward-looking labels with restart masking, models across lookahead horizons with honest baselines, single-signal detectors scored by lift against a permutation null, and a measured check of whether model importance (SHAP) agrees with lift.

**Live demo:** _URL added after deploy_

## The problem
On a continuous line, failures like sheet breaks are rare (124 in four weeks here, 0.67% of rows), expensive and clustered. A classifier trained on a few dozen events gives a weak score that operators cannot act on, and the signals a model ranks highest are not necessarily the ones that precede a failure. A plant needs a short list of operator-readable rules, each with a measured lift and a test that it is not noise.

## What it finds (measured, see METHOD.md)
- **Breaks come in episodes, and that is the strongest signal in the data.** 124 breaks form 69 episodes (breaks < 2 h apart chained). Weibull shape k = 0.87 (95% CI 0.78 to 0.99): risk is highest right after a break. A self-exciting (Hawkes) model fits best by AIC; 40% of breaks are triggered by an earlier one.
- **A 3-parameter hazard model with no sensor data matches or beats the 531-feature gradient-boosting model at every horizon** (20 min PR-AUC 0.166 vs 0.134, prevalence 0.076). The GBM's edge over a plain "breaks cluster" baseline is +0.027 with a bootstrap 95% CI of −0.000 to +0.104.
- **First breaks are the hard problem.** On first-in-episode breaks hazard models fall below prevalence, and the sensor model is the only score above it (0.062 vs 0.046). Follow-on breaks are a restart problem: a hazard alarm warns 18 of 19 test follow-ons at least 10 minutes ahead at about 1.6 false alarms per shift.
- **The one break-level survivor is one bad evening.** 1 of 175 single-sensor rules survives a 5,000-draw permutation test with Benjamini-Hochberg across 74 out-of-sample breaks, but 7 of its 8 hits are in one 10-hour episode. Scored per episode it covers 2 of 50 (q = 0.58) and **0 of 175 rules survive**. A clustered null calibration shows why: the break-level test produced 7 false survivors in 1 of 20 runs, the episode test none.
- **Measured cost:** the timestamp gaps after breaks are cut-out downtime: 20 minutes per break, 41.9 hours over 28 days (1.5 line-hours a day). At 1 false alarm per shift the sensor model never pays back operator checks; the zero-ML watch window pays only if operators prevent at least 38% of the breaks they are warned about.
- **Importance is not lift:** none of the 10 highest-SHAP signals survives; Spearman correlation between SHAP importance and best lift is 0.08.
- **Validation:** hand-calculated lift and downtime match the pipeline; fake break times give 0 survivors in 20 of 20 runs; shifting labels one step away from the break lowers PR-AUC (0.134 to 0.110); Weibull passes a KS test where the exponential fails; Hawkes and Weibull fitters recover known parameters on simulated data.

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
