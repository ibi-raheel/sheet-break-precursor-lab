# Sheet-break precursor lab

Which sensor behaviours actually come before a paper-machine sheet break? A reproducible rare-event analysis on 124 real breaks: forward-looking labels with restart masking, models across lookahead horizons with honest baselines, single-signal detectors scored by lift against a permutation null, and a measured check of whether model importance (SHAP) agrees with lift.

**Live demo:** _URL added after deploy_

## The problem
On a continuous line, failures like sheet breaks are rare (124 in four weeks here, 0.67% of rows), expensive and clustered. A classifier trained on a few dozen events gives a weak score that operators cannot act on, and the signals a model ranks highest are not necessarily the ones that precede a failure. A plant needs a short list of operator-readable rules, each with a measured lift and a test that it is not noise.

## What it finds (measured, see METHOD.md)
- **Prediction horizon:** at 20 minutes ahead, gradient boosting reaches PR-AUC 0.134 against 0.076 for prevalence and 0.106 for a "breaks cluster" persistence baseline. At 2 to 4 minutes it does no better than persistence.
- **Detectors:** 1 of 175 single-signal rules survives a 5,000-draw permutation test with Benjamini-Hochberg at q < 0.05 across 74 out-of-sample breaks: a low-level rule with lift 9.8, firing 14 to 20 minutes before 8 breaks at 0.8 false alarms per day.
- **Importance is not lift:** none of the 10 highest-SHAP signals survives; the survivor is SHAP rank 33 of 59. Spearman correlation between SHAP importance and best lift is 0.08.
- **Validation:** hand-calculated lift matches the pipeline; with fake break times, 0 detectors survive in 20 of 20 runs; shifting labels one step away from the break lowers PR-AUC (0.134 to 0.110), so no leakage jump.

## How it works
- `pipeline/load.py`: parse, break onsets, sampling gaps
- `pipeline/label.py`: forward labels for horizon *h*; restart mask length measured from the post-break transient (7 rows)
- `pipeline/features.py`: past-only rolling mean, std, slope and deviation from rolling median (10 and 30 min)
- `pipeline/run.py`: time-blocked LightGBM vs logistic vs baselines at 6 horizons; TreeSHAP; 175 level/slope/deviation detectors with rolling-origin thresholds; lift, permutation p-values, BH; null calibration; signal families by hierarchical clustering; JSON export
- `pipeline/stats.py`: lift, precision, permutation p-value, Benjamini-Hochberg
- `tests/`: 11 pytest tests, including the hand calculation
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
