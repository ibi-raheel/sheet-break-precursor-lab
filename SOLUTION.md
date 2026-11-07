# Solution: a first-week playbook for rare, clustered failures

**The question.** With a few weeks of line data and a few dozen rare failures, which alarm rules can you trust? And how do you tell a real precursor from one bad evening, or from a signal the model only leans on?

**The measured answer** (124 real paper-machine sheet breaks, 28 days, 59 sensors; every number from `src/data/results.json`):

- The 124 breaks were really 69 episodes (breaks less than 2 h apart chained).
- Scored one break at a time, a permutation test passed 1 of 175 sensor rules, and that rule was one 10-hour evening (7 of its 8 hits). Counted once per episode, none of the 175 held.
- Break history alone (3 parameters) ranked follow-on breaks as well as any sensor model.
- On the first break of an episode, where history has no skill, the sensor model ranked the 20-minute windows at 1.6× the base rate (p = 0.001). That makes it a screening signal, not yet an alarm.
- None of the 10 signals the model leaned on most (SHAP) survived, and SHAP rank vs lift correlated at 0.08.

The rest of this page turns that into steps to test on a new line in its first week. They come from one public dataset and have not yet been validated on a production line.

## The playbook

1. **Size the prize before any model.** Measure downtime per event from the historian and fit the event hazard. Here: 20 min per break, 41.9 h over 28 days, 1.5 line-hours a day (inferred from timestamp gaps, which the dataset does not label). Hawkes branching ratio 0.40, Weibull shape k = 0.87.
2. **Split the events.** Chain events less than 2 h apart into episodes. Label each event first-in-episode or follow-on. Here: 69 and 55.
3. **Follow-ons: start with no ML.** Trial a post-restart watch window set to the 90th percentile of follow-on spacing (here 103 min), or the 3-parameter hazard alarm. Write down what the extra attention costs: the hazard alarm warned 18 of 19 holdout follow-ons, but it was on 30% of the time.
4. **First breaks: point the sensor model only here.** Judge it against a rotation chance null and require at least 10 minutes of lead. Treat it as a screening signal until precision supports an alarm.
5. **Detectors: score each episode once.** Drop rules that pass per event but fail per episode (one-episode artefacts). Do not read SHAP rank as cause. Take the episode-level shortlist to the process engineer, state a false-alarm budget, and measure the share of warned events that operators actually prevent.

## Decision rules and the thresholds measured here

| Decision | Rule | Measured on this line |
| --- | --- | --- |
| Do events cluster? | Hawkes beats exponential on AIC → split first vs follow-on | AIC 1160.3 vs 1169.7; branching 0.40; after a break, 42% chance of another within 2 h vs 29% for any 2-hour window |
| Episode gap | Events < 2 h apart are one episode (agree with operators) | 124 breaks → 69 episodes |
| Watch window after a restart | 90th percentile of follow-on spacing on training weeks | 103 min (median 65) |
| Follow-on alarm | Use the hazard / watch window; add sensors only if they beat it with a CI that excludes zero | Hazard PR-AUC 0.148 vs hybrid 0.126 vs sensor-only 0.044; hybrid − hazard CI −0.054 to +0.024 |
| First-break model | Keep only if PR-AUC beats a rotation chance null at p < 0.05, both at lead ≥ 0 and lead ≥ 10 min, and a grade-only model does not explain it | 0.061 vs base rate 0.0385 (p = 0.001); at ≥ 10 min 0.037 vs 0.023 (p = 0.014); grade-only p = 0.28 |
| Alarm or screen? | An alarm needs a false-alarm rate the shift accepts; otherwise it is a screening list | Sensor model at 10% alarm-on time: about 5 false alarms per shift → screening |
| Detector label | **Holds per episode** if episode-level BH q < 0.05 and lift > 1; **one-episode artefact** if only the break-level test passes; otherwise **noise** | 0 hold; 1 artefact (`x6` level: break-level q = 0.035, covers 2 of 50 episodes, episode q = 0.58); 174 noise |
| SHAP | Never treat a high-SHAP signal as a cause unless it also passes the detector test | 0 of the top 10 SHAP signals pass; Spearman SHAP vs lift 0.08 |

## The tool

`pipeline/triage.py` runs steps 1 to 5 on one CSV. It is a thin wrapper: every statistic comes from the pipeline functions behind METHOD.md (episodes, downtime, Weibull and Hawkes fits, restart mask, rolling features, break- and episode-level permutation tests with Benjamini-Hochberg, the TreeSHAP importance model, and the forward-chaining sensor model with its rotation null).

**Input:** a CSV with `time` (timestamp), `y` (0/1 event flag per row) and sensor columns `x1..xn`. The grade / categorical column is set with `--grade` (default `x28`).

```bash
.venv/bin/python -I pipeline/triage.py your_line.csv               # cost, clustering, routing, detectors, SHAP vs lift
.venv/bin/python -I pipeline/triage.py your_line.csv --model       # + first-break sensor-model check (slower)
# flags: --gap-min 120  --horizon-min 20  --perms 5000  --grade x28  --out triage.json
```

**Output:** a plain-text card on stdout and `triage.json` (with the per-event routing table and every rule's lift, break-level q and episode-level q). On the public data:

```
1. COST (inferred from timestamp gaps)
   124 events over 28.0 days; 103 followed by a gap of 107
   20.3 min per event, 41.9 h total, 1.5 line-hours a day
2. CLUSTERING (fit on the first 75% of the record)
   Weibull k 0.87 (CI 0.78 to 0.99), Hawkes branching 0.40
   watch window (90th pct of follow-on spacing): 103 min
   -> clustered: split first vs follow-on
3. ROUTING  124 events -> 69 episodes (gap 120 min)
   69 first-in-episode -> sensor screening
   55 follow-on        -> hazard / watch window
4. DETECTORS  175 rules, 74 scored events, 50 episodes
   survive per event: 1   survive per episode: 0
   x6:level       lift 9.8  q 0.035  q_ep 0.58  ONE-EPISODE ARTEFACT: drop
5. SHAP vs LIFT
   10 of the top-10 SHAP signals fail the test (model leans on it; does not precede events)
   Spearman SHAP vs best lift: 0.08
6. FIRST-BREAK MODEL (sensor GBM, rotation chance null)
   lead >= 0      PR-AUC 0.061 vs base rate 0.038 (1.6x), p = 0.001
   lead >= 10 min PR-AUC 0.037 vs base rate 0.023 (1.6x), p = 0.011
   -> first-break signal above chance: screening only
```

The lead ≥ 10 min p-value differs slightly from METHOD.md (0.014) because the chance null draws a fresh set of 1,000 rotations. `tests/test_triage.py` checks that the tool reproduces the pipeline's numbers on the real data (124 events, 69 episodes, k, branching, 1 and 0 survivors, the `x6` label, SHAP vs lift, first-break PR-AUC).

## Week 1 on a new line

- **Day 1.** Pull 4 to 8 weeks of historian data and the event or downtime log for one line, and get it into the CSV shape. Run triage. Hand the production manager a cost card: line-hours lost per day to events, measured instead of estimated.
- **Day 2.** Agree the episode gap with the operators (is an event 90 minutes after a restart really a new one?). Route every event as first or follow-on. Propose the post-restart watch window for follow-ons, with its alarm-on cost written down. No ML yet.
- **Day 3.** Screen the detectors once per episode. Drop one-episode artefacts. List the high-SHAP signals that fail the test, so nobody acts on them as causes.
- **Day 4.** Walk the floor with the process engineer. Go through the episode-level shortlist, give each signal a physical name, and rule out grade and line-speed proxies. Set the false-alarm budget per shift together.
- **Day 5.** Hand over this playbook filled in with the line's numbers, plus a 2-to-4-week trial plan for one watch window and one detector, and how the share of warned events operators actually prevent will be measured.

## Limits

74 scored breaks is a small sample. Downtime is an interpretation of timestamp gaps. Sensor names are anonymised in the public data, so no rule can be tied to a physical cause here; on a real line that is step 4's job.
