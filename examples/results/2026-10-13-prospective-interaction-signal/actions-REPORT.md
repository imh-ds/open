# Prospective interaction reliability

Stage signal: 10800/10800 planned datasets completed; 0 missing; 0 audit errors.
Missing and errored results are unknown. No incomplete stratum receives a support label.

This tests two fixed pairs with one frozen Gaussian spline method. It does not certify other engines, causal moderation or the full recursive network.
Curve-pattern support checks relative direction/pattern at common supported locations, not magnitude, pointwise accuracy or the full surface.

Skipped alternatives: 0; these are unassessed, not failed detections.
Predictive support requires low false alarms in both null classes and the alternative control pair, plus reliable true-pair detection.
Curve support adds evidence that displayed patterns match the known relative pattern; report display frequency separately.

| Condition | Benchmark | Detection | Displayed / planned | Predictive support | Pattern support |
|---|---|---:|---:|---|---|
| n300-p5-rho0 | matched-q0.01 | 49/450 (observed_rate_misses_target) | 35/450 (observed_rate_misses_target) | False | False |
| n300-p5-rho0 | matched-q0.04 | 338/450 (observed_rate_misses_target) | 272/450 (uncertain_despite_observed_rate_meeting_target) | False | False |
| n300-p5-rho0 | mixed-b0.8 | 447/450 (qualified) | 425/450 (qualified) | True | True |
| n300-p5-rho0 | pure-b0.8 | 450/450 (qualified) | 448/450 (qualified) | False | False |
| n300-p5-rho0.7 | matched-q0.01 | 14/450 (observed_rate_misses_target) | 13/450 (observed_rate_misses_target) | False | False |
| n300-p5-rho0.7 | matched-q0.04 | 227/450 (observed_rate_misses_target) | 210/450 (qualified) | False | False |
| n300-p5-rho0.7 | mixed-b0.8 | 89/450 (observed_rate_misses_target) | 81/450 (uncertain_despite_observed_rate_meeting_target) | False | False |
| n300-p5-rho0.7 | pure-b0.8 | 368/450 (uncertain_despite_observed_rate_meeting_target) | 347/450 (qualified) | False | False |
| n600-p5-rho0 | matched-q0.01 | 72/450 (observed_rate_misses_target) | 57/450 (uncertain_despite_observed_rate_meeting_target) | False | False |
| n600-p5-rho0 | matched-q0.04 | 436/450 (qualified) | 399/450 (qualified) | True | True |
| n600-p5-rho0 | mixed-b0.8 | 450/450 (qualified) | 447/450 (qualified) | True | True |
| n600-p5-rho0 | pure-b0.8 | 450/450 (qualified) | 450/450 (qualified) | False | False |
| n600-p5-rho0.7 | matched-q0.01 | 8/450 (observed_rate_misses_target) | 8/450 (uncertain_despite_observed_rate_meeting_target) | False | False |
| n600-p5-rho0.7 | matched-q0.04 | 348/450 (observed_rate_misses_target) | 333/450 (qualified) | False | False |
| n600-p5-rho0.7 | mixed-b0.8 | 95/450 (observed_rate_misses_target) | 92/450 (uncertain_despite_observed_rate_meeting_target) | False | False |
| n600-p5-rho0.7 | pure-b0.8 | 438/450 (qualified) | 433/450 (qualified) | True | True |
| n300-p15-rho0.7 | matched-q0.01 | 43/450 (observed_rate_misses_target) | 37/450 (observed_rate_misses_target) | False | False |
| n300-p15-rho0.7 | matched-q0.04 | 246/450 (observed_rate_misses_target) | 233/450 (qualified) | False | False |
| n300-p15-rho0.7 | mixed-b0.8 | 103/450 (observed_rate_misses_target) | 97/450 (uncertain_despite_observed_rate_meeting_target) | False | False |
| n300-p15-rho0.7 | pure-b0.8 | 364/450 (uncertain_despite_observed_rate_meeting_target) | 353/450 (qualified) | False | False |
| n300-p15-rho0.9 | matched-q0.002 | 1/450 (observed_rate_misses_target) | 0/450 (unresolved_no_observations) | False | False |
| n300-p15-rho0.9 | matched-q0.005 | 6/450 (observed_rate_misses_target) | 5/450 (observed_rate_misses_target) | False | False |
| n300-p15-rho0.9 | mixed-b0.8 | 2/450 (observed_rate_misses_target) | 0/450 (unresolved_no_observations) | False | False |
| n300-p15-rho0.9 | pure-b0.8 | 4/450 (observed_rate_misses_target) | 2/450 (uncertain_despite_observed_rate_meeting_target) | False | False |

Lower/upper bounds crossing their targets mean uncertainty, not evidence that the effect is absent. Quantitative curve magnitude is not validated by this pattern criterion. No universal minimum N follows from these fixed conditions.

![Condition-specific bounds](SUMMARY.png)
