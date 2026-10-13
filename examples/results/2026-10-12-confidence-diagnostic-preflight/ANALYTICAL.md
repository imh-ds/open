# Confidence diagnostic construction and analytical evidence

This is a fresh development investigation. The completed study still failed its frozen global assurance gate; its thresholds and decisions remain intact.

The exact calculation exposes a validation-design limitation. At a hypothetical fixed 1% false-alarm rate, the old 200-dataset per-setting gate passes with probability 67.7%. If 20 strata were independent and all had that same fixed rate, their joint passing probability would be about 0.0405%. At a hypothetical 0.5% rate, joint passing would be 18.9%. These are prospective assumptions, not estimated error rates for the completed study. Actual rates vary and calibration thresholds are random.

| Null datasets per setting | Joint gate-passing probability at hypothetical 1% error |
| --- | --- |
|200|0.0405%|
|400|65.7%|
|600|99.1%|
|1000|approximately 100%|

More null replications can make the same assurance requirement easier to certify without changing the error limit or cutoff. These calculations do not recommend a new accepted protocol by themselves.

The maximum-of-100 calibration reference gives expected future exceedance 1/101 under one specified continuous null. Across repeated calibration samples, the probability that their conditional future exceedance rate exceeds 5% is .95^100, about .00592. A conservative union bound across 20 classes exceeds .05; 117 per-class samples would bring that particular bound below .05. This is neither a posterior probability for the realized cutoff nor a confidence rule for arbitrary populations, and it does not revise historical thresholds.

The fresh probe holds distinguishable interaction information constant by adjusting the coefficient. Population reference levels .01/.04 are feasible for rho 0/.7 at overall R-squared .6; impossible targets are rejected. Better performance in matched-information cases would be consistent with an information-difference contribution, not conclusive proof of a single mechanism.

Seven helper tests passed after seven missing-feature RED witnesses. Independent review confirmed the formulas and identified a two-fold input invariant; a failing regression preceded the three-fold guard correction. The runner, workflow and native evidence are documented separately. No empirical detection findings from the complete fresh probe exist yet.
