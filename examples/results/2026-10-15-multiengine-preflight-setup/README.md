# Recursive tree-engine timing preflight launched

[GitHub Actions run 38775613523](https://github.com/imh-ds/open/actions/runs/38775613523)
was dispatched once from ffe85ca57112530871a8ab9e9db7ec5ca8fdf9b2.
Preparation passed the frozen-source and installed-runtime checks. Its saved
authorization matches exactly six cases: Random Forest, GBM and LightGBM, each
at N=80 and N=200. All six case jobs were created on the first attempt.

This measures runtime and checks recursive execution before a larger reliability
study. It cannot establish reliability from one fixture per engine/sample size.
Each case uses shipped defaults, including 100 resamples and repeated 5-fold CV.
The jobs can run in parallel. Reserved compute is 586 runner-minutes combined,
within the owner's new 600-minute ceiling; each case job has a 90-minute limit.
No retries or larger study are authorized automatically.

Software verification: 38 focused tests and 29 inherited checks passed.
The versioned auditor exactly reproduced a preserved ElasticNet case's full audit
row: 300 assessments, 307,814 observation checks, no refits. Independent review
caught and resolved interrupted-run handling defects; regression tests cover them.

After completion, preserve and independently audit all six raw case archives,
including incomplete attempts. Report runtime, correct/missed/extra links and
variable depths separately. Review timing and storage needs before proposing a
funded reliability study. The 990-dataset proposal remains unapproved.
