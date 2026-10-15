# Optimized RF recursive validation

2/2 cases completed and passed independent arithmetic/count checks; 2/2 recovered the exact chain.
Historical prediction/decision equivalence verified: True

|N|Status|Compute minutes|Original minutes|Tuning fits|Original tuning fits|Archive bytes|Original archive bytes|Exact chain|Equivalent|
|---|---|---:|---:|---:|---:|---:|---:|---|---|
|80|complete|44.67|97.14|90000|144000|9084609|9135932|True|True|
|200|complete|71.92|106.26|90000|144000|21750108|21795440|True|True|

If both comparisons pass, the optimized engine retained the original predictions and recursive decisions on these two fixtures while reducing redundant fitting.
This validates implementation/equivalence and measures runtime. It does not establish false-positive rates, recovery probabilities or general sample-size requirements.
The fixed two-case denominator includes incomplete or failed outcomes. No automatic retry or replacement seeds. No larger study follows automatically.
