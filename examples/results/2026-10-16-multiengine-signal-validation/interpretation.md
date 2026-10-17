# Recursive matched-chain validation

Truth: X1 -> X2 -> Y, at depths 2 -> 1 -> 0. Z1/Z2 are unrelated.
All cases use shipped settings; this is linear-chain evidence only.

|Engine|N|Complete / 55|Unknown|Exact network+depth|Extra findings|Recovery 95% lower|Error 95% upper|Gate|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|gbm|80|55|0|31|10|0.423217|0.309049|STOP|
|gbm|200|55|0|53|2|0.874736|0.125264|STOP|
|lightgbm|80|55|0|23|9|0.286545|0.288030|STOP|
|lightgbm|200|55|0|48|5|0.755197|0.199540|STOP|
|random_forest|80|55|0|43|12|0.649898|0.350102|STOP|
|random_forest|200|55|0|49|6|0.777529|0.222471|STOP|

Missing/incomplete/audit-failed cases remain unknown and block a pass.
A pass applies only to this linear Gaussian chain, sample size, and shipped setup.
These per-cell confidence bounds do not provide family-wise error control.
