# Pair validation development results

Complete 2/240; audit errors 0.
No cutoff accepted. Unselected folds are unknown; complete and conditional gains differ.
Incidental counts cover shortlisted pairs only; they are not calibrated false-positive rates. See pair-fold coverage in summary.csv.
Failed tasks: 0. Cost includes 600 comparator fitting/tuning fits and 24 saved-setting comparator verification refits.

|Scenario|N|p|Complete /10|True complete pairs|True fixed / adaptive detail folds|
|---|---:|---:|---:|---:|---|
|noise|80|5|0|None|None / None|
|noise|300|5|0|None|None / None|
|noise|600|5|0|None|None / None|
|noise|300|15|0|None|None / None|
|additive|80|5|0|None|None / None|
|additive|300|5|0|None|None / None|
|additive|600|5|0|None|None / None|
|additive|300|15|0|None|None / None|
|pure|80|5|0|0|0 / 0|
|pure|300|5|0|0|0 / 0|
|pure|600|5|0|0|0 / 0|
|pure|300|15|0|0|0 / 0|
|mixed|80|5|1|1|0 / 3|
|mixed|300|5|0|0|0 / 0|
|mixed|600|5|0|0|0 / 0|
|mixed|300|15|0|0|0 / 0|
|correlated_additive|80|5|0|None|None / None|
|correlated_additive|300|5|0|None|None / None|
|correlated_additive|600|5|0|None|None / None|
|correlated_additive|300|15|0|None|None / None|
|correlated_mixed|80|5|0|0|0 / 0|
|correlated_mixed|300|5|1|1|0 / 3|
|correlated_mixed|600|5|0|0|0 / 0|
|correlated_mixed|300|15|0|0|0 / 0|

![Development gains and support](validation-summary.png)
