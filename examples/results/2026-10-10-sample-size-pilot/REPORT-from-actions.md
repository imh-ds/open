# Sample-size feasibility pilot

Completed 11/12 planned datasets. Incomplete or absent: 1.

One dataset per condition: these results do not establish reliability or a minimum N.

| Case | Status | Seconds | Root recovered | Required edges | False root inputs | Skill loss |
|---|---|---:|---|---|---|---:|
| noise-n80-p5 | complete | 1185.0 | True | 0/0 | 0 | -0.20976024192494203 |
| noise-n300-p5 | complete | 1517.2 | True | 0/0 | 0 | -0.0577653370417488 |
| noise-n300-p15 | complete | 4782.6 | True | 0/0 | 0 | -0.04808690465695964 |
| additive-n80-p5 | complete | 891.9 | True | 1/1 | 0 | -0.05044322317649885 |
| additive-n300-p5 | complete | 1541.0 | True | 1/1 | 0 | -0.009624422335464344 |
| additive-n300-p15 | complete | 4791.2 | True | 1/1 | 0 | -0.019796917106965406 |
| interaction-n80-p5 | complete | 859.1 | False | 0/2 | 0 | -0.09041861672490192 |
| interaction-n300-p5 | complete | 1558.8 | False | 0/2 | 0 | 0.17211434234382017 |
| interaction-n300-p15 | complete | 5153.6 | False | 0/2 | 0 | 0.007093659156778731 |
| chain-n80-p5 | complete | 2835.3 | True | 1/3 | 1 | 0.020896419698296143 |
| chain-n300-p5 | complete | 3777.1 | True | 3/3 | 0 | -0.005232147743851612 |
| chain-n300-p15 | incomplete | 5421.2 | unavailable | unavailable | unavailable | unavailable |

Full evidence includes actual fit/tuning sizes, checkpoints and independent test predictions.
A missing downstream comparison remains a full-path miss; prediction does not certify the graph.
