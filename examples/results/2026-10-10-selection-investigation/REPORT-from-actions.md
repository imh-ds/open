# Selection investigation

Diagnostics completed/audited: 74/74. Continuation: complete.

Development diagnosis on previously inspected datasets; no minimum-N guidance or policy validation.
Original pilot remains historically 11/12 complete. Continuation is reported separately.

| Case | Scope | Model | Inputs | CV skill | Test skill |
|---|---|---|---|---:|---:|
| interaction-n80-p5 | full | gbm_shipped | all | -0.22203364801939518 | -0.09041861672490192 |
| interaction-n80-p5 | full | gbm_shipped | signal | 0.11703504504925466 | 0.30941105073060815 |
| interaction-n80-p5 | full | gbm_deeper | all | -0.1775550978363929 | -0.09041861672490192 |
| interaction-n80-p5 | full | gbm_deeper | signal | 0.12109266500946536 | 0.30941105073060815 |
| interaction-n80-p5 | full | rf_shipped | all | -0.02651177246655445 | 0.05406588550761238 |
| interaction-n80-p5 | full | rf_shipped | signal | 0.24281293726642927 | 0.13830639355582452 |
| interaction-n80-p5 | full | linear | product | 0.5504572689482788 | 0.598137961115123 |
| interaction-n80-p5 | sample0 | gbm_shipped | all | -0.24499215861846935 | -0.01683104054391582 |
| interaction-n80-p5 | sample0 | gbm_shipped | signal | 0.07465736686370494 | 0.16974493256864076 |
| interaction-n80-p5 | sample0 | gbm_deeper | all | -0.2218767398305861 | -0.01683104054391582 |
| interaction-n80-p5 | sample0 | gbm_deeper | signal | 0.05410427791654904 | 0.16974493256864076 |
| interaction-n80-p5 | sample0 | rf_shipped | all | -0.17705438597050227 | 0.059130852358692865 |
| interaction-n80-p5 | sample0 | rf_shipped | signal | -0.017994043446156516 | 0.27035057799735884 |
| interaction-n80-p5 | sample0 | linear | product | 0.5181717071399423 | 0.5967492453006396 |
| interaction-n300-p5 | full | gbm_shipped | all | 0.08116571118000793 | 0.17211434234382017 |
| interaction-n300-p5 | full | gbm_shipped | signal | 0.2796971352359259 | 0.48350193559787036 |
| interaction-n300-p5 | full | gbm_deeper | all | 0.09051916212103928 | 0.3217549431127976 |
| interaction-n300-p5 | full | gbm_deeper | signal | 0.2796971352359259 | 0.48350193559787036 |
| interaction-n300-p5 | full | rf_shipped | all | 0.12899737930957766 | 0.2903148121522303 |
| interaction-n300-p5 | full | rf_shipped | signal | 0.2611846019486722 | 0.40739468109306554 |
| interaction-n300-p5 | full | linear | product | 0.5974760168283237 | 0.6277683063315509 |
| interaction-n300-p5 | sample0 | gbm_shipped | all | 0.1717238776323362 | 0.28064467910122237 |
| interaction-n300-p5 | sample0 | gbm_shipped | signal | 0.326949096948968 | 0.47192905515246397 |
| interaction-n300-p5 | sample0 | gbm_deeper | all | 0.2192123698225763 | 0.33695960359167465 |
| interaction-n300-p5 | sample0 | gbm_deeper | signal | 0.32118169502916794 | 0.4751524469206976 |
| interaction-n300-p5 | sample0 | rf_shipped | all | 0.24517746530990525 | 0.2745589075014364 |
| interaction-n300-p5 | sample0 | rf_shipped | signal | 0.3888575491613353 | 0.38557127576485384 |
| interaction-n300-p5 | sample0 | linear | product | 0.6092457047743738 | 0.6289932589434649 |
| interaction-n300-p15 | full | gbm_shipped | all | -0.08887868279782407 | 0.007093659156778731 |
| interaction-n300-p15 | full | gbm_shipped | signal | 0.3260979492517663 | 0.34269466236704427 |
| interaction-n300-p15 | full | gbm_deeper | all | -0.01878445622523639 | 0.09876624933894718 |
| interaction-n300-p15 | full | gbm_deeper | signal | 0.32007081273056703 | 0.34269466236704427 |
| interaction-n300-p15 | full | rf_shipped | all | 0.03839380352284105 | 0.15490991662720432 |
| interaction-n300-p15 | full | rf_shipped | signal | 0.28865564100554186 | 0.3599590952393046 |
| interaction-n300-p15 | full | linear | product | 0.6157903779484051 | 0.5708480446981217 |
| interaction-n300-p15 | sample0 | gbm_shipped | all | -0.07385652959901634 | 0.03094011817999265 |
| interaction-n300-p15 | sample0 | gbm_shipped | signal | 0.3148226097559257 | 0.3243030284656738 |
| interaction-n300-p15 | sample0 | gbm_deeper | all | -0.04528773671405251 | 0.03094011817999265 |
| interaction-n300-p15 | sample0 | gbm_deeper | signal | 0.3148226097559257 | 0.3243030284656738 |
| interaction-n300-p15 | sample0 | rf_shipped | all | 0.03143698135861317 | 0.11027932225072645 |
| interaction-n300-p15 | sample0 | rf_shipped | signal | 0.2610978644486067 | 0.3188246874608448 |
| interaction-n300-p15 | sample0 | linear | product | 0.553839766511018 | 0.5638727740184792 |
| chain-n80-p5 | full | gbm_shipped | all | 0.5301576499553675 | 0.44912400164758937 |
| chain-n80-p5 | full | gbm_shipped | signal | 0.5689041607069087 | 0.4790367603971135 |
| chain-n80-p5 | full | gbm_shipped | signal_noise | 0.6102547131992293 | 0.4282275819492932 |
| chain-n80-p5 | full | gbm_shipped | noise | -0.048244565030034714 | -0.21173854566956618 |
| chain-n80-p5 | full | gbm_shipped | without_noise | 0.42701136582845745 | 0.4878126811387006 |
| chain-n80-p5 | full | rf_shipped | all | 0.49709108444194405 | 0.4955823790318461 |
| chain-n80-p5 | full | rf_shipped | signal_noise | 0.5701527383761369 | 0.4310438547822374 |
| chain-n80-p5 | full | linear | all | 0.5690124755681134 | 0.5616689037736468 |
| chain-n80-p5 | full | linear | signal | 0.6059930439730621 | 0.578470932915281 |
| chain-n80-p5 | full | linear | signal_noise | 0.608325386728332 | 0.565748473954125 |
| chain-n80-p5 | full | linear | noise | -0.0036111078443736844 | -0.0340191067866944 |
| chain-n80-p5 | sample0 | gbm_shipped | all | 0.5168370172948737 | 0.43549754419594655 |
| chain-n80-p5 | sample0 | gbm_shipped | signal | 0.5423906554581042 | 0.4818751859432343 |
| chain-n80-p5 | sample0 | gbm_shipped | signal_noise | 0.5990137294624955 | 0.4499705527880533 |
| chain-n80-p5 | sample0 | gbm_shipped | noise | -0.1170753053756024 | -0.16782280134833583 |
| chain-n80-p5 | sample0 | gbm_shipped | without_noise | 0.48341855455703686 | 0.42309928325231116 |
| chain-n80-p5 | sample0 | rf_shipped | all | 0.4766215423542973 | 0.49790359352239255 |
| chain-n80-p5 | sample0 | rf_shipped | signal_noise | 0.5303061426169382 | 0.49570167420624245 |
| chain-n80-p5 | sample0 | linear | all | 0.5654915586468854 | 0.5681377074770031 |
| chain-n80-p5 | sample0 | linear | signal | 0.6326872753575727 | 0.5768706865656594 |
| chain-n80-p5 | sample0 | linear | signal_noise | 0.6218864318987616 | 0.5750309328897203 |
| chain-n80-p5 | sample0 | linear | noise | -0.018218641851260653 | -0.010341978506109184 |

| LOCO sample | Fixed-setting x4 contribution / baseline | Retuned contribution / baseline |
|---|---:|---:|
| 0 | 0.037982117177850605 | 0.03341846273783679 |
| 1 | 0.10461519431485221 | 0.071209166357413 |
| 2 | 0.05537784342024794 | 0.033328696309107 |
| 3 | 0.06790165924620309 | 0.059259232625942515 |
| 4 | 0.1039405067916283 | 0.09728294475806043 |
| 5 | 0.08805042620270503 | 0.0808208003912397 |
| 6 | 0.10828411334816615 | 0.09130755267092175 |
| 7 | 0.062387403364243095 | 0.04284909690005948 |
| 8 | 0.007317111420938921 | 0.014265684637164064 |
| 9 | 0.09797845618883089 | 0.12721691577379188 |

Missing/failed tasks keep their planned identities. Negative skill means worse than the training-mean baseline.
CV tests training-only tuning; the same fixed development test rows compare refitted models.
Predictive improvements do not certify a link or justify changing thresholds. LOCO samples overlap.

Continuation structural results: {"root_recovered": true, "false_root_inputs": [], "required_edges_found": 3, "required_edges_total": 3, "missing_edges": [], "extra_edges": [], "path_recovered": true, "required_depth_matches": {"y": true, "x1": true, "x2": true, "x3": true}}
Continuation prediction results: {"full": {"mse": 0.7264028948925063, "skill": 0.5548137474270021}, "retained": {"mse": 0.6977590839832498, "skill": 0.5723685106689471}, "oracle": {"mse": 0.6761365163213362, "skill": 0.5856202060243709}, "skill_loss": -0.01755476324194505}
