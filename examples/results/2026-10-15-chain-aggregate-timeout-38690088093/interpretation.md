# Chain aggregate timeout: analyses completed, verification unfinished

[Run 38690088093](https://github.com/imh-ds/open/actions/runs/38690088093)
failed in its aggregate job. All 110 case jobs succeeded and uploaded their
artifacts. The aggregate downloaded all 110, with no missing or failed downloads.

The independent observation-loss replay then reached its explicit **32-minute
step timeout**. The aggregate job had a 60-minute overall cap, but the shorter
step cap stopped verification first. This is a verification runtime failure,
not evidence that the recursive models failed or that their findings passed.
The replay step limit was set too low for this workload.

The uploaded audit says 0/110 complete and qualified=false because it is the
initial placeholder, written before replay. Results are collected with a blocking
pool.map call and written after all cases return; forced termination prevented
that update. Do not interpret this placeholder as 110 failed model fits or a
scientific failure. No finished per-case replay results were uploaded.

The next step is to finish the independent no-fit audit using the saved 110 case
artifacts, with enough verification time and durable per-case audit progress.
No statistical models need to be fitted again. Keep the original identities,
per-N denominators and pass criteria, and keep the failed workflow status.
The 110 correlated-predictor cases remain pending.

This diagnosis launched no additional Actions jobs and performed no model fits.
The existing reservation is 9,889/10,000 runner-minutes; 111 remain unallocated.
A further Actions recovery must fit that allowance or obtain a new allocation.
The original 110 case archives are still on Actions; they have not yet been
copied into this diagnosis packet. The packet preserves both API pages, the
aggregate log and its uploaded ZIP; SHA256 fingerprints are provided.
