# Longer continuation for the same 38 unfinished datasets

The owner approved increasing the time limits and combining these results with the 72 completed original datasets. [The execution amendment](../../../qualification_continuation_protocol.md) preserves the model settings, seeds, scientific source and all 110 original identities. Each continued case gets a 60-minute job limit, including up to 39 minutes of new computation after saved-work verification. No automatic signal stage or retry is enabled.

Before dispatch, independent no-fit replay verified all **38 partial cases, 3,408 checkpoints and 4,341,760 observation-loss calculations**, with **zero unmatched attempts**. See [partial-replay.json](partial-replay.json) and [the reproduction script](replay.py). Three distinct interruption paths are covered: ordinary deadline, late completed network, and a compute-step interruption. The late completed network is fully audited in place and preserved byte-for-byte.

Software verification: the existing 18 qualification tests passed before changes; all **31 combined qualification/continuation tests passed in 96.08 seconds** after development. Real fitting is forbidden in these tests. The late-network regression was observed failing before its correction. An independent read-only reviewer identified that defect, then confirmed the correction with no remaining definite blockers. All 72 original archive discovery paths were verified against the downloaded artifact layout.

The unchanged original manifest plus 66 continuation source/proof files passed the pinned installed Python 3.11.9 runtime/source check. Continuation digest: `6f071a099f413f213ee03c18ffea2035c9d30b6925291a2df5e8b17278492bbe`. Source code and runtime are unchanged scientifically; the separately frozen adapter and workflow implement only the authorized execution amendment.

The budget reserves 2,356 additional runner-minutes, taking historical-plus-continuation reservations to 4,863. Of the owner's 10,000-minute ceiling, 5,137 remains. The old signal-stage allocation is no longer fully funded; it stays on hold pending a later decision. Counts and thresholds cannot be changed to obtain a passing result.

The original run remains failed. Its full archives and evidence commits are retained locally; two attempted 500 MB GitHub pushes failed. The small [producer audit](producer-audit.json), [SHA256 inventory](producer-sha256.json) and [byte-verification receipt](producer-verification.json) bind its still-available Actions artifacts. A separate publication branch starts from the last successful remote source so dispatch does not depend on uploading that large commit. Durable remote archive preservation remains outstanding and is required before a future signal gate.

Launch and startup receipts will be recorded separately. Software checks and launch do not establish that the null stage qualifies or that OPEN is ready for release.
