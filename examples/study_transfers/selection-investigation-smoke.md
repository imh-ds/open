# Verified continuation checkpoints

`selection-investigation-smoke.zip` transfers four original chain-N300/p15
assessments, depth1/targetx1/run16–19. It contains exactly four checkpoints,
four process metrics files, the original shard receipt and its recorded manifest.
SHA256: `45cd4e528d9bd8fbb46b423a57833a1de629e79e603f96a7674a42ba3cb43406`.

All120saved checkpoints (116original plus these4) passed the existing pilot
context/plan/identity/local-arithmetic audit before transfer. Production/pilot
sources, original data, candidates, settings and assessment IDs are unchanged.
These files avoid computing the same four assessments again on Actions.

The orchestration helper was being edited during the smoke while its loaded
version continued running. The recorded manifest reflects files observed when
the receipt was written, not an assertion that the helper was immutable in memory.
No production scientific source changed. Checkpoint contexts independently
fingerprint those frozen sources, original protocol, data, config and comparison.
The new transfer checksum fixes all bytes; the follow-up checks these contexts
and regenerates original sample plans before using any assessment.
