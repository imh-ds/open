# Raw archive transport

All 180 original archives are preserved in the development checkout. Public archives 176–179 are stored in byte-exact 4 MiB pieces after repeated Git upload timeouts. No records or precision were removed.

Run `python assemble_archives.py` from this result directory before the standalone full-evidence verifier. It checks each piece and the assembled archive against the manifest SHA-256 and byte count, and checks ZIP CRCs. The reconstructed ZIPs are byte-identical to the originals.
