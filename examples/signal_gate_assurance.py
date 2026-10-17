"""Exact analytic assurance of the frozen error gate; no simulation or fitting."""
import argparse
import csv
from pathlib import Path

from scipy.stats import beta, binom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    # R counts independent validation datasets, not participants per dataset.
    # The hypothetical true false-network probability is explicitly fixed at .05.
    for repetitions in [55, 100, 200, 300, 400]:
        allowed = [errors for errors in range(repetitions)
                   if beta.ppf(.975, errors + 1, repetitions - errors) <= .10]
        maximum = max(allowed, default=-1)
        rows.append(dict(validation_datasets=repetitions,
                         maximum_errors_passing=maximum,
                         hypothetical_true_error_probability=.05,
                         upper_interval_quantile=.975,
                         gate_upper_error_limit=.10,
                         probability_passing=float(binom.cdf(maximum, repetitions, .05))))
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
