# Release notes

## Unreleased

- The package is renamed `opennet`, for both the distribution and the import
  (`from opennet.api import ...`). The previous import
  name, `open`, hid Python's built-in `open()`, and `open` is taken on PyPI.
  The distribution name `open-predictive-expansion` is retired. Archived study
  results and frozen protocols keep the old module paths they were run with.

## 0.1.0.dev0 — developer preview

This is the first developer preview of OPEN, an exploratory, outcome-centered
predictive network method.

- Explore recursive predictive networks with OLS, Elastic-Net, or LightGBM.
- Confirm frozen links on independent data or a held-out part of one dataset.
- Analyze multiple outcomes in one call, retaining a separate network for each.
- Export tables and settings, and optionally plot single or multiple networks.
- Install plotting and LightGBM support as optional extras.

This is a development release. Review the [method guide](docs/method.md) and
[validation evidence](docs/validation-evidence.md) before interpreting results.
The method is predictive and exploratory; arrows do not establish causation.
