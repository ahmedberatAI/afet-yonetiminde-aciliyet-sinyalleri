# Postprocess info_v1 Validation - 2026-05-17

Validated change: keep canonical thresholds, then add `bilgi_paylasimi` when the text has a strong information-sharing signal and `prob_bilgi_paylasimi >= 0.20`.

Decision surfaces are OOF and validation. Test is shown as a non-selection regression check.

## Metrics

| dataset | profile | micro F1 | macro F1 | bilgi F1 | bilgi P | bilgi R | bilgi FP | bilgi FN | rule adds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | base | 0.9087 | 0.8143 | 0.8062 | 0.8458 | 0.7702 | 33 | 54 | 0 |
| oof | info_v1 | 0.9136 | 0.8195 | 0.8525 | 0.8115 | 0.8979 | 49 | 24 | 46 |
| validation | base | 0.9040 | 0.8313 | 0.8438 | 0.9310 | 0.7714 | 2 | 8 | 0 |
| validation | info_v1 | 0.9204 | 0.8423 | 0.9429 | 0.9429 | 0.9429 | 2 | 2 | 6 |
| test_regression_check | base | 0.8998 | 0.8753 | 0.6809 | 0.7273 | 0.6400 | 6 | 9 | 0 |
| test_regression_check | info_v1 | 0.9082 | 0.8851 | 0.7692 | 0.7407 | 0.8000 | 7 | 5 | 5 |

## Deltas (info_v1 - base)

| dataset | micro F1 | macro F1 | bilgi F1 | bilgi precision | bilgi recall |
|---|---:|---:|---:|---:|---:|
| oof | +0.0049 | +0.0051 | +0.0463 | -0.0343 | +0.1277 |
| validation | +0.0164 | +0.0110 | +0.0991 | +0.0118 | +0.1714 |
| test_regression_check | +0.0085 | +0.0098 | +0.0884 | +0.0135 | +0.1600 |

## Decision

- `info_v1` improves OOF and validation micro F1, macro F1, and `bilgi_paylasimi` F1 together.
- `guvenlik` rule candidates were not promoted: they added false positives without consistent OOF/validation gain.
- This is an inference-layer improvement, not a model-weight or CV-threshold replacement; use `--postprocess-profile none` for exact baseline reproduction.
