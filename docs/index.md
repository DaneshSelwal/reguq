# reguq

**Regression uncertainty quantification with tested guarantees.**

`reguq` turns any scikit-learn style regressor into one that returns a **prediction interval** or a
full **predictive distribution**. Each method is implemented as its original paper defines it,
states what it guarantees and under which assumptions, and is checked by Monte-Carlo tests.

```python
from sklearn.ensemble import GradientBoostingRegressor
from reguq.conformal import SplitConformalRegressor

cp = SplitConformalRegressor(GradientBoostingRegressor(), random_state=0).fit(X_train, y_train)
lower_upper = cp.predict_interval(X_test, alpha=0.1)   # (n, 2), >= 90 % coverage
```

## What is in the box

| Family | Methods | Guarantee |
|---|---|---|
| [Split conformal and CQR](methods/split-cqr.md) | absolute and normalized scores, CQR (symmetric or per tail) | finite-sample marginal coverage ≥ 1−α |
| [Resampling conformal](methods/resampling.md) | Jackknife+, CV+, minmax, Jackknife+-after-bootstrap | ≥ 1−2α (plus), ≥ 1−α (minmax) |
| [Non-exchangeable](methods/nexcp.md) | weighted NexCP | ≥ 1−α − drift term |
| [Online](methods/online.md) | ACI, DtACI/FACI, SF-OGD, SAOCP, online NexCP | long-run miscoverage → α for any sequence (ACI, DtACI) |
| [Extreme tails](methods/extreme.md) | POT/GPD on conformity scores | none (semi-parametric extrapolation) |
| [Adaptive coverage](methods/acp.md) | ACP (data-dependent α̃) | post-hoc \(\mathbb E[P(\text{miss}\mid\tilde\alpha)/\tilde\alpha]\le 1\) |
| [Quantile regression](methods/quantile.md) | boosted/linear quantiles + rearrangement, QRF, TabNet | none alone; wrap in CQR |
| [Probabilistic](methods/probabilistic.md) | NGBoost, PGBM, IBUG, CARD, HCM, Treeffuser | model-based; conformalize via `StdEstimator` |

## Design principles

1. **The test set is touched once.** Tuning uses cross-validation on the training data, and
   calibration data is held out from the training data inside `fit`.
2. **Time is respected.** `time_ordered=True` disables every shuffle, and online methods are strictly
   causal: the interval at time *t* depends only on outcomes before *t*. A test enforces this.
3. **No silent fallbacks.** If a method cannot run, you get an error, never a different method under
   the same name.
4. **Light core.** `import reguq` loads only numpy, scipy, pandas and scikit-learn. Each backend
   (LightGBM, torch, …) is an optional extra, imported only when used.

## Why a rewrite

reguq 1.0 replaces the code used in the v1.2 analysis notebooks. A line-by-line audit found that several
methods did not compute what their names said, and that hyper-parameters had been selected on the
test set. The full list, with corrected formulas, is in the [audit](audit/notebooks-v1.2.md).

## Citing

Cite reguq with the repository's `CITATION.cff`, **and** cite the papers of the methods you use, which
are listed in the [bibliography](bibliography.md).
