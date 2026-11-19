# Choosing a method

## Decision guide

```text
Are test points exchangeable with the training data (i.i.d., no ordering)?
├── yes
│   ├── plenty of data (n_calib ≥ ~200)?   → split conformal: CQR or normalized score
│   └── small data, can afford refits?     → CV+ / Jackknife+ (or J+aB with bagging)
└── no (time series, drift)
    ├── outcomes arrive one by one?        → online: ACI (simple) or DtACI (no tuning of γ)
    └── batch forecast of a later period?  → NexCP with recency weights
Need a full distribution (CRPS, sampling)? → NGBoost / PGBM / IBUG / CARD,
                                             then conformalize with StdEstimator
Need very small α with few points beyond the threshold? → POT/GPD tail (no guarantee)
Prefer a per-point coverage level that trades off width? → ACP
```

## Guarantees in one table

| Method | Assumption | Guarantee | Width adapts to x? |
|---|---|---|---|
| Split, absolute score | exchangeable | \(1-\alpha \le P \le 1-\alpha+\frac{1}{n+1}\) | no |
| Split, normalized score | exchangeable | same | yes (via σ̂(x)) |
| CQR | exchangeable | ≥ 1−α | yes (via quantiles) |
| Jackknife+ / CV+ | exchangeable | ≥ 1−2α (≈ 1−α in practice) | no |
| Minmax | exchangeable | ≥ 1−α | no |
| J+aB | exchangeable, random B | ≥ 1−2α | no |
| NexCP | none | ≥ 1−α − Σ w̃ᵢ d_TV | no |
| ACI, DtACI | none | long-run miscoverage → α | no |
| SF-OGD, SAOCP | none | regret bounds | no |
| POT/GPD | GPD tail fits | none | no |
| ACP | exchangeable | post-hoc E[P(miss \| α̃)/α̃] ≤ 1 | via α̃ |
| NGBoost, PGBM, IBUG, CARD, HCM, Treeffuser | model correct | none | yes |

"Coverage" here is always **marginal**: averaged over the test distribution. No method guarantees
coverage conditional on a particular x. `metrics.binned_coverage` and `size_stratified_coverage` show
how far a method is from conditional coverage.

## Comparing methods

Rank methods by a **proper** score, not by coverage or width alone:

- **intervals:** `metrics.interval_score` (Winkler). It rewards narrow intervals and penalises misses
  by \(2/\alpha\) times the distance.
- **distributions:** `metrics.crps_*`, and `gaussian_nll` if the Gaussian assumption is reasonable.

Check that coverage is close to 1−α. A method whose coverage falls clearly below target is invalid for
your data, however narrow its intervals. With \(n\) test points, the Monte-Carlo standard error of
coverage is about \(\sqrt{\alpha(1-\alpha)/n}\): about 0.016 for α = 0.1 and n = 365.
