# Conformal foundations

## Setting

We observe \(Z_i=(X_i,Y_i)\), \(i=1,\dots,n\), and a test point \(X_{n+1}\). We want a set
\(\hat C(X_{n+1})\) such that

\[
P\big(Y_{n+1}\in\hat C(X_{n+1})\big)\ge 1-\alpha .
\]

The probability is over the calibration data **and** the test point, so this is *marginal* coverage.

## The conformal quantile

Let \(S_1,\dots,S_n\) be conformity scores computed on data the model was **not** fitted on, for
example \(S_i=|Y_i-\hat\mu(X_i)|\). If the scores and the test score are exchangeable, the rank of
\(S_{n+1}\) among all \(n+1\) scores is uniform. Hence

\[
\hat q=S_{(k)},\qquad k=\big\lceil (n+1)(1-\alpha)\big\rceil ,
\]

the \(k\)-th smallest score, satisfies \(P(S_{n+1}\le\hat q)\ge 1-\alpha\). If the scores have no
ties, the coverage is also at most \(1-\alpha+\frac{1}{n+1}\) (Lei et al. 2018, Thm 2.2).

If \(k>n\), meaning \(\alpha<1/(n+1)\), no finite quantile is valid, and
`reguq.conformal.conformal_quantile` returns \(+\infty\). It never clips to the maximum score,
because that would silently void the guarantee.

```python
from reguq.conformal import conformal_quantile, conformal_rank
conformal_rank(n=99, alpha=0.1)        # 90 -> the 90th smallest of 99 scores
conformal_quantile(scores, alpha=0.1)
```

!!! note "Not `np.quantile`"
    `np.quantile(scores, 1 - alpha)` interpolates and uses level \(1-\alpha\), not
    \(\lceil(n+1)(1-\alpha)\rceil/n\). With small \(n\) it under-covers.

## Weighted quantile

Under distribution shift, scores can be given weights \(w_1,\dots,w_n\) and the test point a weight
\(w_{n+1}\). With \(\tilde w_i=w_i/\sum_{j=1}^{n+1}w_j\),

\[
\hat q=Q_{1-\alpha}\Big(\sum_{i=1}^n\tilde w_i\,\delta_{S_i}+\tilde w_{n+1}\,\delta_{+\infty}\Big),
\]

the smallest score whose cumulative normalized weight reaches \(1-\alpha\). The mass at \(+\infty\)
is what makes the bound hold (Barber et al. 2023; Tibshirani et al. 2019). Dropping it, as the v1.2
notebooks did, gives intervals that are too narrow. See `weighted_conformal_quantile`.

## Which data are calibration data?

All reguq estimators take `(X, y)` in `fit` and split them themselves with `reguq.splits`:

- `time_ordered=False` takes a random `calib_size` share, seeded by `random_state`;
- `time_ordered=True` takes the **last** `calib_size` share, with no shuffling.

The test set is never used for calibration, early stopping or tuning.
