# Probabilistic models

These estimators return a full predictive distribution through `predict_dist(X)`:

- `GaussianPrediction(mu, sigma)`, where **sigma is always a standard deviation**;
- `SamplePrediction(samples)`, with samples of shape `(n, m)`.

Both provide `mean()`, `std()`, `quantiles(taus)`, `interval(alpha)`, `sample(m)`, `crps(y)`,
`nll(y)` and `pit(y)`. Every distributional model is also an interval regressor, through
`predict_interval(X, alpha)`, which uses the equal-tailed central interval.

!!! warning "Model-based, not guaranteed"
    These intervals are only as good as the model. For a finite-sample guarantee, conformalize the
    model with `StdEstimator` (see the [quick start](../guide/quickstart.md#a-predictive-distribution)).

## NGBoost: `NGBoostRegressor`

Duan et al. (2020). Boosts the parameters \((\mu,\log\sigma)\) of a Normal distribution with natural
gradients of the log score. `early_stopping_rounds` uses the validation data passed to `fit`.

## PGBM: `PGBMRegressor`

Sprangers, Schelter & de Rijke (2021). Each leaf stores the mean and variance of its gradient
statistics. These give a per-point mean and variance of the ensemble prediction, which are matched
to a distribution after training (`distribution="normal"`, or sampled for other choices). reguq
uses `pgbm.sklearn`, whose `predict(X, return_std=True)` returns the **standard deviation**.

## IBUG: `IBUGRegressor`

Brophy & Lowd (2022). Wraps any fitted tree ensemble (LightGBM, XGBoost, CatBoost, sklearn forests,
GBR). The affinity between test point \(x\) and training point \(i\) is the number of trees in which
both fall in the same leaf. The predictive distribution is
\(N\big(\hat\mu(x),\ \gamma\,\mathrm{sd}\{Y_i: i\in\text{top-}k\}+\delta\big)\), where:

- \(k\) is chosen from the reference grid {3, 5, 7, …, 701} by validation NLL;
- \((\gamma,\delta)\) is a multiplicative or additive calibration of the std, chosen from the
  reference grid, also by validation NLL.

!!! danger "v1.2 notebook error"
    The notebook took the square root of IBUG's `scale`, which is already a standard deviation.
    Intervals were therefore far too narrow: observed coverage 0.45–0.69 at nominal 0.90
    (audit P2).

## CARD: `CARDRegressor`

Han, Zheng & Zhou (2022), Classification and Regression Diffusion. A pre-trained mean model
\(f_\phi(x)\) (any regressor, given as `mean_estimator`) is the prior mean of the diffusion:

\[
y_t=\sqrt{\bar\alpha_t}\,y_0+(1-\sqrt{\bar\alpha_t})f_\phi(x)+\sqrt{1-\bar\alpha_t}\,\epsilon ,
\]

with a linear β schedule from \(10^{-4}\) to 0.02 over \(T\) steps (`n_steps`, default 1000). The
network \(\epsilon_\theta(x,y_t,f_\phi(x),t)\) has conditional-linear layers with 128 units, softplus
activations and timestep-embedding gates, and is trained with \(\|\epsilon-\epsilon_\theta\|^2\).
Sampling starts at \(y_T\sim N(f_\phi(x),1)\) and follows the posterior with mean
\(\gamma_0\hat y_0+\gamma_1y_t+\gamma_2f_\phi(x)\) and variance \(\tilde\beta_t\):

\[
\gamma_0=\frac{\beta_t\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t},\quad
\gamma_1=\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t},\quad
\gamma_2=1+\frac{(\sqrt{\bar\alpha_t}-1)(\sqrt{\alpha_t}+\sqrt{\bar\alpha_{t-1}})}{1-\bar\alpha_t},\quad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t .
\]

Targets are standardized internally, and the samples are returned in the original units.

!!! danger "v1.2 notebook error"
    The old "CARD" had no noise schedule and never built \(y_t\). Its samples were \(\hat y+N(0,1)\),
    giving σ ≈ 1 for every dataset (audit C1).

## HCM: `HCMRegressor`

Choi et al. (2026), Hyperspherical Confidence Mapping (ICLR). The target is embedded as
\(y_{\text{exp}}=(y,y)\in\mathbb R^2\), with magnitude \(R=\|y_{\text{exp}}\|\) and direction
\(d=y_{\text{exp}}/R\). A two-head network predicts \(\hat R,\hat d\) with loss

\[
e_R^2+R^2\|e_d\|^2+\lambda(\|\hat d\|-1)^2,
\]

and the uncertainty is \(\hat\sigma^2=\hat R^2\,\big|\|\hat d\|^2-1\big|/(D-1)\). The temperature
of σ is fitted on a calibration split **separate** from the early-stopping split.

!!! note "Limitation for scalar targets"
    For a 1-D standardized target, \(d\) only encodes \(\mathrm{sign}(y)\), so \(\|\hat d\|\) mostly
    reflects how hard the sign is to predict, i.e. how close \(y\) is to its mean. HCM is therefore weak
    for scalar regression, which explains the collapse seen in the v1.2 notebook (audit H2). Prefer
    conformalizing it, or use another model.

## Treeffuser: `TreeffuserRegressor`

Beltran-Velez et al. (2024). A conditional score-based diffusion whose score function is learned
by gradient-boosted trees. reguq wraps the `treeffuser` package's `sample(X, n_samples, n_steps)`
and returns a `SamplePrediction`. It has its **own** hyper-parameters (`treeffuser_params`). Feeding
it the point model's parameters is what made the v1.2 results degenerate (audit P4).
