# Adaptive coverage policies (ACP)

Gauthier, Bach & Jordan (2026), *Adaptive coverage policies in conformal prediction*
(AISTATS; arXiv:2510.04318). Class: `AdaptiveCoveragePolicyRegressor`.

## Idea

Instead of one fixed α, choose a level \(\tilde\alpha(x)\) for each test point. A point whose
interval would be very wide at 1−α can accept a lower coverage level, in exchange for a much
narrower interval. Coverage is then guaranteed in the following **post-hoc** (e-value) sense:

\[
\mathbb E\Big[\frac{P(Y\notin\hat C_{\tilde\alpha}(X)\mid\tilde\alpha)}{\tilde\alpha}\Big]\le 1 .
\]

This is **not** marginal \(1-\alpha\) coverage. The docstring and the pipeline report \(\tilde\alpha\)
(`alpha_tilde_`) so that results are not misread.

## Interval at level α̃

The interval is built from calibration scores \(S_1,\dots,S_n\) using an e-value (mean-based)
bound rather than an order statistic:

\[
\hat C_{\tilde\alpha}(x)=\hat\mu(x)\pm\frac{\sum_{i}S_i}{(n+1)\tilde\alpha-1},\qquad \tilde\alpha\in(1/(n+1),1).
\]

## Choosing α̃

The policy minimizes the leave-one-out loss of width plus a price λ on miscoverage:

\[
\mathcal L(\tilde\alpha)=\frac{2T_j}{n\tilde\alpha-1}+\lambda\tilde\alpha,
\]

where \(T_j\) is the leave-one-out score sum.

- `policy="closed_form"` minimizes this for a constant α̃:
  \(\tilde\alpha^\star=\big(1+\sqrt{2\bar T n/\lambda}\big)/n\).
- `policy="mlp"` learns α̃(x) with a small network that takes x as input (`torch` extra).

**λ is in target units**, because it trades off width against probability. reguq sets it in one of
two ways:

- the default, `lam=None`: \(\lambda_\alpha=2\bar T n/(n\alpha-1)^2\). This choice makes
  \(\tilde\alpha^\star=\alpha\) for the constant policy, so ACP starts from the user's α;
- `target_size=`: λ is found by bisection so that the mean interval width matches the target
  (Algorithm 2 of the paper).

!!! danger "v1.2 notebook errors"
    The old AlphaNet/ACP ignored the target α, and its λ was not scaled to the target. With widths of
    about 600 target units against λ = 20, α̃ always moved to its upper clip of 1. A sigmoid near 0 also
    made the denominator negative and blew up the width (audit NX-11, ACP-1 to ACP-4).
