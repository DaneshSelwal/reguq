# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0.dev0] - unreleased (branch `v1.0-rewrite`)
Complete rewrite of the UQ core. See `docs/audit/notebooks-v1.2.md` for why.

### Removed
- `Data_folder/` (copies of the v1.2 notebooks with empty placeholder Excel files) and the
  accidentally committed `build/` directory. The v1.2 notebooks live in
  `Regression_Uncertainty_Quantification_Analysis`; the audit is in `docs/audit/`.

### Changed
- Core install is numpy/scipy/pandas/scikit-learn only; all backends are extras with `>=` bounds.
- One estimator contract: `fit`, `predict -> (n,)`, `predict_interval(X, alpha) -> (n, 2)`,
  `predict_dist` for distributional models.
- Calibration is always on data the model was not fitted on; `time_ordered=True` never shuffles.
- Tuning uses training data only (CV / forward chaining); pruners work.
- Failures are reported, never silently replaced by another method.

### Added
- Paper-faithful: split/normalized CP, CQR, CV+/Jackknife+/minmax, J+aB, NexCP, ACI, DtACI/FACI,
  SF-OGD, SAOCP, online NexCP, POT/GPD tail, ACP; multi-quantile with rearrangement, QRF, TabNet;
  NGBoost, PGBM, IBUG, CARD, HCM, Treeffuser.
- Metrics: Winkler interval score, weighted quantile CRPS, sample CRPS, PIT, ENCE, size-stratified coverage.
- Monte-Carlo tests of coverage guarantees and causality of online methods.
- `reguq.reporting`: interval, PIT-reliability, coverage-vs-width and binned-coverage plots; Excel export.
- Pipeline split into `config` (validated on construction), `data`, `methods`, `evaluate`, `runner`,
  `export`, `cli`; `reguq-run -v`; logging instead of print.
- Engineering: mypy-clean and typed (`py.typed`), coverage, pre-commit, Dependabot, CI build check
  (`twine check`), release workflow (tag -> tested wheel -> GitHub release -> PyPI trusted publishing),
  MkDocs documentation with theory pages, API reference and bibliography.
- `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, issue and PR templates.

### Removed
- 0.2 `run_*` phase API, Colab bootstrap, Excel-first export, mislabelled heuristics
  ("NexCP-Full", fixed-window "SAOCP", "MFCS-Split", "COP").
  MFCS (model-free conformal *selection*) is a selection procedure rather than an interval method and is
  out of scope for v1.0.

## [0.2.0] - 2026-03-22
### Added
- **New Models (4)**: Random Forest, Gradient Boosting, GPBoost, TabNet
  - Total supported models now: 9 (LightGBM, XGBoost, CatBoost, NGBoost, PGBM, RandomForest, GradientBoosting, GPBoost, TabNet)

- **Advanced Conformal Prediction** (`run_conformal_advanced`):
  - NexCP Split: Non-exchangeable CP with exponential weighting
  - NexCP Full: Full conformal prediction
  - NexCP Jackknife+aB: Jackknife+ after Bootstrap
  - NexCP CV+: Cross-Validation Plus
  - Online Split: Online conformal prediction
  - FACI: Fully Adaptive Conformal Inference
  - MFCS Split/Full: Model-Free Conformal Selection
  - CVPlus: PUNCC CV+ (Cross-Validation Plus)
  - CQR: PUNCC Conformalized Quantile Regression

- **Advanced Probabilistic Methods** (`run_probabilistic_advanced`):
  - CARD: Classification And Regression Diffusion models
  - IBUG: Instance-Based Uncertainty using Gradient Boosting
  - Treeffuser: Diffusion-based gradient boosting (optional)

- **Explainability Module** (`run_explainability`):
  - SHAP: SHapley Additive exPlanations
  - LIME: Local Interpretable Model-agnostic Explanations
  - InterpretML: Explainable Boosting Machine

- Comprehensive paper citations in README.md for all implemented methods
- New API exports: `CARDRegressor`, `IBUGRegressor`, `explain_shap`, `explain_lime`, `explain_interpret`
- Optional dependencies in pyproject.toml: `[interpretability]`, `[tabnet]`, `[advanced_probabilistic]`, `[all]`

### Changed
- Version bump to 0.2.0
- Registry expanded with 4 new models and hyperparameter search spaces
- Constants updated with new phases: `PHASE_CONFORMAL_ADVANCED`, `PHASE_EXPLAINABILITY`
- pyproject.toml updated with gpboost dependency and optional dependency groups
- README.md completely restructured with comprehensive documentation and citations

### Technical Notes
- CARD requires PyTorch (optional dependency)
- TabNet requires pytorch-tabnet (optional dependency)
- Treeffuser is an experimental optional dependency
- All advanced methods gracefully fall back when dependencies are missing

## [0.1.1] - 2026-03-10
### Fixed
- Colab import stability for `bootstrap_colab_environment` by documenting and using forced package refresh in the validation notebook.
- Runtime stale-package issues by updating install flow to use `--upgrade --force-reinstall --no-cache-dir`.

### Changed
- Package version bumped to `0.1.1` for clear upgrade path from previous Colab sessions.

## [0.1.0] - 2026-03-09
### Added
- Core `reguq` package modules for tuning, quantile regression, probabilistic regression, and standard conformal prediction.
- Config runner and CLI entrypoint: `reguq-run --config <yaml>`.
- Rich reporting controls in `OutputConfig`:
  - `embed_excel_charts`
  - `show_inline_plots`
  - `chart_detail_level`
  - `legend_position`
  - `style_overrides`
- Detailed chart generation subsystem with phase/model diagnostics.
- Excel chart embedding support via OpenPyXL image anchors.
- Colab bootstrap helper `bootstrap_colab_environment()` for pinned setup + one-time restart.
- Colab validation notebook in `examples/reguq_colab_check.ipynb`.
- CI workflow for tests and release workflow templates.

### Changed
- Standardized report visual styling and chart labeling for phase outputs.
- Updated examples and README to use GitHub repo `DaneshSelwal/reguq`.

### Notes
- Advanced CP methods and CARD are intentionally deferred beyond v1.
- GitHub release is the current ship target; PyPI publish remains a follow-up.
