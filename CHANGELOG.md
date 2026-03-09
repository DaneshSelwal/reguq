# Changelog

All notable changes to this project will be documented in this file.

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
