# Contributing to reguq

## Development setup

```bash
git clone https://github.com/DaneshSelwal/reguq && cd reguq
pip install -e ".[dev,docs]"          # add extras you need, e.g. [boosting,tuning]
pre-commit install
```

Checks run by CI on every push:

```bash
ruff check . && ruff format --check .
mypy
pytest -m "not slow" --cov=reguq
mkdocs build --strict
```

## Rules for methods

reguq exists because the methods it replaced did not do what their names said. Every method must:

1. **Cite** the paper and, where one exists, the reference implementation. Put the citation in the class docstring and in
   `docs/bibliography.md`.
2. **State its guarantee** in the docstring, e.g. "marginal coverage ≥ 1−α under exchangeability",
   or "none: model-based". Anything without a guarantee says so.
3. **Never touch the test set**: calibration data comes from the training data passed to `fit`, and
   `time_ordered=True` must never shuffle.
4. **Never fall back silently**: raise a clear error instead of substituting another method.
5. **Be causal** if it is an online method: the interval at time t uses only `y_1 … y_{t−1}`. There is a
   test for this property, and new online methods must pass it.
6. **Be tested statistically**: a guarantee needs a Monte-Carlo test (marker `statistical`).
7. **Import heavy backends lazily** inside `fit`, so `import reguq` stays light.

## Commits and releases

- Small, focused commits; update `CHANGELOG.md` under *Unreleased*.
- Releases: bump `src/reguq/_version.py`, tag `vX.Y.Z`. The *Release* workflow builds, tests the
  wheel, and publishes to PyPI via trusted publishing.
