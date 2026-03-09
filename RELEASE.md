# GitHub Release Process (v1)

This project currently ships via GitHub Releases (not PyPI).

## 1) Pre-release checks

```bash
pytest -q
```

Confirm:
- Tests pass.
- `examples/reguq_colab_check.ipynb` runs in Colab.
- Chart-rich Excel outputs are generated for quantile/probabilistic/conformal runs.

## 2) Prepare version metadata

- Update `pyproject.toml` version if needed.
- Add release notes to `CHANGELOG.md`.

## 3) Tag and push

```bash
git tag -a v0.1.0 -m "reguq v0.1.0"
git push origin main --tags
```

## 4) Create GitHub Release

- Open repository Releases page.
- Select tag `v0.1.0`.
- Use the matching section from `CHANGELOG.md` for notes.

## 5) Post-release validation

In a fresh Colab runtime:

```python
%pip -q install "git+https://github.com/DaneshSelwal/reguq.git@v0.1.0"
from reguq import bootstrap_colab_environment
bootstrap_colab_environment(repo_url="https://github.com/DaneshSelwal/reguq.git")
```

Then run the full workflow notebook.

## PyPI (future)

`publish-pypi.yml` exists as a future workflow and should be enabled only after GitHub release validation is stable.
