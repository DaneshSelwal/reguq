# RegUQ: Comprehensive Regression Uncertainty Quantification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange?style=for-the-badge)
![Uncertainty Quantification](https://img.shields.io/badge/Uncertainty-Adaptive%20CP%20%7C%20NEXCP%20%7C%20CARD-green?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-0.2.0-purple?style=for-the-badge)

Welcome to **RegUQ** - a comprehensive, production-ready Python package for **Regression Uncertainty Quantification**. This framework goes beyond point predictions by integrating state-of-the-art UQ methods, ensuring every prediction is accompanied by reliable confidence intervals.

---

## Installation

```bash
# Install from GitHub
pip install "git+https://github.com/DaneshSelwal/reguq.git"

# With optional dependencies
pip install "git+https://github.com/DaneshSelwal/reguq.git[all]"
```

---

## Key Features (v0.2.0)

### Supported Models (9 Total)
| Model | Package | Quantile | Probabilistic | Conformal |
|-------|---------|----------|---------------|-----------|
| LightGBM | `lightgbm` | Yes | Yes | Yes |
| XGBoost | `xgboost` | Yes | Yes | Yes |
| CatBoost | `catboost` | Yes | Yes | Yes |
| NGBoost | `ngboost` | No | Yes | Yes |
| PGBM | `pgbm` | No | Yes | Yes |
| Random Forest | `sklearn` | No | Yes | Yes |
| Gradient Boosting | `sklearn` | Yes | Yes | Yes |
| GPBoost | `gpboost` | Yes | Yes | Yes |
| TabNet | `pytorch-tabnet` | No | Yes | Yes |

### Uncertainty Quantification Methods

| Category | Methods |
|----------|---------|
| **Quantile Regression** | Pinball loss, multi-quantile |
| **Probabilistic** | NGBoost, PGBM, CARD, IBUG, Treeffuser |
| **Standard Conformal** | MAPIE (Split, CV+), PUNCC (SplitCP, CVPlus, CQR) |
| **Advanced Conformal** | NexCP (Split, Full, J+aB, CV+), Online CP, FACI, MFCS |
| **Explainability** | SHAP, LIME, InterpretML |

---

## Quick Start

```python
import reguq

# Run quantile regression
result = reguq.run_quantile(
    data="data.csv",
    target_col="target",
    models=["lightgbm", "xgboost"],
    output_config={"output_dir": "./outputs", "export_excel": True}
)
print(result.metrics)

# Run advanced conformal prediction
result = reguq.run_conformal_advanced(
    data="data.csv",
    target_col="target",
    conformal_config={"methods": ["nexcp_split", "faci", "cvplus"]}
)

# Run CARD probabilistic modeling
result = reguq.run_probabilistic_advanced(
    data="data.csv",
    target_col="target",
    methods=["card", "ibug"]
)

# Run explainability analysis
result = reguq.run_explainability(
    data="data.csv",
    target_col="target",
    methods=["shap", "lime"]
)
```

---

## Documentation

- **[GUIDE.md](./GUIDE.md)** - Complete user guide with examples and tutorials
- **[SKILL.md](./SKILL.md)** - Technical architecture for developers
- **[CHANGELOG.md](./CHANGELOG.md)** - Version history
- **[RELEASE.md](./RELEASE.md)** - Release guidelines

---

## Citations & References

RegUQ builds on foundational research in uncertainty quantification. If you use this package, please cite the relevant papers:

### Gradient Boosting Models

**XGBoost**
> Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
> https://arxiv.org/abs/1603.02754

**LightGBM**
> Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
> https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

**CatBoost**
> Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS 2018*.
> https://arxiv.org/abs/1706.09516

**GPBoost**
> Sigrist, F. (2022). Gaussian Process Boosting. *JMLR 2022*.
> https://arxiv.org/abs/2004.02653

### Probabilistic Methods

**NGBoost**
> Duan, T., et al. (2020). NGBoost: Natural Gradient Boosting for Probabilistic Prediction. *ICML 2020*.
> https://arxiv.org/abs/1910.03225

**PGBM**
> Sprangers, O., et al. (2021). Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression. *KDD 2021*.
> https://arxiv.org/abs/2106.01682

**CARD (Diffusion-based UQ)**
> Han, X., et al. (2022). CARD: Classification and Regression Diffusion Models. *NeurIPS 2022*.
> https://arxiv.org/abs/2206.07275

**IBUG (Instance-Based UQ)**
> Brophy, J., et al. (2021). IBUG: Instance-Based Uncertainty Estimation for Gradient Boosted Regression Trees.
> https://arxiv.org/abs/2110.03260

**Treeffuser**
> Jolicoeur-Martineau, A., et al. (2024). Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees. *AISTATS 2024*.
> https://arxiv.org/abs/2309.09968

### Conformal Prediction

**MAPIE**
> Taquet, V., et al. (2022). MAPIE: An Open-Source Library for Distribution-Free Uncertainty Quantification.
> https://arxiv.org/abs/2207.12274

**PUNCC**
> Mendil, M., et al. (2023). PUNCC: A Python Library for Predictive Uncertainty Calibration and Conformalization. *DEEL-AI*.
> https://github.com/deel-ai/puncc

**Conformal Prediction Theory**
> Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.

**Adaptive Conformal Inference**
> Gibbs, I., & Candes, E. (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS 2021*.
> https://arxiv.org/abs/2106.00170

**Non-Exchangeable Conformal Prediction (NexCP)**
> Barber, R.F., et al. (2023). Conformal Prediction Beyond Exchangeability. *Annals of Statistics*.
> https://arxiv.org/abs/2202.13415

**Conformalized Quantile Regression (CQR)**
> Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized Quantile Regression. *NeurIPS 2019*.
> https://arxiv.org/abs/1905.03222

### Explainability

**SHAP**
> Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
> https://arxiv.org/abs/1705.07874

**LIME**
> Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD 2016*.
> https://arxiv.org/abs/1602.04938

**InterpretML**
> Nori, H., et al. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability.
> https://arxiv.org/abs/1909.09223

### Hyperparameter Optimization

**Optuna**
> Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD 2019*.
> https://arxiv.org/abs/1907.10902

### Neural Networks

**TabNet**
> Arik, S.O., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI 2021*.
> https://arxiv.org/abs/1908.07442

---

## API Reference

### Core Functions

```python
# Hyperparameter tuning with Optuna
reguq.run_tuning(data, target_col, models, tuning_config, output_config)

# Quantile regression
reguq.run_quantile(data, target_col, models, params_source, output_config)

# Probabilistic prediction (NGBoost, PGBM)
reguq.run_probabilistic(data, target_col, models, params_source, output_config)

# Standard conformal prediction (MAPIE, PUNCC)
reguq.run_conformal_standard(data, target_col, models, params_source, conformal_config, output_config)

# Advanced conformal prediction (NexCP, FACI, MFCS, CVPlus, CQR)
reguq.run_conformal_advanced(data, target_col, models, params_source, conformal_config, output_config)

# Advanced probabilistic (CARD, IBUG, Treeffuser)
reguq.run_probabilistic_advanced(data, target_col, models, params_source, output_config, methods, card_config)

# Explainability (SHAP, LIME, InterpretML)
reguq.run_explainability(data, target_col, models, params_source, output_config, methods)

# Config-driven pipeline
reguq.run_from_config(config_or_path)
```

### Advanced Classes

```python
# CARD diffusion model
from reguq import CARDRegressor
card = CARDRegressor(base_model, hidden_dim=128, T=50, epochs=200)

# IBUG uncertainty estimation
from reguq import IBUGRegressor
ibug = IBUGRegressor(base_model, n_neighbors=50)
```

---

## Repository Structure

```
reguq/
├── src/reguq/                    # Main package (v0.2.0)
│   ├── __init__.py               # Public API exports
│   ├── api.py                    # User-facing functions
│   ├── registry.py               # Model registry (9 models)
│   ├── tuning.py                 # Optuna hyperparameter tuning
│   ├── quantile.py               # Quantile regression
│   ├── probabilistic.py          # Standard probabilistic (NGBoost, PGBM)
│   ├── probabilistic_advanced.py # CARD, IBUG, Treeffuser
│   ├── conformal_standard.py     # MAPIE, PUNCC
│   ├── conformal_advanced.py     # NexCP, FACI, MFCS, CVPlus, CQR
│   ├── explainability.py         # SHAP, LIME, InterpretML
│   └── ...
├── Data_folder/                  # Original Jupyter notebooks
├── examples/                     # Usage examples
├── tests/                        # Test suite
└── pyproject.toml                # Package metadata
```

---

## Research Background

This framework originated from hydrological sediment load analysis but is **domain-agnostic**. It applies to:
- Environmental monitoring
- Financial forecasting
- Industrial sensor data
- Any regression with uncertainty requirements

**Key Research Contributions:**
1. **Automated Optimization**: Bayesian hyperparameter tuning with Optuna
2. **Interval Estimation**: Quantile regression for conditional bounds
3. **Full Distribution Modeling**: NGBoost, PGBM, CARD for distributional predictions
4. **Robust Uncertainty**: NexCP and Adaptive CP for non-exchangeable data
5. **Model Interpretation**: SHAP, LIME, InterpretML integration

---

## Contributors

This repository is a collaborative project developed under guidance of **Dr. Mahesh Pal** by:
- **Prakriti Bisht**
- **Danesh Selwal**

---

## License

MIT License - see [LICENSE](./LICENSE) for details.
