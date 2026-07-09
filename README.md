# RegUQ: Regression Uncertainty Quantification Library

RegUQ is a professional, high-performance Python library for regression uncertainty quantification (UQ). It provides state-of-the-art implementations of **conformal prediction**, **probabilistic forecasting**, **quantile regression**, and **hyperparameter tuning**, standardizing complex UQ methods under a clean, unified API.

---

## 🛠️ Installation

```bash
pip install reguq
```

For advanced conformal methods and deep learning wrappers (e.g., AlphaNet, FACI):
```bash
pip install reguq[advanced]
```

---

## 🏗️ Architecture Overview

The following diagram illustrates the components and execution flow within RegUQ:

```text
       +-------------------------------------------------------+
       |                     User Input                        |
       |             (Data Frame & Target Column)              |
       +---------------------------+---------------------------+
                                   |
                                   v
       +---------------------------+---------------------------+
       |                  Data Preprocessing                   |
       |                (Train/Test/Calib Split)               |
       +---------------------------+---------------------------+
                                   |
                                   +-------------------+-------------------+
                                   |                   |                   |
                                   v                   v                   v
       +---------------------------+   +---------------+---------------+   +---------------+---------------+
       |    Hyperparameter Tuning  |   |      Quantile Regression      |   |    Probabilistic Models       |
       |      (Optuna TPE Tuning)  |   |     (QuantileRegressor)       |   |  (ProbabilisticRegressor)     |
       +---------------------------+   +---------------+---------------+   +---------------+---------------+
                                                       |                                   |
                                                       +-----------------+-----------------+
                                                                         |
                                                                         v
                                                       +-----------------+-----------------+
                                                       |        Conformal Prediction       |
                                                       |      (Standard & Advanced CP)     |
                                                       +-----------------+-----------------+
                                                                         |
                                                                         v
                                                       +-----------------+-----------------+
                                                       |         Unified Outputs           |
                                                       |   (Excel Metrics, Plotly Charts)  |
                                                       +-----------------------------------+
```

---

## 🚀 Quickstart Examples

### 1. Object-Oriented Wrapper API
RegUQ provides scikit-learn compatible object-oriented wrappers for UQ models.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from reguq.models.conformal import ConformalRegressor

# Generate dummy data
X_train = np.random.randn(100, 5)
y_train = 2.0 * X_train[:, 0] + np.random.randn(100)
X_test = np.random.randn(20, 5)

# Initialize a base point regressor
base_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Wrap it with ConformalRegressor (using Conformalized Quantile Regression "cqr")
uq_model = ConformalRegressor(base_estimator=base_model, method="cqr")

# Fit and predict
uq_model.fit(X_train, y_train)
y_pred, y_lower, y_upper = uq_model.predict(X_test, alpha=0.1)

# Plot predictions and intervals
uq_model.plot_predictions(X_test, alpha=0.1, backend="matplotlib")
```

### 2. Multi-Phase Pipeline API
Execute multi-phase runs programmatically or via configuration dictionaries/YAML files.

```python
from reguq.runner import run_from_config

# Configuration defining the runs and export settings
pipeline_config = {
    "data": {
        "train_path": "data/concrete_train.csv",
        "test_path": "data/concrete_test.csv",
        "target_col": "strength"
    },
    "models": ["randomforest"],
    "phases": ["tuning", "quantile", "conformal_standard"],
    "tuning": {
        "n_trials": 20,
        "metric": "mae"
    },
    "output": {
        "output_dir": "outputs/concrete_run",
        "export_excel": True,
        "export_plots": True
    }
}

# Run the entire pipeline
results = run_from_config(pipeline_config)
print("Pipeline run completed successfully. Outputs saved to outputs/concrete_run.")
```

---

## 📊 API Summary Table

| Phase / Module | Class / Function | Purpose | Input Parameters | Output Type |
| :--- | :--- | :--- | :--- | :--- |
| **Tuning** | `run_tuning` | Automated hyperparameter optimization | `data`, `target_col`, `tuning_config` | `TuningResult` |
| **Quantile** | `run_quantile` | Quantile regression bounds training | `data`, `target_col`, `quantiles` | `PhaseResult` |
| **Probabilistic** | `run_probabilistic` | Distribution parameter estimation | `data`, `target_col`, `alpha` | `PhaseResult` |
| **Standard Conformal** | `run_conformal_standard` | Calibration via MAPIE & PUNCC SplitCP | `data`, `target_col`, `conformal_config` | `ConformalResult` |
| **Advanced Conformal** | `run_conformal_advanced` | Advanced CP (NexCP, FACI, CQR, AlphaNet) | `data`, `target_col`, `conformal_config` | `ConformalResult` |
| **Pipeline Runner** | `run_from_config` | Unified multi-phase pipeline execution | `config_or_path` | `PipelineRunResult` |

---

## 📖 Documentation & Examples

- **User Manual**: Detailed mathematical background and API usage are documented in the [RegUQ PDF Manual](docs/reguq_manual.pdf).
- **Interactive Notebooks & Examples**: Browse the [examples/](examples/) directory for end-to-end scenarios including concrete strength prediction, flood risk estimation, and environmental noise safety.
