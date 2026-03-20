# RegUQ Technical Architecture & Skills Documentation

This document provides an in-depth technical overview of the RegUQ package architecture, implementation details, and the machine learning techniques employed.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Package Structure](#package-structure)
3. [Core Modules](#core-modules)
4. [Uncertainty Quantification Methods](#uncertainty-quantification-methods)
5. [Model Registry](#model-registry)
6. [Data Pipeline](#data-pipeline)
7. [Reporting System](#reporting-system)
8. [Configuration Management](#configuration-management)
9. [Testing Strategy](#testing-strategy)
10. [Extending the Package](#extending-the-package)

---

## Architecture Overview

RegUQ follows a modular, phase-based architecture where each uncertainty quantification method is implemented as an independent module with a consistent interface.

### Design Principles

1. **Modularity**: Each phase (tuning, quantile, probabilistic, conformal) is self-contained
2. **Composability**: Phases can be run independently or chained together
3. **Flexibility**: Support for both programmatic API and configuration-driven workflows
4. **Observability**: Rich output including metrics, plots, and Excel reports
5. **Colab-First**: Designed for reproducibility in Google Colab environments

### High-Level Flow

```
User Input (API/Config)
    ↓
Data Loading & Validation
    ↓
Model Registry & Parameter Resolution
    ↓
Phase Execution (Tuning/Quantile/Probabilistic/Conformal)
    ↓
Metrics Computation
    ↓
Report Generation (Excel/Plots/JSON)
    ↓
Structured Results
```

---

## Package Structure

```
src/reguq/
├── __init__.py           # Public API exports
├── api.py                # API entry points
├── runner.py             # Config-based runner and CLI
├── config.py             # Configuration loading and validation
├── data.py               # Data loading and bundling
├── preprocess.py         # Data preprocessing utilities
├── registry.py           # Model registry and validation
├── params.py             # Parameter resolution logic
├── tuning.py             # Hyperparameter tuning (Optuna)
├── quantile.py           # Quantile regression implementation
├── probabilistic.py      # Probabilistic regression (NGBoost, PGBM)
├── conformal_standard.py # Standard conformal prediction (MAPIE, PUNCC)
├── metrics.py            # Metrics computation
├── charts.py             # Chart generation
├── export.py             # Excel and file export
├── colab.py              # Colab bootstrap utilities
├── types.py              # Type definitions and dataclasses
└── constants.py          # Package constants

tests/
├── conftest.py           # Pytest fixtures
├── test_*.py             # Unit and integration tests
└── ...

Data_folder/              # Original notebook implementations
├── Data/                 # Sample datasets
├── HyperParameter_Tuning/
├── Quantile_Regression/
├── Probabilistic_Distribution/
├── Probabilistic_Distribution(CARD)/
├── Conformal_Predictions(MAPIE,PUNCC)/
└── Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)/

examples/
├── quickstart.py         # Minimal API example
├── pipeline_config.yaml  # Complete config example
└── reguq_colab_check.ipynb  # Colab validation notebook
```

---

## Core Modules

### 1. `api.py` - Public API

Provides the main user-facing functions:

```python
def run_tuning(data, target_col, models, tuning_config, output_config)
def run_quantile(data, target_col, models, params_source, output_config)
def run_probabilistic(data, target_col, models, params_source, output_config)
def run_conformal_standard(data, target_col, models, params_source, conformal_config, output_config)
def run_from_config(config_or_path)
```

**Key Features**:
- Consistent interface across all phases
- Automatic data bundle preparation
- Parameter resolution
- Result aggregation

---

### 2. `data.py` - Data Management

Handles data loading and bundling:

```python
@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    target_name: str

def prepare_data_bundle(data_spec, target_col, split_config) -> DataBundle
```

**Capabilities**:
- Load from CSV paths or DataFrames
- Auto-split single dataset (with temporal-safe defaults)
- Feature/target separation
- Validation and error handling

**Data Input Modes**:

1. **Train/Test Mode**:
   ```python
   data = {
       "train_path": "train.csv",
       "test_path": "test.csv"
   }
   ```

2. **Single Dataset Mode**:
   ```python
   data = {
       "data_path": "full_data.csv",
       "split": {"test_size": 0.2, "shuffle": False}
   }
   ```

3. **DataFrame Mode**:
   ```python
   data = {
       "train": train_df,
       "test": test_df
   }
   ```

---

### 3. `registry.py` - Model Registry

Centralized model configuration and validation:

```python
MODEL_REGISTRY = {
    "lightgbm": {
        "class": lgb.LGBMRegressor,
        "supports": ["quantile", "probabilistic", "conformal", "tuning"],
        "default_params": {...}
    },
    "xgboost": {...},
    "catboost": {...},
    "ngboost": {...},
    "pgbm": {...}
}

def validate_models(models, phase)
def list_supported_models(phase)
def get_model_class(model_name)
```

**Features**:
- Phase-specific model validation
- Default parameter sets
- Easy extension for new models

---

### 4. `params.py` - Parameter Resolution

Smart parameter loading and tuning:

```python
def resolve_model_params(
    model_name,
    params_source,
    data_bundle,
    target_col,
    tuning_config
)
```

**Resolution Modes**:

1. **`defaults`**: Use registry defaults
2. **`load_only`**: Load from saved JSON (error if missing)
3. **`load_or_tune`**: Load if available, otherwise tune
4. **`explicit`**: Use user-provided params

**Parameter Storage**:
```
outputs/tuning/
├── lightgbm_params.json
├── xgboost_params.json
└── ...
```

---

### 5. `tuning.py` - Hyperparameter Tuning

Optuna-based Bayesian optimization:

```python
def run_tuning(data, target_col, models, tuning_config, output_config) -> TuningResult
```

**Features**:
- Bayesian optimization with TPE sampler
- Median pruner for efficient search
- Cross-validation scoring
- Automatic parameter saving

**Tuning Configuration**:
```python
tuning_config = {
    "n_trials": 50,
    "cv": 5,
    "scoring": "neg_root_mean_squared_error",
    "random_state": 42,
    "timeout": 3600  # seconds
}
```

**Search Spaces**:
Each model has predefined search spaces in `tuning.py`:
```python
def _lightgbm_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        ...
    }
```

---

### 6. `quantile.py` - Quantile Regression

Pinball loss-based quantile prediction:

```python
def run_quantile(data, target_col, models, params_source, output_config) -> PhaseResult
```

**Implementation**:
- Uses model-specific quantile loss (e.g., `objective='quantile'` for LightGBM)
- Trains separate models for lower (α=0.05) and upper (α=0.95) quantiles
- Combines with point prediction (α=0.5)

**Metrics**:
- RMSE (point prediction)
- MAE (point prediction)
- Coverage rate (percentage of actuals within interval)
- Mean interval width
- Pinball loss

**Output Format**:
```python
predictions_df = pd.DataFrame({
    "y_true": [...],
    "y_pred": [...],
    "lower_bound": [...],
    "upper_bound": [...]
})
```

---

### 7. `probabilistic.py` - Probabilistic Regression

Full distribution modeling:

```python
def run_probabilistic(data, target_col, models, params_source, output_config) -> PhaseResult
```

**Supported Models**:

1. **NGBoost** (Natural Gradient Boosting):
   - Learns parameters of probability distributions
   - Supports Normal, LogNormal, Exponential distributions
   - Natural gradient descent for distribution parameters

2. **PGBM** (Probabilistic Gradient Boosting Machines):
   - Bayesian approach to gradient boosting
   - Full posterior distribution over predictions
   - Captures epistemic + aleatoric uncertainty

**Metrics**:
- **NLL** (Negative Log-Likelihood): Measures probabilistic accuracy
- **CRPS** (Continuous Ranked Probability Score): Proper scoring rule
- **Calibration**: PIT histogram uniformity

**Calibration Assessment**:
Uses Probability Integral Transform (PIT):
```python
pit = norm.cdf(y_true, loc=mu_pred, scale=sigma_pred)
# Should be uniform [0,1] if well-calibrated
```

---

### 8. `conformal_standard.py` - Conformal Prediction

Distribution-free uncertainty quantification:

```python
def run_conformal_standard(
    data, target_col, models, params_source, conformal_config, output_config
) -> ConformalResult
```

**Methods**:

1. **MAPIE** (Model Agnostic Prediction Interval Estimator):
   - `naive`: Simple split conformal
   - `plus`: Jackknife+
   - `cv+`: Cross-validation+

2. **PUNCC** (Predictive Uncertainty Calibration and Conformalization):
   - Advanced conformal methods
   - Supports various nonconformity scores

**Theory**:
Conformal prediction provides coverage guarantee:
```
P(Y_test ∈ [L(X_test), U(X_test)]) ≥ 1 - α
```
where α is the miscoverage rate (e.g., 0.1 for 90% coverage).

**Implementation**:
```python
from mapie.regression import MapieRegressor

mapie = MapieRegressor(estimator=base_model, method="plus", cv=5)
mapie.fit(X_train, y_train)
y_pred, y_pis = mapie.predict(X_test, alpha=alpha)
```

**Metrics**:
- Actual coverage rate
- Mean prediction interval width
- Coverage by prediction region

---

### 9. `metrics.py` - Metrics Computation

Unified metrics calculation:

```python
def compute_quantile_metrics(y_true, y_pred, y_lower, y_upper) -> dict
def compute_probabilistic_metrics(y_true, mu_pred, sigma_pred) -> dict
def compute_conformal_metrics(y_true, y_pred, y_lower, y_upper, alpha) -> dict
```

**Implemented Metrics**:

- **Regression**: RMSE, MAE, R², MAPE
- **Interval**: Coverage rate, mean width, width std
- **Probabilistic**: NLL, CRPS, calibration error
- **Conformal**: Actual vs target coverage, conditional coverage

---

### 10. `charts.py` - Visualization

Comprehensive chart generation:

```python
def generate_phase_charts(
    phase, model_name, predictions_df, output_dir, chart_config
) -> Dict[str, Path]
```

**Chart Types by Phase**:

**Quantile**:
- Predictions vs Actual (scatter with intervals)
- Residuals plot
- QQ plot
- Interval coverage over predictions

**Probabilistic**:
- Calibration plot (predicted vs actual quantiles)
- PIT histogram
- CRPS evolution
- Uncertainty vs error scatter

**Conformal**:
- Coverage evolution over samples
- Interval width distribution
- Predictions with confidence bands

**Styling**:
```python
chart_config = {
    "chart_detail_level": "detailed",  # or "summary"
    "legend_position": "upper right",
    "style_overrides": {
        "figure.figsize": [12, 8],
        "font.size": 11,
        "axes.grid": True
    }
}
```

---

### 11. `export.py` - Report Generation

Excel and file export:

```python
def export_quantile_excel(model_name, predictions_df, metrics, output_dir, config)
def export_probabilistic_excel(...)
def embed_charts_in_excel(excel_path, chart_paths, sheet_map)
```

**Excel Structure**:

```
Sheet: Predictions
- Index | y_true | y_pred | lower_bound | upper_bound | residual

Sheet: Metrics
- Metric | Value
- RMSE | 0.123
- Coverage | 0.91
- ...

Sheet: Charts (optional)
- Embedded plot images
```

**Chart Embedding**:
Uses OpenPyXL to embed PNG images:
```python
from openpyxl.drawing.image import Image

img = Image(chart_path)
ws.add_image(img, f"A{row}")
```

---

## Uncertainty Quantification Methods

### Quantile Regression

**Mathematical Formulation**:

Minimize pinball loss:
```
L_α(y, q) = (y - q) * (α - 𝟙{y < q})
```

where:
- α is the target quantile (e.g., 0.05, 0.95)
- q is the predicted quantile
- 𝟙 is the indicator function

**Advantages**:
- Model-agnostic (any model supporting quantile loss)
- Directly estimates prediction intervals
- No distribution assumptions

**Limitations**:
- Quantiles may cross (q₀.₀₅ > q₀.₉₅)
- Fixed quantile levels
- No full distributional information

---

### Probabilistic Models

#### NGBoost (Natural Gradient Boosting)

**Key Innovation**:
Uses natural gradient to directly optimize distribution parameters.

**Process**:
1. Initialize base distribution (e.g., Normal(μ₀, σ₀))
2. For each boosting iteration:
   - Compute natural gradient w.r.t. distribution parameters
   - Fit base learner to natural gradient
   - Update distribution parameters

**Formula**:
```
θₜ = θₜ₋₁ + η * base_learner(natural_gradient)
```

**Supported Distributions**:
- Normal
- LogNormal
- Exponential
- Poisson (for counts)

#### PGBM (Probabilistic Gradient Boosting Machines)

**Key Innovation**:
Bayesian inference over tree structure and parameters.

**Process**:
1. Place prior over tree parameters
2. Use variational inference to approximate posterior
3. Predict with full posterior distribution

**Uncertainty Decomposition**:
- Aleatoric: Irreducible data noise
- Epistemic: Model uncertainty (reduces with more data)

---

### Conformal Prediction

**Core Theorem** (Vovk et al.):

For any black-box model, if calibration and test data are exchangeable:
```
P(Yₙ₊₁ ∈ C(Xₙ₊₁)) ≥ 1 - α
```

**Split Conformal**:

1. Split data: train (n₁) and calibration (n₂)
2. Train model on train set
3. Compute nonconformity scores on calibration:
   ```
   sᵢ = |yᵢ - ŷᵢ|
   ```
4. Find quantile: q = Quantile(s, 1-α)
5. Prediction interval: ŷ ± q

**Jackknife+**:

Provides tighter intervals by using cross-validation:

1. Train K models via K-fold CV
2. For each calibration point, use out-of-fold prediction
3. Compute nonconformity scores
4. Construct interval using quantile

**Advantages**:
- Finite-sample coverage guarantee
- Distribution-free
- Model-agnostic

**Limitations**:
- Requires exchangeability assumption
- Intervals may be conservative
- Single target coverage (marginal, not conditional)

---

## Data Pipeline

### Data Flow

```
Input (CSV/DataFrame)
    ↓
Validation (columns, types, missing)
    ↓
Feature/Target Split
    ↓
Train/Test Split (if needed)
    ↓
DataBundle Creation
    ↓
Model Training
    ↓
Predictions
    ↓
Metrics & Reports
```

### Data Validation

Checks performed:
- Target column exists
- No all-NaN columns
- Sufficient samples (min 10)
- Feature/target separation

### Preprocessing

**Current Implementation**:
- Automatic numeric feature selection
- Target extraction
- Index preservation for time series

**Future Enhancements**:
- Categorical encoding
- Feature scaling
- Missing value imputation
- Feature engineering

---

## Reporting System

### Output Structure

```
outputs/
└── run_name/
    ├── model_name/
    │   ├── predictions.csv
    │   ├── metrics.json
    │   ├── report.xlsx (with embedded charts)
    │   └── plots/
    │       ├── predictions_vs_actual.png
    │       ├── residuals.png
    │       └── ...
    └── summary.json
```

### Report Components

1. **Predictions CSV**:
   - Raw predictions for post-processing
   - Easy to load in other tools

2. **Metrics JSON**:
   - Structured metrics for programmatic access
   - Versioned format for compatibility

3. **Excel Report**:
   - Human-readable summary
   - Embedded charts (if enabled)
   - Multiple sheets for organization

4. **Plots**:
   - High-resolution PNG images
   - Publication-ready quality
   - Customizable styling

---

## Configuration Management

### Configuration Schema

```python
@dataclass
class OutputConfig:
    output_dir: str
    export_excel: bool = True
    export_plots: bool = True
    embed_excel_charts: bool = False
    show_inline_plots: bool = False
    chart_detail_level: str = "detailed"
    legend_position: str = "best"
    save_json: bool = True
    style_overrides: Optional[dict] = None

@dataclass
class ParamsSourceConfig:
    mode: str  # "defaults", "load_only", "load_or_tune", "explicit"
    params: Optional[dict] = None
    params_dir: Optional[str] = None

@dataclass
class SplitConfig:
    test_size: float = 0.2
    shuffle: bool = False  # False for time series
    random_state: Optional[int] = 42
```

### YAML to Python

```python
def load_config(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)

def coerce_config(raw_config: dict) -> dict:
    # Convert YAML config to proper Python types
    # Handle aliases (e.g., "report" -> "output")
    # Fill in defaults
    return coerced_config
```

---

## Testing Strategy

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_data.py          # Data loading tests
├── test_params.py        # Parameter resolution tests
├── test_phases.py        # Phase execution tests
├── test_config.py        # Config loading tests
├── test_charts.py        # Chart generation tests
├── test_export.py        # Export tests
└── test_runner.py        # Integration tests
```

### Test Categories

1. **Unit Tests**:
   - Individual function behavior
   - Edge cases and error handling
   - Mock external dependencies

2. **Integration Tests**:
   - End-to-end phase execution
   - Config-based workflows
   - Output generation

3. **Smoke Tests**:
   - Quick validation of core functionality
   - Run with minimal data
   - Fast execution (<1s per test)

### Fixtures

```python
@pytest.fixture
def sample_data():
    """Generate synthetic regression data."""
    X = np.random.randn(100, 5)
    y = X.sum(axis=1) + np.random.randn(100) * 0.1
    return pd.DataFrame(X), pd.Series(y)

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path / "test_outputs"
```

### Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_phases.py

# Run with coverage
pytest --cov=reguq --cov-report=html

# Run fast tests only
pytest -m "not slow"
```

---

## Extending the Package

### Adding a New Model

1. **Update Registry** (`registry.py`):
```python
MODEL_REGISTRY["new_model"] = {
    "class": NewModelRegressor,
    "supports": ["quantile", "probabilistic"],
    "default_params": {
        "n_estimators": 100,
        ...
    }
}
```

2. **Add Search Space** (`tuning.py`):
```python
def _new_model_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        ...
    }

SEARCH_SPACES["new_model"] = _new_model_search_space
```

3. **Add Tests**:
```python
def test_new_model_quantile():
    result = run_quantile(
        data=sample_data,
        target_col="y",
        models=["new_model"],
        ...
    )
    assert "new_model" in result.metrics
```

---

### Adding a New Phase

1. **Create Module** (`src/reguq/new_phase.py`):
```python
def run_new_phase(data, target_col, models, phase_config, output_config):
    bundle = prepare_data_bundle(data, target_col)
    results = {}

    for model_name in models:
        # Train model
        # Make predictions
        # Compute metrics
        # Generate reports
        results[model_name] = {...}

    return PhaseResult(
        phase="new_phase",
        predictions=predictions,
        metrics=metrics
    )
```

2. **Export in API** (`api.py`):
```python
from .new_phase import run_new_phase

__all__ = [..., "run_new_phase"]
```

3. **Add to Runner** (`runner.py`):
```python
PHASE_HANDLERS["new_phase"] = run_new_phase
```

4. **Update Config Schema** (`config.py`):
```python
def validate_config(config):
    if "new_phase" in config.get("phases", []):
        assert "new_phase_config" in config
```

---

### Adding New Metrics

1. **Implement Metric** (`metrics.py`):
```python
def compute_new_metric(y_true, y_pred, **kwargs):
    """Compute new metric.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        **kwargs: Additional parameters

    Returns:
        Metric value (float)
    """
    return metric_value
```

2. **Integrate in Phase**:
```python
metrics["new_metric"] = compute_new_metric(y_true, y_pred)
```

3. **Add Tests**:
```python
def test_new_metric():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    metric = compute_new_metric(y_true, y_pred)
    assert metric > 0
```

---

### Adding New Charts

1. **Implement Chart** (`charts.py`):
```python
def plot_new_visualization(data, output_path, config):
    """Generate new visualization.

    Args:
        data: Data for plotting
        output_path: Path to save plot
        config: Chart configuration

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=config.get("figsize", (10, 6)))

    # Create plot
    ax.plot(...)

    # Style
    ax.set_xlabel(...)
    ax.set_ylabel(...)
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path
```

2. **Integrate in Phase Charts**:
```python
def generate_phase_charts(phase, model_name, data, output_dir, config):
    chart_paths = {}

    # Existing charts
    ...

    # New chart
    chart_paths["new_viz"] = plot_new_visualization(
        data, output_dir / "new_viz.png", config
    )

    return chart_paths
```

---

## Performance Considerations

### Computational Complexity

**Tuning Phase**:
- Time: O(n_trials × n_cv_folds × n_samples × model_complexity)
- Most expensive phase
- Parallelizable with Optuna

**Quantile Phase**:
- Time: O(n_quantiles × n_samples × model_complexity)
- Train 3 models (lower, median, upper)
- Sequential execution

**Probabilistic Phase**:
- Time: O(n_samples × model_complexity)
- Single model per base estimator
- NGBoost: Similar to regular boosting
- PGBM: Additional inference overhead

**Conformal Phase**:
- Time: O(n_calibration + n_test)
- Lightweight once model is trained
- Main cost in base model training

### Memory Optimization

1. **Streaming Predictions**:
   - Process in batches for large test sets
   - Avoid loading all predictions in memory

2. **Model Persistence**:
   - Save models to disk after training
   - Load on-demand for inference

3. **Chart Generation**:
   - Generate charts incrementally
   - Close figures after saving

---

## Dependencies

### Core Dependencies

```toml
[project.dependencies]
numpy = "1.26.4"
pandas = "2.2.3"
scikit-learn = "1.6.1"
scipy = "1.14.1"
matplotlib = "3.10.1"
PyYAML = "6.0.2"
```

### Model Libraries

```toml
lightgbm = "4.6.0"
xgboost = "3.1.2"
catboost = "1.2.8"
ngboost = "0.5.8"
pgbm = "2.2.0"
```

### Uncertainty Quantification

```toml
mapie = "0.6.0"    # Conformal prediction
puncc = "0.8.0"    # Advanced conformal methods
properscoring = "0.1"  # CRPS and other proper scores
```

### Optimization

```toml
optuna = "4.6.0"   # Hyperparameter tuning
```

### Export

```toml
openpyxl = "3.1.5"      # Excel reading/writing
XlsxWriter = "3.2.9"    # Excel writing (alternative)
```

---

## Future Enhancements

### Planned Features

1. **Advanced Conformal Methods**:
   - NEXCP (Non-Exchangeable Conformal Prediction)
   - Adaptive Conformal Prediction
   - Time series conformal methods

2. **CARD Integration**:
   - Diffusion-based distribution modeling
   - Support for complex conditional distributions

3. **Model Ensembles**:
   - Weighted ensemble predictions
   - Ensemble uncertainty aggregation

4. **Feature Engineering**:
   - Automated feature selection
   - Polynomial features
   - Interaction terms

5. **Monitoring**:
   - Coverage monitoring over time
   - Distribution shift detection
   - Model degradation alerts

6. **Calibration**:
   - Post-hoc calibration methods
   - Isotonic regression
   - Platt scaling for intervals

---

## References

### Uncertainty Quantification

- Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association*.

- Hersbach, H. (2000). "Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems." *Weather and Forecasting*.

### Conformal Prediction

- Vovk, V., Gammerman, A., & Shafer, G. (2005). "Algorithmic Learning in a Random World." Springer.

- Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression." *NeurIPS*.

- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). "Predictive Inference with the Jackknife+." *Annals of Statistics*.

### Probabilistic Models

- Duan, T., Avati, A., Ding, D. Y., et al. (2020). "NGBoost: Natural Gradient Boosting for Probabilistic Prediction." *ICML*.

- Sprangers, O., Schelter, S., & de Rijke, M. (2021). "Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression." *KDD*.

### Quantile Regression

- Koenker, R., & Bassett Jr, G. (1978). "Regression Quantiles." *Econometrica*.

---

**Document Version**: 1.0
**Last Updated**: March 2026
**Maintainers**: Prakriti Bisht, Danesh Selwal
