# RegUQ User Guide

**RegUQ** (Regression Uncertainty Quantification) is a comprehensive Python package that transforms your regression uncertainty quantification workflow from Jupyter notebooks into a production-ready library. This guide will help you understand and use the package effectively.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Workflow Phases](#workflow-phases)
7. [Examples](#examples)
8. [Google Colab Usage](#google-colab-usage)
9. [Output and Reporting](#output-and-reporting)
10. [Advanced Topics](#advanced-topics)

---

## Quick Start

Here's a minimal example to get you started:

```python
from reguq import run_quantile

# Run quantile regression with default settings
result = run_quantile(
    data={
        "train_path": "./Data_folder/Data/train.csv",
        "test_path": "./Data_folder/Data/test.csv",
    },
    target_col="target",
    models=["lightgbm", "xgboost"],
    params_source={"mode": "defaults"},
    output_config={
        "output_dir": "./outputs/my_run",
        "export_excel": True,
        "export_plots": True,
    }
)

print(f"Results: {result.metrics}")
```

---

## Installation

### From GitHub (Recommended)

```bash
pip install "git+https://github.com/DaneshSelwal/reguq.git"
```

### For Development

```bash
git clone https://github.com/DaneshSelwal/reguq.git
cd reguq
pip install -e .
```

### Google Colab

See the [Google Colab Usage](#google-colab-usage) section for special Colab instructions.

---

## Core Concepts

### What is RegUQ?

RegUQ provides uncertainty quantification methods for regression tasks. Instead of just predicting a single point estimate, it provides:

- **Prediction Intervals**: Upper and lower bounds for predictions
- **Probability Distributions**: Full distributional forecasts
- **Conformal Predictions**: Distribution-free coverage guarantees
- **Adaptive Methods**: For handling distribution shift and temporal data

### Pipeline Architecture

The RegUQ pipeline consists of multiple phases:

1. **Hyperparameter Tuning**: Optimize model parameters using Optuna
2. **Quantile Regression**: Predict conditional quantiles (e.g., 5th and 95th percentiles)
3. **Probabilistic Models**: Full distribution modeling with NGBoost and PGBM
4. **Conformal Prediction**: Distribution-free uncertainty quantification

### Supported Models

RegUQ supports the following models:

- **LightGBM**: Fast gradient boosting
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Gradient boosting with categorical support
- **NGBoost**: Natural gradient boosting for probabilistic predictions
- **PGBM**: Probabilistic gradient boosting machines

---

## API Reference

### Main Functions

#### `run_tuning()`

Perform hyperparameter tuning using Optuna.

```python
from reguq import run_tuning

result = run_tuning(
    data={"train_path": "train.csv", "test_path": "test.csv"},
    target_col="target",
    models=["lightgbm", "xgboost", "catboost"],
    tuning_config={
        "n_trials": 50,
        "cv": 5,
        "scoring": "neg_root_mean_squared_error",
        "random_state": 42
    },
    output_config={
        "output_dir": "./outputs/tuning",
        "save_params": True
    }
)
```

**Parameters:**
- `data`: Dictionary with train/test paths or DataFrames
- `target_col`: Name of the target column
- `models`: List of model names to tune
- `tuning_config`: Configuration for Optuna tuning
- `output_config`: Output directory and export settings

**Returns:** `TuningResult` with optimized parameters for each model

---

#### `run_quantile()`

Run quantile regression to predict conditional quantiles.

```python
from reguq import run_quantile

result = run_quantile(
    data={"train_path": "train.csv", "test_path": "test.csv"},
    target_col="target",
    models=["lightgbm", "xgboost"],
    params_source={
        "mode": "load_or_tune",  # or "defaults", "load_only"
        "params_dir": "./outputs/tuning"
    },
    output_config={
        "output_dir": "./outputs/quantile",
        "export_excel": True,
        "export_plots": True,
        "embed_excel_charts": True,
        "chart_detail_level": "detailed"
    }
)
```

**Parameters:**
- `data`: Dictionary with train/test paths or DataFrames
- `target_col`: Name of the target column
- `models`: List of model names
- `params_source`: How to obtain model parameters
  - `"defaults"`: Use default parameters
  - `"load_only"`: Load from saved params
  - `"load_or_tune"`: Load if available, otherwise tune
- `output_config`: Output settings including Excel and plot exports

**Returns:** `PhaseResult` with predictions and metrics

---

#### `run_probabilistic()`

Run probabilistic regression to predict full probability distributions.

```python
from reguq import run_probabilistic

result = run_probabilistic(
    data={"train_path": "train.csv", "test_path": "test.csv"},
    target_col="target",
    models=["ngboost", "pgbm"],
    params_source={"mode": "defaults"},
    output_config={
        "output_dir": "./outputs/probabilistic",
        "export_excel": True,
        "export_plots": True
    }
)
```

**Parameters:**
- Similar to `run_quantile()`
- Models should support probabilistic predictions (ngboost, pgbm)

**Returns:** `PhaseResult` with distributional predictions and calibration metrics

---

#### `run_conformal_standard()`

Run standard conformal prediction methods (MAPIE, PUNCC).

```python
from reguq import run_conformal_standard

result = run_conformal_standard(
    data={"train_path": "train.csv", "test_path": "test.csv"},
    target_col="target",
    models=["lightgbm", "xgboost"],
    params_source={"mode": "defaults"},
    conformal_config={
        "alpha": 0.1,  # 90% coverage
        "methods": ["mapie", "puncc"],
        "mapie_method": "plus"  # or "naive", "cv+"
    },
    output_config={
        "output_dir": "./outputs/conformal",
        "export_excel": True
    }
)
```

**Parameters:**
- `conformal_config`: Conformal prediction settings
  - `alpha`: Miscoverage rate (0.1 = 90% coverage)
  - `methods`: List of conformal methods to use
  - `mapie_method`: MAPIE-specific method

**Returns:** `ConformalResult` with prediction intervals and coverage metrics

---

#### `run_from_config()`

Run a complete pipeline from a YAML configuration file.

```python
from reguq import run_from_config

result = run_from_config("config.yaml")
# or
result = run_from_config({
    "data": {"train_path": "train.csv", "test_path": "test.csv"},
    "target_col": "target",
    "models": ["lightgbm", "xgboost"],
    "phases": ["tuning", "quantile", "probabilistic"]
})
```

**Parameters:**
- `config_or_path`: Path to YAML file or configuration dictionary

**Returns:** `PipelineRunResult` with results from all phases

---

## Configuration

### YAML Configuration Format

Create a `config.yaml` file:

```yaml
data:
  train_path: ./Data_folder/Data/train.csv
  test_path: ./Data_folder/Data/test.csv
  target_col: target

models:
  - lightgbm
  - xgboost
  - catboost

phases:
  - tuning
  - quantile
  - probabilistic
  - conformal_standard

tuning:
  n_trials: 50
  cv: 5
  scoring: neg_root_mean_squared_error
  random_state: 42

params_source:
  mode: load_or_tune

quantile:
  quantiles: [0.05, 0.95]

probabilistic:
  alpha: 0.1

conformal_standard:
  alpha: 0.1
  methods: [mapie, puncc]
  mapie_method: plus

output:
  output_dir: ./outputs/pipeline_run
  export_excel: true
  export_plots: true
  embed_excel_charts: true
  show_inline_plots: false
  chart_detail_level: detailed
  legend_position: upper right
  save_json: true
```

### CLI Usage

Run the pipeline from command line:

```bash
reguq-run --config config.yaml
```

---

## Workflow Phases

### Phase 1: Hyperparameter Tuning

**Purpose**: Optimize model hyperparameters using Bayesian optimization.

**Implementation**: Uses Optuna with efficient pruning strategies.

**Output**:
- Optimized parameters saved to JSON
- Optimization history plots
- Best trial metrics

**When to use**:
- Starting a new project
- When default parameters are not sufficient
- To maximize model performance

---

### Phase 2: Quantile Regression

**Purpose**: Predict conditional quantiles to create prediction intervals.

**Method**: Uses pinball loss to estimate specific quantiles (e.g., 5th, 95th percentiles).

**Output**:
- Point predictions
- Lower and upper bounds
- Coverage metrics
- Prediction interval width

**Use cases**:
- Risk assessment
- Anomaly detection
- Decision making under uncertainty

---

### Phase 3: Probabilistic Distribution

**Purpose**: Predict the full conditional probability distribution.

**Models**:
- **NGBoost**: Natural gradient boosting with various distributions (Normal, LogNormal, etc.)
- **PGBM**: Probabilistic gradient boosting using Bayesian inference

**Metrics**:
- **NLL** (Negative Log-Likelihood): Lower is better
- **CRPS** (Continuous Ranked Probability Score): Measures probabilistic accuracy
- **PIT** (Probability Integral Transform): Assesses calibration

**Output**:
- Mean and variance predictions
- Calibration plots
- Reliability diagrams

---

### Phase 4: Standard Conformal Prediction

**Purpose**: Provide distribution-free coverage guarantees.

**Methods**:
- **MAPIE**: Various conformal methods (naive, plus, cv+)
- **PUNCC**: Advanced conformal prediction library

**Guarantee**: Valid coverage under exchangeability assumption.

**Output**:
- Prediction intervals with coverage guarantees
- Actual coverage rates
- Interval widths

**When to use**:
- When you need guaranteed coverage
- Distribution-free uncertainty quantification
- Model-agnostic predictions

---

## Examples

### Example 1: Basic Quantile Regression

```python
from reguq import run_quantile

result = run_quantile(
    data={
        "train_path": "./data/train.csv",
        "test_path": "./data/test.csv",
    },
    target_col="price",
    models=["lightgbm"],
    params_source={"mode": "defaults"},
    output_config={
        "output_dir": "./outputs/basic_quantile",
        "export_excel": True
    }
)

# Access results
print(f"RMSE: {result.metrics['lightgbm']['rmse']}")
print(f"Coverage: {result.metrics['lightgbm']['coverage']}")
```

---

### Example 2: Complete Pipeline with Tuning

```python
from reguq import run_tuning, run_quantile, run_probabilistic

# Step 1: Tune hyperparameters
tuning_result = run_tuning(
    data={"data_path": "./data/full_data.csv"},
    target_col="target",
    models=["lightgbm", "xgboost"],
    tuning_config={
        "n_trials": 50,
        "cv": 5
    },
    output_config={"output_dir": "./outputs/tuning"}
)

# Step 2: Run quantile regression with tuned params
quantile_result = run_quantile(
    data={"data_path": "./data/full_data.csv"},
    target_col="target",
    models=["lightgbm", "xgboost"],
    params_source={
        "mode": "load_only",
        "params_dir": "./outputs/tuning"
    },
    output_config={"output_dir": "./outputs/quantile"}
)

# Step 3: Run probabilistic models
prob_result = run_probabilistic(
    data={"data_path": "./data/full_data.csv"},
    target_col="target",
    models=["ngboost"],
    params_source={"mode": "defaults"},
    output_config={"output_dir": "./outputs/probabilistic"}
)
```

---

### Example 3: Using DataFrames Instead of Files

```python
import pandas as pd
from reguq import run_quantile

# Load your data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

result = run_quantile(
    data={
        "train": train_df,
        "test": test_df
    },
    target_col="y",
    models=["xgboost"],
    params_source={"mode": "defaults"},
    output_config={"output_dir": "./outputs/df_example"}
)
```

---

### Example 4: Config-Based Workflow

Create `my_config.yaml`:

```yaml
data:
  data_path: ./data/full_data.csv
  target_col: sales
  split:
    test_size: 0.2
    shuffle: false

models:
  - lightgbm
  - catboost

phases:
  - quantile
  - conformal_standard

output:
  output_dir: ./outputs/sales_forecast
  export_excel: true
  embed_excel_charts: true
```

Run it:

```python
from reguq import run_from_config

result = run_from_config("my_config.yaml")
```

Or from CLI:

```bash
reguq-run --config my_config.yaml
```

---

## Google Colab Usage

RegUQ is designed with Colab-first compatibility.

### Installation in Colab

```python
# Install the package
!pip install --upgrade --force-reinstall --no-cache-dir "git+https://github.com/DaneshSelwal/reguq.git"

# Bootstrap Colab environment (one-time setup)
from reguq import bootstrap_colab_environment

# This will install dependencies and restart runtime automatically
bootstrap_colab_environment(
    repo_url="https://github.com/DaneshSelwal/reguq.git"
)
```

**Important**: After `bootstrap_colab_environment()` runs for the first time, the Colab runtime will automatically restart. This ensures all dependencies are properly loaded.

### After Runtime Restart

```python
# After restart, import and use normally
from reguq import run_quantile

result = run_quantile(
    data={
        "train_path": "/content/drive/MyDrive/data/train.csv",
        "test_path": "/content/drive/MyDrive/data/test.csv"
    },
    target_col="target",
    models=["lightgbm"],
    params_source={"mode": "defaults"},
    output_config={
        "output_dir": "/content/drive/MyDrive/outputs",
        "export_excel": True,
        "show_inline_plots": True  # Show plots directly in notebook
    }
)
```

### Tips for Colab

1. **Mount Google Drive** to persist outputs:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Use `show_inline_plots`** to display plots in notebook cells:
   ```python
   output_config = {
       "show_inline_plots": True,
       "export_plots": True
   }
   ```

3. **Check the validation notebook**: See `examples/reguq_colab_check.ipynb` for a complete Colab example.

---

## Output and Reporting

### Output Structure

When you run a phase, RegUQ creates structured outputs:

```
outputs/
└── my_run/
    ├── lightgbm/
    │   ├── predictions.csv
    │   ├── metrics.json
    │   ├── plots/
    │   │   ├── predictions_vs_actual.png
    │   │   ├── residuals.png
    │   │   └── interval_coverage.png
    │   └── report.xlsx
    ├── xgboost/
    │   └── ...
    └── summary.json
```

### Excel Reports

Excel reports include:
- Predictions with intervals
- Metrics summary
- Embedded charts (when `embed_excel_charts=True`)

### Chart Types

Depending on the phase, you'll get:

**Quantile Regression**:
- Predictions vs Actual scatter plot
- Residuals plot
- Interval coverage plot
- QQ plot

**Probabilistic**:
- Calibration plot
- PIT histogram
- CRPS over time
- Reliability diagram

**Conformal**:
- Coverage evolution
- Interval width distribution
- Predictions with intervals

### Output Configuration

Control output behavior with `output_config`:

```python
output_config = {
    # Directories
    "output_dir": "./outputs/my_run",

    # Excel exports
    "export_excel": True,
    "embed_excel_charts": True,  # Embed charts in Excel

    # Plot exports
    "export_plots": True,
    "show_inline_plots": False,  # Display in notebook

    # Chart styling
    "chart_detail_level": "detailed",  # or "summary"
    "legend_position": "upper right",
    "style_overrides": {
        "figure.figsize": [10, 6],
        "font.size": 12
    },

    # JSON exports
    "save_json": True
}
```

---

## Advanced Topics

### Custom Data Splitting

Control how data is split:

```python
data = {
    "data_path": "./data/timeseries.csv",
    "split": {
        "test_size": 0.2,
        "shuffle": False,  # Important for time series!
        "random_state": 42
    }
}
```

### Multiple Quantiles

Predict multiple quantiles:

```python
# Note: Currently supports fixed quantiles [0.05, 0.95]
# For custom quantiles, modify the quantile config
```

### Model Parameters

Provide custom parameters:

```python
params_source = {
    "mode": "explicit",
    "params": {
        "lightgbm": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1
        },
        "xgboost": {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.05
        }
    }
}
```

### Saving and Loading Results

Results are automatically saved when `save_json=True`. Load them:

```python
import json

with open("outputs/my_run/lightgbm/metrics.json") as f:
    metrics = json.load(f)
```

### Accessing Raw Predictions

```python
result = run_quantile(...)

# Access predictions DataFrame
predictions_df = result.predictions["lightgbm"]

# Contains columns: y_true, y_pred, lower_bound, upper_bound
print(predictions_df.head())
```

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` in Colab after installation

**Solution**: Make sure to run `bootstrap_colab_environment()` and wait for the runtime restart.

---

**Issue**: Tests failing with import errors

**Solution**: Install in editable mode:
```bash
pip install -e .
```

---

**Issue**: Charts not showing in Excel

**Solution**: Make sure `embed_excel_charts=True` in `output_config` and you have `openpyxl` installed.

---

**Issue**: Memory errors with large datasets

**Solution**:
- Reduce `n_trials` in tuning
- Use smaller models
- Process data in batches

---

## Best Practices

1. **Start with defaults**: Use `params_source={"mode": "defaults"}` for quick experiments
2. **Tune when needed**: Use tuning phase for production models
3. **Use Colab bootstrap**: Always use `bootstrap_colab_environment()` in Colab
4. **Preserve temporal order**: Set `shuffle=False` for time series data
5. **Monitor coverage**: Check coverage metrics to ensure reliable intervals
6. **Export everything**: Enable all exports during development for inspection

---

## Getting Help

- **GitHub Issues**: https://github.com/DaneshSelwal/reguq/issues
- **Examples**: Check `examples/` directory
- **Documentation**: This guide and `README.md`

---

## Citation

If you use RegUQ in your research, please cite:

```
RegUQ: Regression Uncertainty Quantification Toolkit
Authors: Prakriti Bisht, Danesh Selwal
Under guidance of: Dr. Mahesh Pal
GitHub: https://github.com/DaneshSelwal/reguq
```

---

**Last Updated**: March 2026
**Version**: 0.1.1
