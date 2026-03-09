# 🚀 Comprehensive Regression Analysis Pipeline: Advanced Uncertainty Quantification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange?style=for-the-badge)
![Uncertainty Quantification](https://img.shields.io/badge/Uncertainty-Adaptive%20CP%20%7C%20NEXCP%20%7C%20Quantile-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

Welcome to the **End-to-End Regression Analysis Pipeline**. This repository is engineered as a modular, "plug-and-play" framework for robust regression tasks. It goes beyond simple point predictions by integrating a suite of **Uncertainty Quantification (UQ)** methods, ensuring that every prediction is accompanied by a reliable confidence interval.

Whether you are analyzing environmental data, financial time-series, or industrial sensor readings, this pipeline allows you to swap in your dataset and immediately leverage state-of-the-art Hyperparameter Tuning, Quantile Regression, Probabilistic Modeling, and Adaptive Conformal Prediction.

---

## 📑 Table of Contents (Navigation)

1. [📌 Project Overview](#-project-overview)
2. [📂 Repository Structure](#-repository-structure)
3. [📊 Dataset & Usage](#-dataset--usage)
4. [🛠️ Workflow & Methodology](#-workflow--methodology)
    - [Phase 1: Hyperparameter Tuning](#phase-1-hyperparameter-tuning)
    - [Phase 2: Quantile Regression](#phase-2-quantile-regression)
    - [Phase 3: Probabilistic Distribution](#phase-3-probabilistic-distribution)
    - [Phase 3b: Probabilistic Distribution (CARD)](#phase-3b-probabilistic-distribution-card)
    - [Phase 4: Standard Conformal Predictions](#phase-4-standard-conformal-predictions)
    - [Phase 5: Adaptive & Non-Exchangeable CP](#phase-5-adaptive--non-exchangeable-cp)
5. [🚀 Getting Started](#-getting-started)

---

## 📌 Project Overview

This framework provides a rigorous path from raw data to confident predictions. It is designed to be **domain-agnostic**: while the inspiration comes from hydrological sediment load analysis, the methods are applicable to any regression problem, especially those involving time-series or non-exchangeable data.

**Key Features:**
*   **Automated Optimization**: Harnessing **Optuna** for Bayesian optimization of complex regressors.
*   **Interval Estimation**: **Quantile Regression** for estimating conditional bounds (e.g., 5th and 95th percentiles).
*   **Full Distribution Modeling**: Using **NGBoost** and **PGBM** to predict the full probability distribution parameters ($\mu, \sigma$).
*   **Generative Modeling**: Leveraging **CARD (Classification and Regression Diffusion)** models to generate conditional distributions using diffusion processes.
*   **Robust Uncertainty**: Implementation of **NEXCP (Non-Exchangeable Conformal Prediction)** and **Adaptive CP**, crucial for handling data drift and temporal dependencies where standard methods fail.

---

## 📂 Repository Structure

The project is encapsulated within the `Data_folder`, organized by analysis phase.

```
.
├── Data_folder/
│   ├── Data/                                       # 📍 Input Data (Entry Point)
│   │   ├── train.csv                               # Training dataset
│   │   └── test.csv                                # Testing dataset
│   │
│   ├── HyperParameter_Tuning/                      # 🎛️ Phase 1: Optimization
│   │   ├── Optuna_autosampler.ipynb                # Optuna Bayesian Optimization script
│   │   └── models/                                 # Saved optimized models
│   │
│   ├── Quantile_Regression/                        # 📉 Phase 2: Quantile Methods
│   │   ├── Quantile_Regression.ipynb               # Script for Quantile Regression
│   │   └── Results/                                # Prediction outputs
│   │
│   ├── Probabilistic_Distribution/                 # 📊 Phase 3: Distributional Models
│   │   ├── Probabilistic__Distribution.ipynb       # NGBoost & PGBM implementation
│   │   └── Results/                                # Calibration plots & CRPS scores
│   │
│   ├── Probabilistic_Distribution(CARD)/           # 🌫️ Phase 3b: Diffusion Models (CARD)
│   │   └── Probabilistic__Distribution(CARD).ipynb # Diffusion-based distribution modeling
│   │
│   ├── Conformal_Predictions(MAPIE,PUNCC)/         # 🛡️ Phase 4: Standard CP
│   │   └── Conformal Predictions(MAPIE,PUNCC).ipynb
│   │
│   └── Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)/ # 🛡️ Phase 5: Advanced Time-Series CP
│       └── Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb
│
└── README.md
```

---

## 📊 Dataset & Usage

**This is a Template Pipeline.**

To use this repository with your own data:

1.  **Prepare your data**: You need a training set and a testing set.
2.  **Format**: Ensure your files are in `.csv` format.
3.  **Replace**:
    *   Replace `Data_folder/Data/train.csv` with your training data.
    *   Replace `Data_folder/Data/test.csv` with your testing data.
4.  **Configure**:
    *   **Column Names**: Open the notebooks (e.g., `Optuna_autosampler.ipynb`) and ensure the column names match your dataset's target variable and features.
    *   **File Paths**: Some notebooks may contain hardcoded paths (e.g., `/content/drive/MyDrive/...`) from the original Google Colab environment. You must update these paths to point to your local `Data_folder` location.

The default configuration assumes a structure with predictor columns and a target column. Adjust the "Target" variable name in the scripts to match your specific regression problem.

---

## 🛠️ Workflow & Methodology

### Phase 1: Hyperparameter Tuning
**Location**: `Data_folder/HyperParameter_Tuning`
Before any uncertainty quantification, we must ensure our base estimators are accurate.
*   **Tool**: **Optuna**.
*   **Process**: We search over hyperparameter spaces for XGBoost, CatBoost, LightGBM, etc., using efficient pruners (Hyperband) to find the best configuration.
*   **Output**: Optimized model parameters saved for subsequent steps.

### Phase 2: Quantile Regression
**Location**: `Data_folder/Quantile_Regression`
We move beyond the mean.
*   **Goal**: Predict conditional quantiles (e.g., $Q_{0.05}$ and $Q_{0.95}$) to bracket the target value.
*   **Loss Function**: Pinball Loss.
*   **Result**: A prediction interval that captures a specified percentage of the data (e.g., 90%).

### Phase 3: Probabilistic Distribution
**Location**: `Data_folder/Probabilistic_Distribution`
Treating the target as a random variable $Y|X \sim \mathcal{D}(\theta)$.
*   **Models**: **NGBoost** (Natural Gradient Boosting) and **PGBM** (Probabilistic Gradient Boosting Machines).
*   **Metrics**: Negative Log-Likelihood (NLL) and Continuous Ranked Probability Score (CRPS).
*   **Visualization**: Probability Integral Transform (PIT) histograms to verify calibration.

### Phase 3b: Probabilistic Distribution (CARD)
**Location**: `Data_folder/Probabilistic_Distribution(CARD)`
Using generative diffusion models to capture complex conditional distributions.
*   **Models**: **CARD** (Classification and Regression Diffusion).
*   **Method**: Converts the regression target into a noise distribution and learns to reverse the diffusion process conditioned on features.
*   **Advantage**: Capable of modeling multi-modal distributions and complex dependencies.

### Phase 4: Standard Conformal Predictions
**Location**: `Data_folder/Conformal_Predictions(MAPIE,PUNCC)`
For data that satisfies the **exchangeability** assumption (i.e., order doesn't matter).
*   **Libraries**: `MAPIE`, `PUNCC`.
*   **Methods**: Split Conformal, CV+, Jackknife+.
*   **Guarantee**: Provides marginal coverage guarantees with finite-sample validity.

### Phase 5: Adaptive & Non-Exchangeable CP
**Location**: `Data_folder/Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)`
**Crucial for Time-Series**.
Real-world data often drifts or has temporal dependencies.
*   **NEXCP**: Non-Exchangeable Conformal Prediction. Weights recent observations more heavily to adapt to distribution shifts.
*   **Adaptive CP**: Dynamically updates the interval width $C_t$ based on recent coverage errors.
*   **Result**: Valid coverage even during volatile periods (e.g., market crashes, floods).

---

## 🚀 Getting Started

1.  **Clone the Repository**:
    ```bash
    git clone <repo_url>
    cd <repo_directory>
    ```

2.  **Install Dependencies**:
    Ensure you have Python 3.10+ and the required libraries:
    ```bash
    pip install optuna xgboost lightgbm catboost ngboost pgbm mapie puncc
    ```
    *(Note: Check individual notebooks for specific library versions)*

3.  **Run the Pipeline**:
    Execute the notebooks in the order presented in the **Repository Structure** (Hyperparameter Tuning $\rightarrow$ Quantile/Probabilistic $\rightarrow$ Conformal Predictions).


     <sup>**</sup>This repository is a collaborative project developed under guidance of Dr. Mahesh Pal by Prakriti Bisht and Danesh Selwal.
---

---

## 📦 `reguq` Python Package (v1)

This repository now includes an installable package under `src/reguq` that replaces the multi-notebook flow with task-based APIs and a config runner.

### Install (GitHub, current)

```bash
pip install "git+https://github.com/DaneshSelwal/reguq.git"
```

### API Quickstart

```python
from reguq import run_quantile

result = run_quantile(
    data={
        "train_path": "./Data_folder/Data/train.csv",
        "test_path": "./Data_folder/Data/test.csv",
    },
    target_col="target",
    models=["lightgbm", "xgboost", "catboost"],
    params_source={"mode": "load_or_tune", "params": {}},
    output_config={
        "output_dir": "./outputs/run_01",
        "export_excel": True,
        "export_plots": True,
        "embed_excel_charts": True,
        "show_inline_plots": False,
        "chart_detail_level": "detailed",
        "legend_position": "upper right",
    },
)

print(result.metrics)
```

### Config Runner

```bash
reguq-run --config examples/pipeline_config.yaml
```

The runner supports YAML config files and executes selected phases:
- `tuning`
- `quantile`
- `probabilistic`
- `conformal_standard`

Reporting settings can be passed under `output` (preferred) or `report` (alias).

### Public APIs

- `run_tuning(data, target_col, models, tuning_config, output_config)`
- `run_quantile(data, target_col, models, params_source, output_config)`
- `run_probabilistic(data, target_col, models, params_source, output_config)`
- `run_conformal_standard(data, target_col, models, params_source, conformal_config, output_config)`
- `run_from_config(config_or_path)`
- `bootstrap_colab_environment(repo_url, marker_path, quiet)`

### Data Input Contract

- Train/test mode: provide `train_path` + `test_path` (or DataFrames).
- Single-data mode: provide `data_path` (or DataFrame), package auto-splits with temporal-safe default (`shuffle=False`).
- `target_col` is required; feature columns are auto-inferred as all non-target columns.

### Output Contract

- APIs always return structured Python results (DataFrames + metadata).
- Excel/plot artifacts are optional and controlled via `output_config`.
- `output_config` supports rich reporting flags:
  - `export_plots`
  - `embed_excel_charts`
  - `show_inline_plots`
  - `chart_detail_level` (`detailed` recommended)
  - `legend_position` (for chart legend placement)
- When output paths are used through runner defaults, artifacts are written under `outputs/<run_id>/`.

### Colab Notes

This package is designed for Colab-first reproducibility with pinned dependencies in `pyproject.toml`. For notebook usage:

```bash
!pip uninstall -y reguq
!pip install --upgrade --force-reinstall --no-cache-dir "git+https://github.com/DaneshSelwal/reguq.git@main"
```

Then run one-time bootstrap inside Colab:

```python
from reguq import bootstrap_colab_environment

# First run installs pinned deps and restarts runtime automatically.
bootstrap_colab_environment(repo_url="https://github.com/DaneshSelwal/reguq.git")
```

### PyPI Release (later)

The package is currently GitHub-installable. A PyPI workflow template is included under `.github/workflows/publish-pypi.yml` for later release publishing.

### Release Docs

- Changelog: `CHANGELOG.md`
- GitHub release checklist: `RELEASE.md`
