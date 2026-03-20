# RegUQ Future Work: Advanced Methods

This document outlines the planned implementation of advanced uncertainty quantification methods that are present in the Data_folder notebooks but not yet integrated into the main package.

## Overview

The RegUQ package currently implements core uncertainty quantification methods (v0.1.1):
- ✅ Hyperparameter Tuning (Optuna)
- ✅ Quantile Regression
- ✅ Probabilistic Distribution (NGBoost, PGBM)
- ✅ Standard Conformal Prediction (MAPIE, PUNCC)

The following advanced methods are planned for future releases:

## 1. Advanced Conformal Prediction Methods

### Status: Planned for v0.2.0

### Methods to Implement

#### A. NEXCP (Non-Exchangeable Conformal Prediction)

**Location in Data_folder**: `Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)/`

**Purpose**: Handle distribution shift and temporal dependencies where standard conformal methods fail.

**Key Features**:
- Time-weighted nonconformity scores
- Adaptive to distribution drift
- Maintains coverage under non-exchangeability

**Implementation Plan**:
1. Create `src/reguq/conformal_advanced.py` module
2. Implement NEXCP algorithm:
   - Exponentially weighted nonconformity scores
   - Time-aware calibration
   - Rolling window updates
3. Add tests in `tests/test_conformal_advanced.py`
4. Update documentation and examples

**API Design**:
```python
from reguq import run_conformal_advanced

result = run_conformal_advanced(
    data={"train": train_df, "test": test_df},
    target_col="target",
    models=["lightgbm"],
    conformal_config={
        "method": "nexcp",
        "alpha": 0.1,
        "decay": 0.95,  # Weight decay for temporal weighting
        "window_size": 100
    },
    output_config={...}
)
```

**References**:
- Barber et al. (2022). "Conformal Prediction Beyond Exchangeability"
- Gibbs & Candès (2021). "Adaptive Conformal Inference Under Distribution Shift"

---

#### B. Adaptive Conformal Prediction

**Location in Data_folder**: `Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)/`

**Purpose**: Dynamically adjust prediction intervals based on recent coverage performance.

**Key Features**:
- Online learning of interval width
- Tracks coverage errors over time
- Automatically tightens/widens intervals

**Implementation Plan**:
1. Add to `src/reguq/conformal_advanced.py`
2. Implement adaptive algorithm:
   - Coverage error tracking
   - Dynamic interval adjustment
   - Learning rate scheduling
3. Add visualization for coverage evolution

**API Design**:
```python
result = run_conformal_advanced(
    data={"train": train_df, "test": test_df},
    target_col="target",
    models=["xgboost"],
    conformal_config={
        "method": "adaptive",
        "alpha": 0.1,
        "learning_rate": 0.01,
        "adaptation_window": 50
    },
    output_config={...}
)
```

**References**:
- Gibbs & Candès (2021). "Adaptive Conformal Inference"
- Angelopoulos et al. (2021). "Uncertainty Sets for Image Classifiers using Conformal Prediction"

---

#### C. mfcs (Model-Free Conformal Selection)

**Location in Data_folder**: `Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)/`

**Purpose**: Select optimal base model using conformal prediction scores.

**Key Features**:
- Model selection without separate validation set
- Conformal score-based model comparison
- Ensemble construction

**Implementation Plan**:
1. Add model selection utilities
2. Implement conformal ensemble methods
3. Add model comparison metrics

---

### Implementation Challenges

1. **Temporal Dependencies**:
   - Need to handle time series data correctly
   - Ensure proper ordering in calibration
   - Implement sliding window mechanisms

2. **Computational Efficiency**:
   - NEXCP requires recomputing weights
   - Adaptive methods need online updates
   - May need caching strategies

3. **Coverage Tracking**:
   - Real-time coverage monitoring
   - Alert systems for coverage violations
   - Visualization dashboards

---

## 2. CARD (Classification And Regression Diffusion)

### Status: Planned for v0.3.0

**Location in Data_folder**: `Probabilistic_Distribution(CARD)/`

**Purpose**: Use diffusion models to generate complex conditional distributions for regression.

### Overview

CARD uses diffusion processes (similar to image generation models like DALL-E) to model the conditional distribution p(y|x).

**Key Advantages**:
- Can model multi-modal distributions
- Captures complex dependencies
- State-of-the-art probabilistic predictions

### Implementation Plan

#### Phase 1: Research & Prototyping
1. Study CARD paper and existing implementations
2. Review Data_folder notebook implementation
3. Identify dependencies (likely requires PyTorch)
4. Design API consistent with existing probabilistic methods

#### Phase 2: Core Implementation
1. Create `src/reguq/card.py` module
2. Implement diffusion process:
   - Forward diffusion (adding noise)
   - Reverse diffusion (denoising)
   - Conditional generation given features
3. Training utilities:
   - Loss functions
   - Sampling strategies
   - Hyperparameter optimization

#### Phase 3: Integration
1. Add to model registry
2. Integrate with `run_probabilistic()` API
3. Add CARD-specific configuration options
4. Implement efficient inference

#### Phase 4: Validation
1. Extensive testing on various datasets
2. Comparison with NGBoost and PGBM
3. Computational benchmarking
4. Documentation and examples

### API Design

```python
from reguq import run_probabilistic

result = run_probabilistic(
    data={"train": train_df, "test": test_df},
    target_col="target",
    models=["card"],  # New model type
    params_source={
        "mode": "explicit",
        "params": {
            "card": {
                "n_diffusion_steps": 1000,
                "beta_schedule": "linear",
                "hidden_dim": 128,
                "n_layers": 3,
                "training_epochs": 100
            }
        }
    },
    output_config={...}
)
```

### Dependencies

CARD will likely require:
- PyTorch (not currently a dependency)
- Additional computational resources (GPU recommended)
- Longer training times compared to tree-based models

### Implementation Challenges

1. **Computational Cost**:
   - Diffusion models are slower than tree methods
   - May need GPU support
   - Inference requires multiple denoising steps

2. **Hyperparameter Sensitivity**:
   - Many hyperparameters to tune
   - Requires careful initialization
   - Training stability issues

3. **Integration Complexity**:
   - Different paradigm from tree-based models
   - Need to maintain consistent API
   - Output format compatibility

4. **Package Size**:
   - PyTorch is a large dependency
   - May need optional installation
   - Consider making CARD an optional feature

### References

- Han et al. (2022). "CARD: Classification and Regression Diffusion Models"
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
- Song et al. (2021). "Score-Based Generative Modeling"

---

## 3. Additional Planned Features

### A. Model Ensembles (v0.2.x)

**Purpose**: Combine predictions from multiple models for better performance.

**Features**:
- Weighted ensemble predictions
- Stacking and blending
- Uncertainty aggregation

**API Design**:
```python
result = run_ensemble(
    data={...},
    target_col="target",
    base_models=["lightgbm", "xgboost", "catboost"],
    ensemble_method="weighted",  # or "stacking", "blending"
    weights=[0.4, 0.4, 0.2],  # Optional manual weights
    output_config={...}
)
```

---

### B. Feature Engineering (v0.2.x)

**Purpose**: Automated feature transformation and selection.

**Features**:
- Polynomial features
- Interaction terms
- Automated feature selection
- Target encoding

**API Design**:
```python
result = run_quantile(
    data={...},
    target_col="target",
    models=["lightgbm"],
    feature_config={
        "polynomial_degree": 2,
        "interaction_depth": 2,
        "selection_method": "mutual_info",
        "max_features": 20
    },
    output_config={...}
)
```

---

### C. Monitoring & Alerting (v0.3.x)

**Purpose**: Track model performance and uncertainty over time.

**Features**:
- Coverage monitoring dashboard
- Distribution shift detection
- Automated alerts for coverage violations
- Performance degradation detection

**API Design**:
```python
from reguq import CoverageMonitor

monitor = CoverageMonitor(
    target_coverage=0.9,
    window_size=100,
    alert_threshold=0.05
)

# In production loop
for batch in data_stream:
    preds = model.predict(batch)
    monitor.update(y_true=batch.y, predictions=preds)

    if monitor.coverage_violated():
        print("Alert: Coverage below threshold!")
        monitor.send_notification()
```

---

### D. Calibration Methods (v0.2.x)

**Purpose**: Post-hoc calibration of prediction intervals.

**Features**:
- Isotonic regression calibration
- Platt scaling for intervals
- Temperature scaling
- Conformal calibration

---

## Implementation Timeline

### v0.2.0 (Q2 2026)
- ✅ Complete documentation (GUIDE.md, SKILL.md)
- 🔄 Advanced conformal methods (NEXCP, Adaptive CP)
- 🔄 Model ensembles
- 🔄 Feature engineering basics
- 🔄 Enhanced monitoring

### v0.3.0 (Q3 2026)
- 🔄 CARD implementation
- 🔄 Advanced calibration methods
- 🔄 Dashboard for monitoring
- 🔄 Performance optimizations

### v1.0.0 (Q4 2026)
- 🔄 Full feature parity with notebooks
- 🔄 Comprehensive benchmarks
- 🔄 Production-ready stability
- 🔄 PyPI release

---

## Contributing

If you'd like to contribute to implementing these features:

1. **Pick a Feature**: Choose from the list above
2. **Create an Issue**: Discuss implementation approach
3. **Fork & Develop**: Work on a feature branch
4. **Add Tests**: Maintain >90% coverage
5. **Update Docs**: Add examples and API docs
6. **Submit PR**: Include benchmarks and examples

### Development Guidelines

- Follow existing code style (see `SKILL.md`)
- Add comprehensive tests
- Update documentation
- Maintain API consistency
- Include examples in `examples/`

---

## Questions & Discussion

For questions or suggestions about future features:

- **GitHub Discussions**: https://github.com/DaneshSelwal/reguq/discussions
- **Issues**: https://github.com/DaneshSelwal/reguq/issues
- **Email**: Contact project maintainers

---

## References

### Academic Papers

1. Barber et al. (2022). "Conformal Prediction Beyond Exchangeability"
2. Gibbs & Candès (2021). "Adaptive Conformal Inference Under Distribution Shift"
3. Han et al. (2022). "CARD: Classification and Regression Diffusion Models"
4. Romano et al. (2019). "Conformalized Quantile Regression"
5. Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"

### Existing Implementations

- **MAPIE**: https://github.com/scikit-learn-contrib/MAPIE
- **PUNCC**: https://github.com/deel-ai/puncc
- **CARD**: https://github.com/XzwHan/CARD
- **Diffusion Models**: https://github.com/huggingface/diffusers

---

**Last Updated**: March 2026
**Document Version**: 1.0
**Maintainers**: Prakriti Bisht, Danesh Selwal
