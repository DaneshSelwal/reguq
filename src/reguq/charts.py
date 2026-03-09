"""Chart generation utilities for reports and notebook visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .export import safe_sheet_name
from .types import ConformalResult, OutputConfig, PhaseResult


@dataclass
class ChartRenderResult:
    image_paths: list[Path] = field(default_factory=list)
    images_by_sheet: dict[str, list[Path]] = field(default_factory=dict)


def _style_value(output_cfg: OutputConfig, key: str, default):
    if output_cfg.style_overrides and key in output_cfg.style_overrides:
        return output_cfg.style_overrides[key]
    return default


def _legend_loc(output_cfg: OutputConfig) -> str:
    value = str(output_cfg.legend_position or "upper right").strip().lower().replace("_", " ")
    valid = {
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    }
    return value if value in valid else "upper right"


def _display_inline(fig, enabled: bool) -> None:
    if not enabled:
        return
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        # If IPython is unavailable, silently skip inline rendering.
        pass


def _finalize_figure(
    fig,
    output_path: Path | None,
    show_inline: bool,
) -> Path | None:
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    _display_inline(fig, enabled=show_inline)
    plt.close(fig)
    return output_path


def _plot_trajectory(pred_df: pd.DataFrame, title: str, output_cfg: OutputConfig):
    if pred_df.empty:
        return None
    max_points = int(_style_value(output_cfg, "max_plot_points", 300))
    subset = pred_df.head(max_points).copy()
    x = np.arange(len(subset))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        x,
        subset["y_true"],
        label="True",
        linewidth=float(_style_value(output_cfg, "line_width_true", 1.7)),
        color=str(_style_value(output_cfg, "color_true", "#1f77b4")),
    )
    if "y_pred" in subset:
        ax.plot(
            x,
            subset["y_pred"],
            label="Prediction",
            linewidth=float(_style_value(output_cfg, "line_width_pred", 1.4)),
            color=str(_style_value(output_cfg, "color_pred", "#ff7f0e")),
        )

    if "y_lower" in subset and "y_upper" in subset:
        ax.fill_between(
            x,
            subset["y_lower"],
            subset["y_upper"],
            alpha=float(_style_value(output_cfg, "interval_alpha", 0.22)),
            label="Interval",
            color=str(_style_value(output_cfg, "color_interval", "#2ca02c")),
        )

    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Target")
    ax.legend(loc=_legend_loc(output_cfg))
    ax.grid(alpha=0.25)
    return fig


def _plot_residual_histogram(pred_df: pd.DataFrame, title: str, output_cfg: OutputConfig):
    if "y_pred" not in pred_df:
        return None
    if pred_df.empty:
        return None

    residual = pred_df["y_true"].to_numpy() - pred_df["y_pred"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = int(_style_value(output_cfg, "residual_bins", 30))
    ax.hist(
        residual,
        bins=bins,
        alpha=0.85,
        edgecolor="white",
        color=str(_style_value(output_cfg, "color_residual", "#9467bd")),
        label="Residual",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="Zero Error")
    ax.set_title(title)
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Count")
    ax.legend(loc=_legend_loc(output_cfg))
    ax.grid(alpha=0.2)
    return fig


def _plot_interval_diagnostics(pred_df: pd.DataFrame, title: str, output_cfg: OutputConfig):
    if not {"y_lower", "y_upper"}.issubset(pred_df.columns):
        return None
    if pred_df.empty:
        return None

    y_true = pred_df["y_true"].to_numpy()
    y_lower = pred_df["y_lower"].to_numpy()
    y_upper = pred_df["y_upper"].to_numpy()
    width = y_upper - y_lower
    covered = ((y_true >= y_lower) & (y_true <= y_upper)).astype(int)
    rolling = pd.Series(covered).rolling(window=min(30, len(covered)), min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(
        width,
        covered,
        alpha=0.45,
        s=20,
        color=str(_style_value(output_cfg, "color_width_coverage", "#17becf")),
        label="Sample",
    )
    axes[0].set_title(f"{title} - Width vs Coverage")
    axes[0].set_xlabel("Interval Width")
    axes[0].set_ylabel("Covered (0/1)")
    axes[0].legend(loc=_legend_loc(output_cfg))
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        np.arange(len(rolling)),
        rolling.to_numpy(),
        color=str(_style_value(output_cfg, "color_rolling", "#d62728")),
        linewidth=1.5,
        label="Rolling Coverage",
    )
    axes[1].set_title(f"{title} - Rolling Coverage")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc=_legend_loc(output_cfg))
    axes[1].grid(alpha=0.2)

    return fig


def _plot_uncertainty_diagnostics(pred_df: pd.DataFrame, title: str, output_cfg: OutputConfig):
    if not {"y_std", "y_pred"}.issubset(pred_df.columns):
        return None
    if pred_df.empty:
        return None

    sigma = pred_df["y_std"].to_numpy()
    abs_err = np.abs(pred_df["y_true"].to_numpy() - pred_df["y_pred"].to_numpy())

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.scatter(
        sigma,
        abs_err,
        alpha=0.5,
        s=24,
        color=str(_style_value(output_cfg, "color_sigma_error", "#8c564b")),
        label="Sample",
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Std Dev")
    ax.set_ylabel("Absolute Error")
    ax.legend(loc=_legend_loc(output_cfg))
    ax.grid(alpha=0.2)
    return fig


def _plot_phase_summary(metrics_df: pd.DataFrame, title: str, output_cfg: OutputConfig):
    if metrics_df.empty or "model" not in metrics_df.columns:
        return None

    preferred = [
        "rmse",
        "mae",
        "r2",
        "coverage",
        "avg_interval_width",
        "nll",
        "crps",
        "cv_score",
    ]
    metric_cols = [c for c in preferred if c in metrics_df.columns]
    if not metric_cols:
        return None

    if output_cfg.chart_detail_level == "detailed":
        metric_cols = metric_cols[:5]
    else:
        metric_cols = metric_cols[:3]

    summary_df = metrics_df[["model", *metric_cols]].copy().set_index("model")
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(summary_df)), 4.5))
    summary_df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric Value")
    ax.legend(loc=_legend_loc(output_cfg))
    ax.grid(alpha=0.25)
    return fig


def _model_sheet_name(model_id: str, method_name: str | None) -> str:
    if method_name:
        return safe_sheet_name(f"{method_name}_{model_id}")
    return safe_sheet_name(f"pred_{model_id}")


def _metrics_sheet_name(method_name: str | None) -> str:
    if method_name:
        return safe_sheet_name(f"m_{method_name}")
    return "metrics"


def _merge_chart_results(base: ChartRenderResult, other: ChartRenderResult) -> ChartRenderResult:
    base.image_paths.extend(other.image_paths)
    for sheet_name, paths in other.images_by_sheet.items():
        base.images_by_sheet.setdefault(sheet_name, []).extend(paths)
    return base


def generate_phase_charts(
    phase_result: PhaseResult,
    phase_name: str,
    output_cfg: OutputConfig,
    output_dir: Path | None,
    method_name: str | None = None,
    metrics_sheet_name: str | None = None,
) -> ChartRenderResult:
    chart_result = ChartRenderResult()

    persist_images = output_dir is not None and (output_cfg.export_plots or output_cfg.embed_excel_charts)
    charts_root = None
    if persist_images and output_dir is not None:
        charts_root = output_dir / "charts" / safe_sheet_name(phase_name)

    for model_id, pred_df in phase_result.predictions.items():
        model_sheet = _model_sheet_name(model_id=model_id, method_name=method_name)
        model_prefix = f"{safe_sheet_name(phase_name)}_{model_id}"

        figure_specs = [
            ("trajectory", _plot_trajectory(pred_df, f"{phase_name}: {model_id} Prediction", output_cfg)),
            ("residual", _plot_residual_histogram(pred_df, f"{phase_name}: {model_id} Residual Distribution", output_cfg)),
            ("interval_diag", _plot_interval_diagnostics(pred_df, f"{phase_name}: {model_id}", output_cfg)),
            ("uncertainty", _plot_uncertainty_diagnostics(pred_df, f"{phase_name}: {model_id} Uncertainty vs Error", output_cfg)),
        ]

        for chart_key, fig in figure_specs:
            if fig is None:
                continue
            output_path = None
            if charts_root is not None:
                output_path = charts_root / model_id / f"{model_prefix}_{chart_key}.png"
            saved_path = _finalize_figure(fig=fig, output_path=output_path, show_inline=output_cfg.show_inline_plots)
            if saved_path is not None:
                chart_result.image_paths.append(saved_path)
                chart_result.images_by_sheet.setdefault(model_sheet, []).append(saved_path)

    summary_fig = _plot_phase_summary(
        metrics_df=phase_result.metrics,
        title=f"{phase_name}: Summary Metrics",
        output_cfg=output_cfg,
    )
    if summary_fig is not None:
        summary_path = None
        if charts_root is not None:
            summary_path = charts_root / f"{safe_sheet_name(phase_name)}_summary_metrics.png"
        saved_summary = _finalize_figure(
            fig=summary_fig,
            output_path=summary_path,
            show_inline=output_cfg.show_inline_plots,
        )
        if saved_summary is not None:
            metrics_sheet = (
                safe_sheet_name(metrics_sheet_name)
                if metrics_sheet_name is not None
                else _metrics_sheet_name(method_name=method_name)
            )
            chart_result.image_paths.append(saved_summary)
            chart_result.images_by_sheet.setdefault(metrics_sheet, []).append(saved_summary)

    return chart_result


def generate_conformal_charts(
    conformal_result: ConformalResult,
    output_cfg: OutputConfig,
    output_dir: Path | None,
) -> ChartRenderResult:
    aggregate = ChartRenderResult()
    for method_name, method_result in conformal_result.methods.items():
        method_phase_name = f"{conformal_result.phase}_{method_name}"
        method_charts = generate_phase_charts(
            phase_result=method_result,
            phase_name=method_phase_name,
            output_cfg=output_cfg,
            output_dir=output_dir,
            method_name=method_name,
        )
        _merge_chart_results(aggregate, method_charts)
    return aggregate
