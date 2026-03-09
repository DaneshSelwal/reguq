"""Minimal reguq usage example."""

from reguq import run_quantile

result = run_quantile(
    data={
        "train_path": "./Data_folder/Data/train.csv",
        "test_path": "./Data_folder/Data/test.csv",
    },
    target_col="target",
    models=["lightgbm", "xgboost"],
    params_source={"mode": "defaults"},
    output_config={
        "output_dir": "./outputs/quickstart",
        "export_excel": True,
        "export_plots": True,
        "embed_excel_charts": True,
        "show_inline_plots": False,
    },
)

print(result.metrics)
