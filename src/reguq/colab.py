"""Colab environment bootstrap helpers."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


PINNED_PACKAGES = [
    "numpy==1.26.4",
    "pandas==2.2.3",
    "scipy==1.14.1",
    "scikit-learn==1.6.1",
    "lightgbm==4.6.0",
    "xgboost==3.1.2",
    "catboost==1.2.8",
    "ngboost==0.5.8",
    "pgbm==2.2.0",
    "mapie==0.6.0",
    "puncc==0.8.0",
    "optuna==4.6.0",
    "properscoring==0.1",
    "openpyxl==3.1.5",
    "XlsxWriter==3.2.9",
    "PyYAML==6.0.2",
]


def is_colab_runtime() -> bool:
    return "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


def bootstrap_colab_environment(
    repo_url: str = "https://github.com/DaneshSelwal/reguq.git",
    marker_path: str = "/content/.reguq_env_ready_v2",
    quiet: bool = True,
) -> bool:
    """Install pinned dependencies in Colab and restart runtime once.

    Returns True when installation happened and restart was triggered, False when
    environment was already prepared.
    """

    marker = Path(marker_path)
    if marker.exists():
        return False

    pip_flags = ["-q"] if quiet else []

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            *pip_flags,
            "numpy==1.26.4",
            "pandas==2.2.3",
            "scipy==1.14.1",
            "scikit-learn==1.6.1",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *pip_flags,
            *PINNED_PACKAGES[4:],
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *pip_flags,
            "--no-deps",
            f"git+{repo_url}",
        ],
        check=True,
    )

    marker.write_text("ok", encoding="utf-8")

    if is_colab_runtime():
        os.kill(os.getpid(), signal.SIGKILL)

    return True
