#!/usr/bin/env python3
"""
E2E Test Runner for reguq.
Executes the 4-tier opaque-box E2E test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

def main() -> int:
    try:
        import pytest
    except ImportError:
        print("Error: pytest is required to run the E2E test suite. Install with: pip install pytest")
        return 1

    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")
    print("Running E2E test suite (Tiers 1-4)...")

    # Command line arguments to pass to pytest
    args = [
        str(project_root / "tests" / "e2e"),
        "-v",
        "-ra",
    ]

    # Run pytest and exit with its code
    exit_code = pytest.main(args)
    return int(exit_code)

if __name__ == "__main__":
    sys.exit(main())
