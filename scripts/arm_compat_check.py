"""ARM64 / x86_64 compatibility smoke check.

Run this on the target VM (Oracle Cloud Free Tier ARM Ampere) immediately
after `pip install -r requirements.txt` to verify every Week-1 dependency
imports cleanly. Fails fast with a clear list of missing/broken packages.
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Iterable

# Packages we must import successfully for Week 1 to function.
WEEK1_PACKAGES: tuple[str, ...] = (
    "dotenv",
    "yaml",
    "pydantic",
    "pydantic_settings",
    "loguru",
    "requests",
    "urllib3",
    "tenacity",
    "defusedxml",
    "pandas",
    "numpy",
    "pyarrow",
    "yfinance",
    "curl_cffi",
    "nsepython",
    "tqdm",
    "rich",
    "sklearn",
    "xgboost",
    "joblib",
    "optuna",
    "pytest",
    "freezegun",
)


def check(packages: Iterable[str] = WEEK1_PACKAGES) -> int:
    print("=" * 60)
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"Machine  : {platform.machine()}")
    print("=" * 60)
    failed: list[tuple[str, str]] = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"  OK  : {pkg}")
        except Exception as e:  # noqa: BLE001
            failed.append((pkg, str(e)))
            print(f"  FAIL: {pkg}: {e}")
    print("=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}/{len(list(packages))}):")
        for pkg, err in failed:
            print(f"  - {pkg}: {err}")
        return 2
    print(f"OK: all {len(list(packages))} Week-1 packages imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
