"""
support.py

Small shared utilities used across the project: logging setup, safe
directory creation, JSON save/load, and a timing context manager.
Keeping these here avoids repeating boilerplate in every module.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional

import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    else:
        root.setLevel(level)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(file_path: str | Path) -> Path:
    resolved = Path(file_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(data: Any, file_path: str | Path, indent: int = 2) -> Path:
    out_path = ensure_parent_dir(file_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return out_path


def load_json(file_path: str | Path) -> Any:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def df_overview(df: pd.DataFrame, name: str = "dataframe") -> dict:
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def preview_rows(df: pd.DataFrame, columns: Optional[List[str]] = None, limit: int = 5) -> list:
    view = df[columns] if columns is not None else df
    return view.head(limit).to_dict(orient="records")


@contextmanager
def timed(label: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Usage: with timed('training step'): ..."""
    log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    log.info("Started: %s", label)
    try:
        yield
    finally:
        log.info("Finished: %s | took %.3fs", label, time.perf_counter() - start)
