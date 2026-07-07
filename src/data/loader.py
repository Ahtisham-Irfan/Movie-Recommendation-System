"""
loader.py

Job of this file:
    Read the raw MovieLens CSVs (movies, ratings, tags, links) from disk,
    make sure nothing is missing, and hand back clean pandas DataFrames
    with sane dtypes so nothing breaks downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# filename each dataset should have inside data/raw
RAW_FILENAMES = {
    "movies": "movies.csv",
    "ratings": "ratings.csv",
    "tags": "tags.csv",
    "links": "links.csv",
}


class MissingDatasetError(FileNotFoundError):
    """Raised when one or more raw MovieLens files can't be found."""


def _build_paths(raw_dir: str | Path) -> Dict[str, Path]:
    raw_dir = Path(raw_dir).resolve()
    return {name: raw_dir / fname for name, fname in RAW_FILENAMES.items()}


def _check_files_exist(paths: Dict[str, Path]) -> None:
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        details = "\n".join(f"  - {name} -> expected at {paths[name]}" for name in missing)
        raise MissingDatasetError(
            "Can't find the following raw dataset file(s):\n"
            f"{details}\n"
            "Make sure movies.csv, ratings.csv, tags.csv and links.csv "
            "are all placed inside the data/raw folder."
        )
    logger.info("All 4 raw MovieLens files found.")


def _read_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"movieId": "int64", "title": "string", "genres": "string"})
    logger.info("movies.csv -> %d rows", len(df))
    return df


def _read_ratings(path: Path, with_datetime: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
    )
    if with_datetime:
        df["rated_at"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    logger.info("ratings.csv -> %d rows", len(df))
    return df


def _read_tags(path: Path, with_datetime: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"userId": "int64", "movieId": "int64", "tag": "string", "timestamp": "int64"},
    )
    if with_datetime:
        df["tagged_at"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    logger.info("tags.csv -> %d rows", len(df))
    return df


def _read_links(path: Path) -> pd.DataFrame:
    # imdbId / tmdbId can be missing for some movies, so use nullable Int64
    df = pd.read_csv(path, dtype={"movieId": "int64", "imdbId": "Int64", "tmdbId": "Int64"})
    logger.info("links.csv -> %d rows", len(df))
    return df


def load_all(raw_dir: str | Path = "data/raw", with_datetime: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load every raw MovieLens table and return them as a dict:
        {"movies": df, "ratings": df, "tags": df, "links": df}

    A plain dict is used on purpose (instead of a custom class) so the
    rest of the pipeline can just do dataset["movies"], dataset["ratings"]
    without importing a special type everywhere.
    """
    paths = _build_paths(raw_dir)
    _check_files_exist(paths)

    dataset = {
        "movies": _read_movies(paths["movies"]),
        "ratings": _read_ratings(paths["ratings"], with_datetime),
        "tags": _read_tags(paths["tags"], with_datetime),
        "links": _read_links(paths["links"]),
    }

    logger.info(
        "Dataset loaded -> movies=%d ratings=%d tags=%d links=%d",
        len(dataset["movies"]), len(dataset["ratings"]),
        len(dataset["tags"]), len(dataset["links"]),
    )
    return dataset


def quick_summary(dataset: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """Small dict of counts, handy for logs / notebooks / sanity checks."""
    return {
        "movies_count": int(len(dataset["movies"])),
        "ratings_count": int(len(dataset["ratings"])),
        "tags_count": int(len(dataset["tags"])),
        "links_count": int(len(dataset["links"])),
        "unique_users": int(dataset["ratings"]["userId"].nunique()),
        "unique_movies_rated": int(dataset["ratings"]["movieId"].nunique()),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ds = load_all()
    for k, v in quick_summary(ds).items():
        print(f"{k}: {v}")
