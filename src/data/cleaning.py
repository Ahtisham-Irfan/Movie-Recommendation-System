"""
cleaning.py

Job of this file:
    Take the raw MovieLens tables and turn them into one clean,
    movie-level table with a single "metadata_text" column that
    the TF-IDF step can consume.

Steps done here:
    1. split "Toy Story (1995)" into title + year
    2. turn "Adventure|Comedy" into a clean token list
    3. aggregate all user tags per movie into one text blob
    4. compute rating_count / rating_mean / rating_median per movie
    5. merge everything and build metadata_text
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.data.loader import load_all

logger = logging.getLogger(__name__)

YEAR_IN_TITLE = re.compile(r"^(?P<name>.*)\s\((?P<yr>\d{4})\)$")


def normalize_text(value: object) -> str:
    """lowercase, strip punctuation, collapse whitespace -> safe empty string for NaN."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_title_year(raw_title: object) -> Tuple[str, "pd.NA | int"]:
    """'Heat (1995)' -> ('Heat', 1995). No year found -> (title, pd.NA)."""
    if raw_title is None or pd.isna(raw_title):
        return "", pd.NA

    text = str(raw_title).strip()
    m = YEAR_IN_TITLE.match(text)
    if m:
        return m.group("name").strip(), int(m.group("yr"))
    return text, pd.NA


def genres_to_tokens(raw_genres: object) -> List[str]:
    """'Adventure|Comedy' -> ['adventure', 'comedy']. Handles the '(no genres listed)' case."""
    if raw_genres is None or pd.isna(raw_genres):
        return []
    text = str(raw_genres).strip()
    if text == "(no genres listed)":
        return []
    return [tok for g in text.split("|") if (tok := normalize_text(g))]


def clean_movies_table(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Add clean_title / release_year / title_text / genres_list / genres_text columns."""
    movies = movies_df.copy()

    parsed = movies["title"].apply(split_title_year)
    movies["clean_title"] = parsed.apply(lambda t: t[0]).astype("string")
    movies["release_year"] = parsed.apply(lambda t: t[1]).astype("Int64")
    movies["title_text"] = movies["clean_title"].apply(normalize_text).astype("string")

    movies["genres_list"] = movies["genres"].apply(genres_to_tokens)
    movies["genres_text"] = movies["genres_list"].apply(lambda toks: " ".join(toks)).astype("string")

    logger.info("Cleaned movies table: %d rows.", len(movies))
    return movies


def aggregate_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the tags table (one row per user-tag) down to one row per movie:
        movieId | tag_count | tags_text
    """
    empty_cols = ["movieId", "tag_count", "tags_text"]
    if tags_df.empty:
        logger.warning("tags.csv has no rows.")
        return pd.DataFrame(columns=empty_cols).astype(
            {"movieId": "int64", "tag_count": "int64", "tags_text": "string"}
        )

    working = tags_df.copy()
    working["tag_clean"] = working["tag"].apply(normalize_text)
    working = working[working["tag_clean"] != ""]

    if working.empty:
        return pd.DataFrame(columns=empty_cols).astype(
            {"movieId": "int64", "tag_count": "int64", "tags_text": "string"}
        )

    grouped = working.groupby("movieId", as_index=False)["tag_clean"].agg(
        tag_count="count",
        tags_text=lambda vals: " ".join(sorted(set(vals))),
    )
    grouped["tags_text"] = grouped["tags_text"].astype("string")
    grouped["tag_count"] = grouped["tag_count"].astype("int64")

    logger.info("Aggregated tags for %d movies.", len(grouped))
    return grouped


def rating_stats_per_movie(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """movieId | rating_count | rating_mean | rating_median"""
    cols = ["movieId", "rating_count", "rating_mean", "rating_median"]
    if ratings_df.empty:
        return pd.DataFrame(columns=cols).astype(
            {"movieId": "int64", "rating_count": "int64", "rating_mean": "float64", "rating_median": "float64"}
        )

    stats = ratings_df.groupby("movieId", as_index=False)["rating"].agg(
        rating_count="count", rating_mean="mean", rating_median="median"
    )
    stats["rating_count"] = stats["rating_count"].astype("int64")
    logger.info("Built rating stats for %d movies.", len(stats))
    return stats


def make_metadata_text(row: pd.Series) -> str:
    """
    Combine title + genres (weighted x2, since they're reliable) + tags
    into one text blob for TF-IDF. Genres are repeated because MovieLens
    tags are noisy/sparse, so genres deserve a bit more weight.
    """
    title = row.get("title_text", "") or ""
    genres = row.get("genres_text", "") or ""
    tags = row.get("tags_text", "") or ""
    blob = " ".join(p for p in [title, genres, genres, tags] if str(p).strip())
    return normalize_text(blob)


def build_processed_table(dataset: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Full cleaning pipeline: merges movies + tags + links + rating stats into one table."""
    movies = clean_movies_table(dataset["movies"])
    tags_agg = aggregate_tags(dataset["tags"])
    rating_stats = rating_stats_per_movie(dataset["ratings"])

    out = movies.merge(tags_agg, on="movieId", how="left")
    out = out.merge(dataset["links"], on="movieId", how="left")
    out = out.merge(rating_stats, on="movieId", how="left")

    out["tag_count"] = out["tag_count"].fillna(0).astype("int64")
    out["tags_text"] = out["tags_text"].fillna("").astype("string")
    out["rating_count"] = out["rating_count"].fillna(0).astype("int64")
    out["rating_mean"] = out["rating_mean"].astype("float64")
    out["rating_median"] = out["rating_median"].astype("float64")
    out["has_tmdb_mapping"] = out["tmdbId"].notna()

    out["metadata_text"] = out.apply(make_metadata_text, axis=1).astype("string")

    final_cols = [
        "movieId", "title", "clean_title", "release_year", "genres",
        "genres_list", "genres_text", "tags_text", "tag_count",
        "imdbId", "tmdbId", "has_tmdb_mapping",
        "rating_count", "rating_mean", "rating_median", "metadata_text",
    ]
    result = out[final_cols].copy()
    logger.info("Processed table ready: %d rows, %d columns.", len(result), len(result.columns))
    return result


def save_processed(df: pd.DataFrame, output_path: str | Path = "data/processed/movies_metadata.csv") -> Path:
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export = df.copy()
    export["genres_list"] = export["genres_list"].apply(lambda v: "|".join(v) if isinstance(v, list) else "")
    export.to_csv(out_path, index=False)

    logger.info("Saved processed metadata -> %s", out_path)
    return out_path


def run_cleaning_pipeline(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/processed/movies_metadata.csv",
) -> pd.DataFrame:
    """load raw -> clean -> save -> return the processed DataFrame."""
    dataset = load_all(raw_dir=raw_dir)
    processed = build_processed_table(dataset)
    save_processed(processed, output_path)
    return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = run_cleaning_pipeline()
    print(f"Processed rows: {len(df)}")
    print(df.head(5)[["movieId", "clean_title", "release_year", "metadata_text"]])
