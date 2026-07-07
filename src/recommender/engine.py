"""
engine.py

Job of this file:
    Load saved TF-IDF artifacts, resolve a title to a movieId, then score
    that one movie against every other movie (on demand) to return the
    top-N most similar titles.

Why on-demand scoring instead of a precomputed similarity matrix:
    We only ever need one movie's row of scores per request. Computing
    that row on the fly (source_vector vs whole matrix) is cheap and
    avoids keeping/loading a dense N x N matrix in memory just to read
    a single row out of it.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)


class RecommenderEngine:
    """Holds the loaded artifacts and exposes the recommend/search methods."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        tfidf_matrix: csr_matrix,
        id_to_row: Dict[int, int],
        row_to_id: Dict[int, int],
    ):
        self.movies_df = movies_df
        self.tfidf_matrix = tfidf_matrix
        self.id_to_row = id_to_row
        self.row_to_id = row_to_id
        self._validate()

        # normalized title columns, precomputed once for fast lookups
        self._search_df = movies_df.copy()
        self._search_df["_title_norm"] = self._search_df["title"].fillna("").str.lower().str.strip()
        self._search_df["_clean_title_norm"] = self._search_df["clean_title"].fillna("").str.lower().str.strip()

    def _validate(self) -> None:
        n_movies = len(self.movies_df)
        if n_movies == 0:
            raise ValueError("movies_df is empty — nothing to recommend from.")
        if self.tfidf_matrix.shape[0] != n_movies:
            raise ValueError(
                f"tfidf_matrix rows ({self.tfidf_matrix.shape[0]}) don't match "
                f"movies_df rows ({n_movies}) — artifacts are out of sync, retrain."
            )
        if len(self.id_to_row) != n_movies or len(self.row_to_id) != n_movies:
            raise ValueError("Index maps don't match movies_df size — retrain the pipeline.")

    # ---------- title search ----------

    def find_candidates(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Partial, case-insensitive title search — doesn't need TF-IDF at all."""
        query = (query or "").strip().lower()
        if not query:
            raise ValueError("Search query cannot be empty.")

        mask = (
            self._search_df["_title_norm"].str.contains(query, regex=False)
            | self._search_df["_clean_title_norm"].str.contains(query, regex=False)
        )
        matches = self._search_df.loc[mask].copy()
        if matches.empty:
            return matches

        matches = matches.sort_values(
            by=["rating_count", "rating_mean", "release_year"],
            ascending=[False, False, False],
        )
        cols = ["movieId", "title", "clean_title", "release_year", "genres", "rating_count", "rating_mean"]
        return matches[cols].head(limit).reset_index(drop=True)

    def resolve_title(self, title: str) -> int:
        """Exact match first (title or clean_title), partial match as fallback."""
        title = (title or "").strip()
        if not title:
            raise ValueError("Title cannot be empty.")

        query = title.lower()
        exact = self._search_df[
            (self._search_df["_title_norm"] == query) | (self._search_df["_clean_title_norm"] == query)
        ]
        if not exact.empty:
            exact = exact.sort_values(
                by=["rating_count", "rating_mean", "release_year"], ascending=[False, False, False]
            )
            return int(exact.iloc[0]["movieId"])

        partial = self.find_candidates(title, limit=1)
        if partial.empty:
            raise ValueError(f"No movie found matching title: '{title}'")
        return int(partial.iloc[0]["movieId"])

    # ---------- recommendations ----------

    def _score_against_all(self, row_index: int) -> np.ndarray:
        """cosine similarity of one row vs the whole matrix (TF-IDF rows are L2-normalized,
        so linear_kernel == cosine_similarity here, without needing sklearn's cosine_similarity call)."""
        vector = self.tfidf_matrix[row_index]
        return linear_kernel(vector, self.tfidf_matrix).ravel()

    def recommend_by_id(
        self,
        movie_id: int,
        top_n: int = 10,
        include_seed: bool = False,
        min_rating_count: int = 0,
    ) -> List[dict]:
        if movie_id not in self.id_to_row:
            raise ValueError(f"movieId {movie_id} not found in the loaded index.")
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0.")

        row_index = self.id_to_row[movie_id]
        scores = self._score_against_all(row_index)
        ranked_rows = np.argsort(scores)[::-1]

        results: List[dict] = []
        for candidate_row in ranked_rows:
            candidate_id = self.row_to_id[int(candidate_row)]

            if not include_seed and candidate_id == movie_id:
                continue

            movie_row = self.movies_df.iloc[int(candidate_row)]
            rating_count = int(movie_row["rating_count"]) if not pd.isna(movie_row["rating_count"]) else 0
            if rating_count < min_rating_count:
                continue

            results.append(self._to_result_dict(movie_row, float(scores[int(candidate_row)])))
            if len(results) >= top_n:
                break

        logger.info("recommend_by_id(%d) -> %d results", movie_id, len(results))
        return results

    def recommend_by_title(
        self,
        title: str,
        top_n: int = 10,
        include_seed: bool = False,
        min_rating_count: int = 0,
    ) -> List[dict]:
        movie_id = self.resolve_title(title)
        return self.recommend_by_id(
            movie_id, top_n=top_n, include_seed=include_seed, min_rating_count=min_rating_count
        )

    @staticmethod
    def _to_result_dict(row: pd.Series, score: float) -> dict:
        def _int_or_none(v):
            return None if pd.isna(v) else int(v)

        def _float_or_none(v):
            return None if pd.isna(v) else float(v)

        return {
            "movie_id": int(row["movieId"]),
            "title": str(row.get("title", "")),
            "clean_title": str(row.get("clean_title", "")),
            "release_year": _int_or_none(row.get("release_year")),
            "similarity_score": round(score, 4),
            "genres": str(row.get("genres", "")),
            "rating_count": int(row.get("rating_count", 0)) if not pd.isna(row.get("rating_count", 0)) else 0,
            "rating_mean": _float_or_none(row.get("rating_mean")),
        }


# ---------- loading from disk ----------

def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def load_engine(
    processed_movies_path: str | Path = "data/processed/movies_metadata.csv",
    artifact_dir: str | Path = "artifacts",
) -> RecommenderEngine:
    """Load everything needed from disk and return a ready-to-use engine."""
    processed_path = Path(processed_movies_path).resolve()
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed movies file not found at {processed_path}. Run the cleaning pipeline first."
        )
    movies_df = pd.read_csv(processed_path)

    artifact_base = Path(artifact_dir).resolve()
    matrix_path = artifact_base / "tfidf_matrix.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"TF-IDF matrix not found at {matrix_path}. Run the feature pipeline first.")

    tfidf_matrix = load_npz(matrix_path)
    if not isinstance(tfidf_matrix, csr_matrix):
        tfidf_matrix = tfidf_matrix.tocsr()

    index_maps = _load_pickle(artifact_base / "movie_index_maps.pkl")

    engine = RecommenderEngine(
        movies_df=movies_df,
        tfidf_matrix=tfidf_matrix,
        id_to_row=index_maps["id_to_row"],
        row_to_id=index_maps["row_to_id"],
    )
    logger.info("Recommender engine loaded: %d movies.", len(movies_df))
    return engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    engine = load_engine()
    seed = "Toy Story"
    print(f"Recommendations for: {seed}")
    for r in engine.recommend_by_title(seed, top_n=10, min_rating_count=10):
        print(f"  {r['title']:40s} score={r['similarity_score']:.3f}  genres={r['genres']}")
