"""
vectorize.py

Job of this file:
    Turn the cleaned metadata_text column into TF-IDF vectors, and save
    everything needed at runtime: the fitted vectorizer, the sparse
    matrix, and a movieId <-> row-index lookup.

Design choice:
    We keep the TF-IDF matrix sparse and score "one movie vs all" at
    request time, instead of pre-computing a full dense NxN similarity
    matrix. For ~9700 movies a dense matrix is only ~380MB, but there's
    no reason to pay that storage/load cost when on-demand scoring is
    just as fast for a single lookup. A dense build is kept optional
    for debugging.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

TEXT_COLUMN = "metadata_text"
ARTIFACT_DIR_DEFAULT = "artifacts"
DENSE_SIM_FILENAME = "similarity_matrix.npy"


class FeatureBundle:
    """Plain container for everything the recommender needs at runtime."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        tfidf_matrix: csr_matrix,
        vectorizer: TfidfVectorizer,
        id_to_row: Dict[int, int],
        row_to_id: Dict[int, int],
        dense_similarity: Optional[np.ndarray] = None,
    ):
        self.movies_df = movies_df
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        self.id_to_row = id_to_row
        self.row_to_id = row_to_id
        self.dense_similarity = dense_similarity


def _validate_input(movies_df: pd.DataFrame, text_column: str) -> None:
    required = {"movieId", "title", text_column}
    missing = required - set(movies_df.columns)
    if missing:
        raise ValueError(f"Processed movies table is missing column(s): {sorted(missing)}")
    if movies_df.empty:
        raise ValueError("Processed movies table is empty, nothing to vectorize.")


def _corpus_from(movies_df: pd.DataFrame, text_column: str) -> pd.Series:
    corpus = movies_df[text_column].fillna("").astype("string").str.strip()
    n_empty = int(corpus.eq("").sum())
    if n_empty:
        logger.warning("%d rows have empty '%s' text -> will get near-zero TF-IDF weight.", n_empty, text_column)
    return corpus


def make_vectorizer(
    max_features: Optional[int] = 20_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.8,
) -> TfidfVectorizer:
    """
    unigrams + bigrams so short phrases ('space war', 'high school') survive,
    english stopwords stripped, sublinear_tf so repeated words don't dominate.
    """
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def fit_vectorizer(corpus: pd.Series, vectorizer: Optional[TfidfVectorizer] = None) -> Tuple[TfidfVectorizer, csr_matrix]:
    vectorizer = vectorizer or make_vectorizer()
    matrix = vectorizer.fit_transform(corpus)
    logger.info("TF-IDF fit done -> shape=%s vocab_size=%d", matrix.shape, len(vectorizer.vocabulary_))
    return vectorizer, matrix


def build_dense_similarity(tfidf_matrix: csr_matrix) -> np.ndarray:
    """Optional, debugging-only: full movie-to-movie cosine similarity matrix."""
    logger.warning("Building a dense NxN similarity matrix — for debugging only, not used at runtime.")
    return cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32, copy=False)


def build_id_row_maps(movies_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    ids = movies_df["movieId"].astype("int64").tolist()
    id_to_row = {mid: i for i, mid in enumerate(ids)}
    row_to_id = {i: mid for i, mid in enumerate(ids)}
    return id_to_row, row_to_id


def build_features(
    movies_df: pd.DataFrame,
    text_column: str = TEXT_COLUMN,
    include_dense_similarity: bool = False,
) -> FeatureBundle:
    _validate_input(movies_df, text_column)
    corpus = _corpus_from(movies_df, text_column)
    vectorizer, tfidf_matrix = fit_vectorizer(corpus)
    id_to_row, row_to_id = build_id_row_maps(movies_df)

    dense_sim = build_dense_similarity(tfidf_matrix) if include_dense_similarity else None

    return FeatureBundle(
        movies_df=movies_df.copy(),
        tfidf_matrix=tfidf_matrix,
        vectorizer=vectorizer,
        id_to_row=id_to_row,
        row_to_id=row_to_id,
        dense_similarity=dense_sim,
    )


def load_processed_movies(path: str | Path = "data/processed/movies_metadata.csv") -> pd.DataFrame:
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed metadata not found at {file_path}. Run the cleaning pipeline first."
        )
    return pd.read_csv(file_path)


def save_features(bundle: FeatureBundle, artifact_dir: str | Path = ARTIFACT_DIR_DEFAULT) -> Dict[str, Path]:
    out_dir = Path(artifact_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_path = out_dir / "tfidf_vectorizer.pkl"
    with vec_path.open("wb") as f:
        pickle.dump(bundle.vectorizer, f)

    matrix_path = out_dir / "tfidf_matrix.npz"
    save_npz(matrix_path, bundle.tfidf_matrix)

    maps_path = out_dir / "movie_index_maps.pkl"
    with maps_path.open("wb") as f:
        pickle.dump({"id_to_row": bundle.id_to_row, "row_to_id": bundle.row_to_id}, f)

    saved = {"vectorizer": vec_path, "tfidf_matrix": matrix_path, "movie_index_maps": maps_path}

    sim_path = out_dir / DENSE_SIM_FILENAME
    if bundle.dense_similarity is not None:
        np.save(sim_path, bundle.dense_similarity)
        saved["dense_similarity"] = sim_path
    elif sim_path.exists():
        sim_path.unlink()  # remove stale artifact from an older run

    logger.info("Saved feature artifacts to %s", out_dir)
    return saved


def run_feature_pipeline(
    processed_path: str | Path = "data/processed/movies_metadata.csv",
    artifact_dir: str | Path = ARTIFACT_DIR_DEFAULT,
    include_dense_similarity: bool = False,
) -> FeatureBundle:
    movies_df = load_processed_movies(processed_path)
    bundle = build_features(movies_df, include_dense_similarity=include_dense_similarity)
    save_features(bundle, artifact_dir)
    return bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    bundle = run_feature_pipeline()
    print(f"Movies: {len(bundle.movies_df)}")
    print(f"TF-IDF matrix shape: {bundle.tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(bundle.vectorizer.vocabulary_)}")
