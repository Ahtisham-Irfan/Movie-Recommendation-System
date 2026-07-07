"""
pipeline.py

Runs the full build process end to end:
    1. load raw MovieLens CSVs
    2. clean + merge into one metadata table
    3. fit TF-IDF and save runtime artifacts
    4. sanity-check with a few sample recommendations
    5. write a JSON summary of everything that happened

This is what `main.py train` calls under the hood.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data.loader import load_all, quick_summary
from src.data.cleaning import build_processed_table, save_processed
from src.features.vectorize import build_features, save_features, FeatureBundle
from src.recommender.engine import RecommenderEngine
from src.utils.support import save_json

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_TITLES = ("Toy Story", "Jumanji", "Heat")


def run_pipeline(
    raw_dir: str | Path = "data/raw",
    processed_output_path: str | Path = "data/processed/movies_metadata.csv",
    artifact_dir: str | Path = "artifacts",
    summary_output_path: str | Path = "artifacts/training_summary.json",
    sample_titles: Tuple[str, ...] = DEFAULT_SAMPLE_TITLES,
    recommendation_top_n: int = 5,
    min_rating_count: int = 10,
    include_dense_similarity: bool = False,
) -> Dict[str, Any]:
    """Runs load -> clean -> vectorize -> sample-check -> save summary. Returns the summary dict."""

    logger.info("Step 1/4: loading raw MovieLens data from %s", raw_dir)
    raw_dataset = load_all(raw_dir=raw_dir)
    raw_summary = quick_summary(raw_dataset)

    logger.info("Step 2/4: cleaning + merging into one metadata table")
    processed_df = build_processed_table(raw_dataset)
    save_processed(processed_df, output_path=processed_output_path)

    logger.info("Step 3/4: building TF-IDF features (dense_similarity=%s)", include_dense_similarity)
    bundle: FeatureBundle = build_features(processed_df, include_dense_similarity=include_dense_similarity)
    saved_paths = save_features(bundle, artifact_dir=artifact_dir)

    logger.info("Step 4/4: sample recommendation sanity checks")
    engine = RecommenderEngine(
        movies_df=bundle.movies_df,
        tfidf_matrix=bundle.tfidf_matrix,
        id_to_row=bundle.id_to_row,
        row_to_id=bundle.row_to_id,
    )
    sample_recs = _sample_recommendation_check(engine, sample_titles, recommendation_top_n, min_rating_count)

    summary = {
        "raw_dataset_summary": raw_summary,
        "processed_movies_count": len(processed_df),
        "processed_columns": processed_df.columns.tolist(),
        "tfidf_matrix_shape": list(bundle.tfidf_matrix.shape),
        "dense_similarity_built": bundle.dense_similarity is not None,
        "vocabulary_size": len(bundle.vectorizer.vocabulary_),
        "saved_artifacts": {name: str(p) for name, p in saved_paths.items()},
        "sample_recommendations": sample_recs,
    }
    save_json(summary, summary_output_path)
    logger.info("Pipeline complete. Summary saved to %s", summary_output_path)
    return summary


def _sample_recommendation_check(
    engine: RecommenderEngine,
    titles: Tuple[str, ...],
    top_n: int,
    min_rating_count: int,
) -> Dict[str, List[dict]]:
    """Runs a handful of known titles through the engine as a quick sanity check."""
    output: Dict[str, List[dict]] = {}
    for title in titles:
        try:
            output[title] = engine.recommend_by_title(title, top_n=top_n, min_rating_count=min_rating_count)
        except ValueError as exc:
            candidates = engine.find_candidates(title, limit=5).to_dict(orient="records")
            output[title] = [{"error": str(exc), "candidate_titles": candidates}]
    return output


def print_summary(summary: Dict[str, Any]) -> None:
    print("Pipeline finished successfully.")
    print("-" * 70)
    print("Raw dataset:")
    for k, v in summary["raw_dataset_summary"].items():
        print(f"  {k}: {v}")

    print("-" * 70)
    print(f"Processed movies: {summary['processed_movies_count']}")
    print(f"TF-IDF matrix shape: {summary['tfidf_matrix_shape']}")
    print(f"Vocabulary size: {summary['vocabulary_size']}")

    print("-" * 70)
    print("Sample recommendations:")
    for title, recs in summary["sample_recommendations"].items():
        print(f"\n  Seed: {title}")
        for r in recs:
            if "error" in r:
                print(f"    error: {r['error']}")
                continue
            print(f"    - {r['title']} (score={r['similarity_score']})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    summary = run_pipeline()
    print_summary(summary)
