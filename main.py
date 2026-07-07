"""
main.py

Single command-line entry point for the whole project.

Commands:
    python main.py train       -> run the full build pipeline
    python main.py candidates  -> search for a movie title
    python main.py recommend   -> get similar movies for a title
    python main.py serve       -> start the Flask API
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.pipeline import run_pipeline, print_summary  # noqa: E402
from src.recommender.engine import load_engine  # noqa: E402
from src.utils.support import setup_logging, timed  # noqa: E402

logger = logging.getLogger(__name__)
LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


def _resolve(path_value: str | Path) -> Path:
    p = Path(path_value)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _log_level(value: str) -> int:
    upper = value.strip().upper()
    if upper not in LOG_LEVELS:
        raise ValueError(f"Invalid log level '{value}'. Choose from: {sorted(LOG_LEVELS)}")
    return getattr(logging, upper)


def _parse_titles(value: str) -> Tuple[str, ...]:
    titles = tuple(t.strip() for t in value.split(",") if t.strip())
    if not titles:
        raise ValueError("Provide at least one sample title.")
    return titles


def _print_df(df: pd.DataFrame) -> None:
    print(df.to_string(index=False) if not df.empty else "No rows found.")


def cmd_train(args: argparse.Namespace) -> int:
    with timed("full_pipeline", logger):
        summary = run_pipeline(
            raw_dir=_resolve(args.data_dir),
            processed_output_path=_resolve(args.processed_output_path),
            artifact_dir=_resolve(args.artifact_dir),
            summary_output_path=_resolve(args.summary_output_path),
            sample_titles=_parse_titles(args.sample_titles),
            recommendation_top_n=args.recommendation_top_n,
            min_rating_count=args.min_rating_count,
            include_dense_similarity=args.build_similarity_matrix,
        )
    print_summary(summary)
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    """Lightweight path: only reads the processed CSV, no TF-IDF artifacts needed."""
    processed_path = _resolve(args.processed_movies_path)
    query = (args.query or "").strip().lower()
    if not query:
        raise ValueError("Query cannot be empty.")

    with timed("candidate_lookup", logger):
        movies_df = pd.read_csv(processed_path)
        title_norm = movies_df["title"].fillna("").str.lower().str.strip()
        clean_norm = movies_df["clean_title"].fillna("").str.lower().str.strip()
        mask = title_norm.str.contains(query, regex=False) | clean_norm.str.contains(query, regex=False)
        matches = movies_df.loc[mask].sort_values(
            by=["rating_count", "rating_mean", "release_year"], ascending=[False, False, False]
        )
        cols = ["movieId", "title", "clean_title", "release_year", "genres", "rating_count", "rating_mean"]
        matches = matches[cols].head(args.limit).reset_index(drop=True)
    _print_df(matches)
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    with timed("recommend_lookup", logger):
        engine = load_engine(
            processed_movies_path=_resolve(args.processed_movies_path),
            artifact_dir=_resolve(args.artifact_dir),
        )
        results = engine.recommend_by_title(
            args.title,
            top_n=args.top_n,
            include_seed=args.include_input_movie,
            min_rating_count=args.min_rating_count,
        )
    _print_df(pd.DataFrame(results))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from app.app import create_app  # deferred import so other commands don't pay Flask's import cost
    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Movie Recommendation System CLI")
    parser.add_argument("--log-level", default="INFO", help="CRITICAL, ERROR, WARNING, INFO, or DEBUG")

    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run the full build pipeline")
    p_train.add_argument("--data-dir", default="data/raw")
    p_train.add_argument("--processed-output-path", default="data/processed/movies_metadata.csv")
    p_train.add_argument("--artifact-dir", default="artifacts")
    p_train.add_argument("--summary-output-path", default="artifacts/training_summary.json")
    p_train.add_argument("--sample-titles", default="Toy Story,Jumanji,Heat")
    p_train.add_argument("--recommendation-top-n", type=int, default=5)
    p_train.add_argument("--min-rating-count", type=int, default=10)
    p_train.add_argument(
        "--build-similarity-matrix", action="store_true",
        help="Also build the optional dense NxN similarity matrix (debugging only)",
    )
    p_train.set_defaults(func=cmd_train)

    p_cand = sub.add_parser("candidates", help="Search for a movie by partial title")
    p_cand.add_argument("--query", required=True)
    p_cand.add_argument("--limit", type=int, default=10)
    p_cand.add_argument("--processed-movies-path", default="data/processed/movies_metadata.csv")
    p_cand.set_defaults(func=cmd_candidates)

    p_rec = sub.add_parser("recommend", help="Get similar movies for a title")
    p_rec.add_argument("--title", required=True)
    p_rec.add_argument("--top-n", type=int, default=10)
    p_rec.add_argument("--min-rating-count", type=int, default=10)
    p_rec.add_argument("--include-input-movie", action="store_true")
    p_rec.add_argument("--processed-movies-path", default="data/processed/movies_metadata.csv")
    p_rec.add_argument("--artifact-dir", default="artifacts")
    p_rec.set_defaults(func=cmd_recommend)

    p_serve = sub.add_parser("serve", help="Start the Flask API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        setup_logging(_log_level(args.log_level))
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("Unhandled error")
        print(f"Unhandled error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
