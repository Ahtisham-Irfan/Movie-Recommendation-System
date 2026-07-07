"""
app.py

A thin Flask layer over the recommender engine.

Endpoints:
    GET /                -> service info
    GET /health           -> cheap liveness check (does NOT load the model)
    GET /ready             -> loads the engine and confirms it works
    GET /candidates?query= -> partial title search (metadata only, no TF-IDF)
    GET /recommend?title=  -> top-N similar movies for a title
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.recommender.engine import load_engine  # noqa: E402
from src.utils.support import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_value: str | None, default_relative: str) -> Path:
    if not env_value:
        return (PROJECT_ROOT / default_relative).resolve()
    candidate = Path(env_value)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _processed_path() -> Path:
    return _resolve_path(os.getenv("PROCESSED_MOVIES_PATH"), "data/processed/movies_metadata.csv")


def _artifact_dir() -> Path:
    return _resolve_path(os.getenv("ARTIFACT_DIR"), "artifacts")


@lru_cache(maxsize=1)
def get_engine():
    """Load the recommender engine once and reuse it across requests."""
    logger.info("Loading recommender engine (processed=%s, artifacts=%s)", _processed_path(), _artifact_dir())
    return load_engine(processed_movies_path=_processed_path(), artifact_dir=_artifact_dir())


def create_app() -> Flask:
    setup_logging()
    app = Flask(__name__)

    @app.get("/")
    def index():
        return jsonify({
            "service": "Movie Recommendation API",
            "status": "ok",
            "type": "content_based (TF-IDF, on-demand scoring)",
            "endpoints": {
                "health": "/health",
                "ready": "/ready",
                "candidates": "/candidates?query=toy",
                "recommend": "/recommend?title=Toy+Story&top_n=10&min_rating_count=10",
            },
        })

    @app.get("/health")
    def health():
        """Cheap liveness check — does not force the engine to load."""
        try:
            return jsonify({
                "status": "ok",
                "liveness": "alive",
                "engine_loaded": get_engine.cache_info().currsize > 0,
                "processed_movies_path": str(_processed_path()),
                "artifact_dir": str(_artifact_dir()),
            })
        except Exception as exc:
            logger.exception("Health check failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.get("/ready")
    def ready():
        """Heavier check — actually loads the engine and reports its shape."""
        try:
            engine = get_engine()
            return jsonify({
                "status": "ok",
                "readiness": "ready",
                "movies_count": len(engine.movies_df),
                "tfidf_matrix_shape": list(engine.tfidf_matrix.shape),
            })
        except Exception as exc:
            logger.exception("Readiness check failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.get("/candidates")
    def candidates():
        query = request.args.get("query", "", type=str).strip()
        limit = request.args.get("limit", default=10, type=int)

        if not query:
            return jsonify({"error": "Missing required query parameter: query"}), 400
        if limit <= 0:
            return jsonify({"error": "limit must be greater than 0"}), 400

        try:
            engine = get_engine()
            matches = engine.find_candidates(query, limit=limit)
            return jsonify({
                "query": query,
                "limit": limit,
                "count": len(matches),
                "matches": matches.to_dict(orient="records"),
            })
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.exception("Candidate lookup failed")
            return jsonify({"error": str(exc)}), 500

    @app.get("/recommend")
    def recommend():
        title = request.args.get("title", "", type=str).strip()
        top_n = request.args.get("top_n", default=10, type=int)
        min_rating_count = request.args.get("min_rating_count", default=10, type=int)
        include_seed = request.args.get("include_input_movie", default=False, type=_parse_bool)

        if not title:
            return jsonify({"error": "Missing required query parameter: title"}), 400
        if top_n <= 0:
            return jsonify({"error": "top_n must be greater than 0"}), 400
        if min_rating_count < 0:
            return jsonify({"error": "min_rating_count cannot be negative"}), 400

        try:
            engine = get_engine()
            results = engine.recommend_by_title(
                title, top_n=top_n, include_seed=include_seed, min_rating_count=min_rating_count
            )
            return jsonify({
                "query": title,
                "top_n": top_n,
                "min_rating_count": min_rating_count,
                "include_input_movie": include_seed,
                "count": len(results),
                "recommendations": results,
            })
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.exception("Recommendation request failed")
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = _parse_bool(os.getenv("FLASK_DEBUG"), default=False)
    app.run(host=host, port=port, debug=debug)
