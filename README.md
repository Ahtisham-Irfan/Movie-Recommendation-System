# Movie Recommendation System

A content-based movie recommender built on the MovieLens (Latest Small) dataset.
Given a movie title, it returns the most similar movies based on title, genres,
and user tags — using TF-IDF text features and on-demand cosine similarity.

## What it does

- Loads and validates the four MovieLens raw CSVs (movies, ratings, tags, links)
- Cleans and merges them into one movie-level metadata table
- Builds a `metadata_text` field per movie (title + genres + tags)
- Fits a TF-IDF vectorizer over that text
- Scores a chosen movie against every other movie on demand (no giant
  precomputed similarity matrix sitting on disk)
- Serves everything through a small Flask API and a CLI

## Why on-demand scoring instead of a precomputed similarity matrix

A full movie-to-movie similarity matrix for ~9,700 movies is roughly 380MB
in memory (dense NxN floats), and it has to be rebuilt every time the
dataset changes. Since a single recommendation request only needs **one**
row of scores (the seed movie vs everyone else), it's cheaper to keep the
sparse TF-IDF matrix around and compute that one row when it's needed. The
dense matrix build is still available (`--build-similarity-matrix`) for
debugging or comparison, it's just not part of the default runtime path.

## Project structure

```
main.py                     CLI entry point (train / candidates / recommend / serve)
app/app.py                  Flask API
src/data/loader.py           Raw CSV loading + validation
src/data/cleaning.py         Title/year parsing, genre + tag cleaning, metadata_text
src/features/vectorize.py    TF-IDF fitting + artifact saving
src/recommender/engine.py    Title resolution + on-demand similarity scoring
src/models/pipeline.py       Wires the above into one end-to-end build pipeline
src/utils/support.py         Logging, JSON, timing helpers
tests/                       Unit tests for the cleaning functions
notebooks/eda.ipynb          Exploratory data analysis
data/raw/                    Place movies.csv, ratings.csv, tags.csv, links.csv here
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

Make sure `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv` are inside `data/raw/`.

## Usage

```bash
# 1. Run the full build pipeline (cleans data, fits TF-IDF, saves artifacts)
python main.py train

# 2. Not sure of the exact title? Search for it
python main.py candidates --query "heat" --limit 5

# 3. Get recommendations
python main.py recommend --title "Heat" --top-n 10 --min-rating-count 10

# 4. Or start the API
python main.py serve
```

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Service info |
| `GET /health` | Cheap liveness check (doesn't load the model) |
| `GET /ready` | Loads the engine and confirms it's usable |
| `GET /candidates?query=toy` | Partial title search |
| `GET /recommend?title=Toy+Story&top_n=10&min_rating_count=10` | Recommendations |

## Limitations

- Content-based only — no collaborative filtering, no hybrid ranking
- MovieLens small doesn't include plot summaries, only title/genres/tags
- Tag coverage is sparse (~1,500 of ~9,700 movies have any tags at all)

## Possible next steps

- Enrich metadata with TMDB plot overviews/keywords
- Add a collaborative-filtering baseline and blend it with this content score
- Popularity-aware re-ranking
- Package the API behind a production WSGI server
