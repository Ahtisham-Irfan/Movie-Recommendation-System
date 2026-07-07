"""Basic sanity tests for the cleaning + recommender pipeline."""

import pandas as pd

from src.data.cleaning import split_title_year, genres_to_tokens, normalize_text


def test_split_title_year_with_year():
    title, year = split_title_year("Toy Story (1995)")
    assert title == "Toy Story"
    assert year == 1995


def test_split_title_year_without_year():
    title, year = split_title_year("Heat")
    assert title == "Heat"
    assert pd.isna(year)


def test_genres_to_tokens_basic():
    assert genres_to_tokens("Adventure|Comedy|Fantasy") == ["adventure", "comedy", "fantasy"]


def test_genres_to_tokens_no_genres_listed():
    assert genres_to_tokens("(no genres listed)") == []


def test_normalize_text_lowercases_and_strips_punctuation():
    assert normalize_text("Toy Story!! (1995)") == "toy story 1995"


def test_normalize_text_handles_nan():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
