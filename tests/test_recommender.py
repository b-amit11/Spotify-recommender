from pathlib import Path

import pandas as pd
import pytest

from src.recommender import REQUIRED_COLUMNS, recommend, validate_catalogue

CATALOGUE = Path(__file__).resolve().parents[1] / "data" / "tracks.csv"


def make_catalogue(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


@pytest.fixture
def small():
    return make_catalogue(
        [
            ("Source", "A", "Pop", 0.80, 0.80, 0.60, 0.10, 120),
            ("Twin", "B", "Pop", 0.81, 0.79, 0.61, 0.11, 121),
            ("Opposite", "C", "Folk", 0.20, 0.15, 0.30, 0.90, 70),
            ("Middle", "D", "Folk", 0.50, 0.50, 0.45, 0.50, 95),
        ]
    )


def test_most_similar_track_ranks_first(small):
    results = recommend(small, track_index=0, limit=3)
    assert results["track_name"].tolist() == ["Twin", "Middle", "Opposite"]
    assert results["similarity"].is_monotonic_decreasing


def test_source_track_is_never_recommended(small):
    results = recommend(small, track_index=0, limit=10)
    assert "Source" not in results["track_name"].tolist()
    assert len(results) == 3


def test_similarity_is_bounded(small):
    results = recommend(small, track_index=0, limit=10)
    assert results["similarity"].between(-1, 1).all()


def test_genre_filter_limits_candidates(small):
    results = recommend(small, track_index=0, limit=10, genre="Folk")
    assert set(results["genre"]) == {"Folk"}


def test_empty_genre_returns_empty_frame(small):
    results = recommend(small, track_index=0, genre="Jazz")
    assert results.empty


def test_limit_is_respected_and_at_least_one(small):
    assert len(recommend(small, track_index=0, limit=2)) == 2
    assert len(recommend(small, track_index=0, limit=0)) == 1


def test_constant_feature_does_not_divide_by_zero(small):
    small["tempo"] = 100
    results = recommend(small, track_index=0, limit=3)
    assert results["similarity"].notna().all()
    assert results.iloc[0]["track_name"] == "Twin"


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="tempo"):
        validate_catalogue(pd.DataFrame({"track_name": ["x"], "artist_name": ["y"]}))


def test_non_numeric_rows_are_dropped(small):
    small["energy"] = small["energy"].astype(object)
    small.loc[3, "energy"] = "loud"
    cleaned = validate_catalogue(small)
    assert len(cleaned) == 3
    assert cleaned.index.tolist() == [0, 1, 2]


def test_needs_at_least_two_tracks(small):
    with pytest.raises(ValueError, match="at least two"):
        validate_catalogue(small.head(1))


def test_unknown_track_index_raises(small):
    with pytest.raises(IndexError):
        recommend(small, track_index=99)


def test_bundled_catalogue_is_valid():
    catalogue = validate_catalogue(pd.read_csv(CATALOGUE))
    assert len(catalogue) >= 10
    results = recommend(catalogue, track_index=0, limit=5)
    assert len(results) == 5
