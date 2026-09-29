"""Content-based music recommendation utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np

FEATURE_COLUMNS = ["danceability", "energy", "valence", "acousticness", "tempo"]
REQUIRED_COLUMNS = ["track_name", "artist_name", "genre", *FEATURE_COLUMNS]


def validate_catalogue(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a track catalogue before modelling it."""
    missing = set(REQUIRED_COLUMNS) - set(catalogue.columns)
    if missing:
        raise ValueError(f"Catalogue is missing required columns: {', '.join(sorted(missing))}")
    cleaned = catalogue.copy()
    cleaned[FEATURE_COLUMNS] = cleaned[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if len(cleaned) < 2:
        raise ValueError("The catalogue needs at least two complete tracks.")
    return cleaned


def recommend(catalogue: pd.DataFrame, track_index: int, limit: int = 5, genre: str = "All") -> pd.DataFrame:
    """Return the most similar tracks, excluding the selected source track."""
    cleaned = validate_catalogue(catalogue)
    if track_index not in cleaned.index:
        raise IndexError("Selected track is not in the catalogue.")
    candidates = cleaned if genre == "All" else cleaned[cleaned["genre"].eq(genre)]
    candidates = candidates.drop(index=track_index, errors="ignore")
    if candidates.empty:
        return candidates.assign(similarity=pd.Series(dtype=float))
    values = cleaned[FEATURE_COLUMNS].to_numpy(dtype=float)
    standard_deviation = values.std(axis=0)
    standard_deviation[standard_deviation == 0] = 1
    matrix = (values - values.mean(axis=0)) / standard_deviation
    selected_position = cleaned.index.get_loc(track_index)
    selected_vector = matrix[selected_position]
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(selected_vector)
    similarities = np.divide(matrix @ selected_vector, denominator, out=np.zeros_like(denominator), where=denominator != 0)
    scored = cleaned.assign(similarity=similarities)
    return scored.loc[candidates.index].sort_values("similarity", ascending=False).head(max(1, limit)).loc[:, ["track_name", "artist_name", "genre", "similarity"]].reset_index(drop=True)
