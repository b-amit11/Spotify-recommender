# Spotify-Style Music Recommender

A content-based music recommendation app that finds similar tracks from audio features such as danceability, energy, valence, tempo, and acousticness. It includes a Streamlit interface and a small demo catalogue so it runs immediately after installation.

> This is an independent educational project; it does not use or claim affiliation with Spotify's services or branding.

## Features

- Recommends tracks using cosine similarity over standardized audio features.
- Lets users control the number of recommendations and filter by genre.
- Includes a transparent similarity score and track metadata.
- Accepts a replacement CSV catalogue for experimentation with other datasets.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit (normally `http://localhost:8501`).

## Data format

The app reads `data/tracks.csv`. To use a different catalogue, provide columns named `track_name`, `artist_name`, `genre`, `danceability`, `energy`, `valence`, `acousticness`, and `tempo`.
