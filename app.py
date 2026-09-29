"""Streamlit entry point for the music recommender."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import recommend, validate_catalogue

st.set_page_config(page_title="Music Recommender", page_icon="🎧", layout="wide")
st.title("🎧 Music Recommender")
st.caption("Discover tracks with similar audio-feature profiles.")


@st.cache_data
def load_catalogue() -> pd.DataFrame:
    path = Path(__file__).parent / "data" / "tracks.csv"
    return validate_catalogue(pd.read_csv(path))


catalogue = load_catalogue()
catalogue["label"] = catalogue["track_name"] + " — " + catalogue["artist_name"]

with st.sidebar:
    st.header("Controls")
    selected_label = st.selectbox("Choose a track", catalogue["label"].tolist())
    genres = ["All", *sorted(catalogue["genre"].unique())]
    genre = st.selectbox("Optional genre filter", genres)
    limit = st.slider("Recommendations", min_value=1, max_value=10, value=5)

selected_index = catalogue.index[catalogue["label"].eq(selected_label)][0]
selected = catalogue.loc[selected_index]
st.subheader(f"Because you selected {selected['track_name']}")
st.write(f"**Artist:** {selected['artist_name']} · **Genre:** {selected['genre']}")

results = recommend(catalogue.drop(columns="label"), selected_index, limit, genre)
if results.empty:
    st.info("No other tracks match that genre filter. Choose another genre or select All.")
else:
    results["similarity"] = results["similarity"].map(lambda value: f"{value:.0%}")
    st.dataframe(results.rename(columns={"similarity": "Similarity"}), hide_index=True, use_container_width=True)

with st.expander("How it works"):
    st.write("The app standardizes five audio features and ranks tracks by cosine similarity. Similarity is a relative score within the current catalogue, not a measure of listener preference.")
