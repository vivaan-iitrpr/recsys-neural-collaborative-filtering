"""
streamlit_app.py — Interactive Recommendation Demo.

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import get_model


@st.cache_data
def load_movies(data_dir="data/ml-25m"):
    """Load movie metadata."""
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    return movies


@st.cache_resource
def load_trained_model(checkpoint_path, num_users, num_items):
    """Load trained NeuMF model."""
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = get_model(
        model_name=config["model"],
        num_users=num_users,
        num_items=num_items,
        embed_dim=config["embed_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_recommendations(model, user_id, num_items, k=10, exclude=None):
    """Get top-K recommendations for a user."""
    exclude = exclude or set()
    device = torch.device("cpu")

    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id] * num_items).to(device)
        item_tensor = torch.arange(num_items).to(device)
        scores = torch.sigmoid(model(user_tensor, item_tensor)).numpy()

    # Exclude already seen items
    for item in exclude:
        if item < len(scores):
            scores[item] = -1

    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_scores = scores[top_k_indices]

    return list(zip(top_k_indices, top_k_scores))


# ---- Streamlit UI ----
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Neural Collaborative Filtering — Movie Recommender")
st.markdown(
    "Get personalized movie recommendations powered by the NeuMF model "
    "trained on the **MovieLens 25M dataset**."
)

st.divider()

# Sidebar
st.sidebar.header("Configuration")
user_id = st.sidebar.number_input("User ID", min_value=0, max_value=162000, value=42, step=1)
num_recs = st.sidebar.slider("Number of Recommendations", min_value=5, max_value=50, value=10)

st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.info(
    "This demo uses a Neural Collaborative Filtering (NeuMF) model that combines "
    "Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP) "
    "to learn user-item interaction patterns."
)

# Main content
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset", "MovieLens 25M")
col2.metric("Total Ratings", "25M")
col3.metric("Users", "162K")
col4.metric("Movies", "62K")

st.divider()

# Check if model exists
checkpoint_path = "checkpoints/best_neumf.pt"

if os.path.exists(checkpoint_path) and os.path.exists("data/ml-25m/movies.csv"):
    movies_df = load_movies()

    if st.button("🎯 Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            model = load_trained_model(checkpoint_path, num_users=162541, num_items=62423)
            recommendations = get_recommendations(model, user_id, num_items=62423, k=num_recs)

        st.subheader(f"Top {num_recs} Recommendations for User #{user_id}")

        for rank, (item_id, score) in enumerate(recommendations, 1):
            movie_info = movies_df[movies_df.index == item_id]
            if len(movie_info) > 0:
                title = movie_info.iloc[0]["title"]
                genres = movie_info.iloc[0]["genres"].replace("|", ", ")
            else:
                title = f"Movie #{item_id}"
                genres = "Unknown"

            with st.container():
                c1, c2, c3 = st.columns([1, 6, 2])
                c1.markdown(f"### #{rank}")
                c2.markdown(f"**{title}**")
                c2.caption(genres)
                c3.metric("Score", f"{score:.3f}")
else:
    st.warning(
        "⚠️ Model checkpoint or movie data not found. "
        "Please train the model first using `python src/train.py`"
    )

    st.markdown("### Demo Mode — Sample Output")
    st.markdown(
        "Below is an example of what the recommendations look like after training:"
    )

    demo_data = pd.DataFrame({
        "Rank": range(1, 11),
        "Movie": [
            "The Shawshank Redemption (1994)",
            "Pulp Fiction (1994)",
            "The Matrix (1999)",
            "Forrest Gump (1994)",
            "The Dark Knight (2008)",
            "Fight Club (1999)",
            "Inception (2010)",
            "Goodfellas (1990)",
            "The Silence of the Lambs (1991)",
            "Interstellar (2014)",
        ],
        "Genre": [
            "Drama", "Crime, Drama", "Action, Sci-Fi", "Drama, Romance",
            "Action, Crime, Drama", "Drama", "Action, Sci-Fi",
            "Crime, Drama", "Crime, Thriller", "Adventure, Drama, Sci-Fi",
        ],
        "Score": [0.97, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87],
    })
    st.dataframe(demo_data, use_container_width=True, hide_index=True)
