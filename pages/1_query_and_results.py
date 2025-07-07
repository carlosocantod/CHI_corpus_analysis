import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_models import Embeddings, MetadataWithScore
from settings import APP_NAME, DEFAULT_QUERY
from setup_streamlit import load_data

st.title(f"{APP_NAME} – Query & Results")

min_score = 0.4
number_of_results = st.select_slider("Number of results for display", options=[10, 50, 100], value=50)

model, embeddings, metadata, _ = load_data()

input_text = st.text_input("Enter search query", value=DEFAULT_QUERY)
input_embeddings = model.encode([input_text], show_progress_bar=False)
similarity_scores = cosine_similarity(input_embeddings, embeddings.drop(columns=Embeddings.doi).to_numpy()).ravel()
metadata[MetadataWithScore.score] = similarity_scores

metadata_display = metadata.sort_values(MetadataWithScore.score, ascending=False, ignore_index=True)
metadata_display = metadata_display[metadata_display[MetadataWithScore.score] >= min_score].iloc[:number_of_results]

st.text(f"Number of results above threshold: {len(metadata_display)}")
if len(metadata_display) > 0:
    st.dataframe(metadata_display, height=700)
