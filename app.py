"""
Simple streamlit application for querying CHI papers database
"""
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from settings import COL_COSINE_SIMILARITY
from settings import COL_DOI
from settings import DEFAULT_QUERY
from settings import PATH_CLEAN_CHI_METADATA
from settings import PATH_EMBEDDINGS
from settings import SBERT_MODEL_NAME


@st.cache_data()
def load_data() -> tuple[SentenceTransformer, pd.DataFrame, pd.DataFrame]:
    """
    Load heavy items only once then cache
    :return: model, embeddings, metadata
    """
    model = SentenceTransformer(SBERT_MODEL_NAME)
    embeddings = pd.read_csv(PATH_EMBEDDINGS)
    metadata = pd.read_csv(PATH_CLEAN_CHI_METADATA)
    # TODO: remove this drop once data final is available
    metadata = metadata.drop(columns=["Abstract"])
    return model, embeddings, metadata


def main():
    """
    Main running loop
    """
    st.title("CHI papers search engine")
    c1, c2, _ = st.columns((4, 4, 4))
    with c1:
        min_score = st.number_input("Min similarity score", min_value=-1.0, max_value=1.0, value=0.4, step=0.05)
    with c2:
        number_of_results = st.select_slider(label="Number of results for display", options=[10, 50, 100], value=50)
    model, embeddings, metadata = load_data()
    input_text = st.text_input(label="input_text", value=DEFAULT_QUERY)
    input_embeddings = model.encode([input_text], show_progress_bar=False)
    similarity_scores = cosine_similarity(input_embeddings, embeddings.drop(columns=COL_DOI).to_numpy()).ravel()
    embeddings_aux = embeddings[[COL_DOI]].copy()
    embeddings_aux[COL_COSINE_SIMILARITY] = similarity_scores
    metadata = metadata.merge(embeddings_aux, on=COL_DOI, how="inner")
    metadata_display = metadata.sort_values(COL_COSINE_SIMILARITY, ascending=False, ignore_index=True)
    metadata_display = metadata_display[metadata_display[COL_COSINE_SIMILARITY] >= min_score].iloc[: number_of_results]
    metadata_display.index += 1
    st.text(f"Number of results for this query above threshold {len(metadata_display)}")
    if len(metadata_display) > 0:
        st.dataframe(metadata_display, height=700)


if __name__ == "__main__":
    main()
