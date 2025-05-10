"""
Simple streamlit application for querying CHI papers database
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from pandera import check_types
from pandera.typing import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from data_models import Embeddings
from data_models import MetadataWithPositions
from data_models import MetadataWithScore
from settings import APP_NAME
from settings import DEFAULT_QUERY
from settings import PATH_CLEAN_CHI_METADATA_POSITIONS
from settings import PATH_EMBEDDINGS
from settings import SBERT_MODEL_NAME

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@check_types()
@st.cache_data()
def load_data() -> tuple[SentenceTransformer, DataFrame[Embeddings], DataFrame[MetadataWithPositions]]:
    """
    Load heavy items only once then cache
    :return: model, embeddings, metadata
    """
    model = SentenceTransformer(SBERT_MODEL_NAME)
    embeddings = pd.read_parquet(PATH_EMBEDDINGS)
    metadata = pd.read_parquet(PATH_CLEAN_CHI_METADATA_POSITIONS)
    return model, embeddings, metadata


def main() -> None:
    """
    Main running loop
    """
    st.title(APP_NAME)
    c1, c2, _ = st.columns((4, 4, 4))
    with c1:
        min_score = st.number_input("Min similarity score", min_value=-1.0, max_value=1.0, value=0.4, step=0.05)
    with c2:
        number_of_results = st.select_slider(label="Number of results for display", options=[10, 50, 100], value=50)
    model, embeddings, metadata = load_data()

    _margins = 1.05
    min_y, max_y = metadata[MetadataWithScore.y].min(), metadata[MetadataWithScore.y].max()
    min_y, max_y = min_y*_margins, max_y*_margins
    min_x, max_x = metadata[MetadataWithScore.x].min(), metadata[MetadataWithScore.x].max()
    min_x, max_x = min_x*_margins, max_x * _margins

    input_text = st.text_input(label="input_text", value=DEFAULT_QUERY)
    input_embeddings = model.encode([input_text], show_progress_bar=False)
    similarity_scores = cosine_similarity(input_embeddings, embeddings.drop(columns=Embeddings.doi).to_numpy()).ravel()
    metadata[MetadataWithScore.score] = similarity_scores
    metadata_display = metadata.sort_values(MetadataWithScore.score, ascending=False, ignore_index=True)
    metadata_display = metadata_display[metadata_display[MetadataWithScore.score] >= min_score].iloc[: number_of_results]
    st.text(f"Number of results for this query above threshold {len(metadata_display)}")
    if len(metadata_display) > 0:
        st.dataframe(metadata_display, height=700)

    _MIN_YEAR, _MAX_YEAR = metadata[MetadataWithPositions.year].min(), metadata[MetadataWithPositions.year].max()

    min_year_selected, max_year_selected = st.slider(
        "Years", value=(_MIN_YEAR, _MAX_YEAR), min_value=_MIN_YEAR, max_value=_MAX_YEAR)

    metadata_carto = metadata[
        (metadata[MetadataWithScore.year] >= min_year_selected) &
        (metadata[MetadataWithScore.year] <= max_year_selected)
    ]

    fig = px.scatter(
        data_frame=metadata_carto,
        x=MetadataWithScore.x,
        y=MetadataWithScore.y,
        opacity=0.5,
        hover_data={MetadataWithScore.title: True,
                    MetadataWithScore.score: False,
                    MetadataWithPositions.x: False,
                    MetadataWithPositions.y: False},
        height=600,
    )

    fig.update_layout(
        yaxis=dict(showgrid=False, showticklabels=True, showline=False, zeroline=False), yaxis_title=None,
        xaxis=dict(showgrid=False, showticklabels=True, showline=False, zeroline=False), xaxis_title=None,
        yaxis_range=[min_y, max_y],
        xaxis_range=[min_x, max_x],
    )
    st.plotly_chart(figure_or_data=fig)


if __name__ == "__main__":
    main()
