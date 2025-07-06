"""
Simple streamlit application for querying CHI papers database
"""
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from data_models import Embeddings
from data_models import MetadataWithCluster, TopWordsPositionsCluster
from data_models import MetadataWithScore
from settings import APP_NAME
from settings import DEFAULT_QUERY
from setup_streamlit import load_data

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    model, embeddings, metadata, top_words_topic = load_data()

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

    _MIN_YEAR, _MAX_YEAR = metadata[MetadataWithCluster.year].min(), metadata[MetadataWithCluster.year].max()

    min_year_selected, max_year_selected = st.slider(
        "Years", value=(_MIN_YEAR, _MAX_YEAR), min_value=_MIN_YEAR, max_value=_MAX_YEAR)

    metadata_carto = metadata[
        (metadata[MetadataWithScore.year] >= min_year_selected) &
        (metadata[MetadataWithScore.year] <= max_year_selected)
    ]
    metadata_carto[MetadataWithScore.cluster] = metadata_carto[MetadataWithScore.cluster].astype(str)
    metadata_carto = metadata_carto.sort_values(TopWordsPositionsCluster.cluster)
    opacity = (metadata_carto[MetadataWithScore.cluster] != "-1")*0.7 + 0.3

    fig = px.scatter(
        data_frame=metadata_carto,
        x=MetadataWithScore.x,
        y=MetadataWithScore.y,
        opacity=0.5,
        color=MetadataWithScore.cluster,
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data={MetadataWithCluster.title: True,
                    MetadataWithScore.score: False,
                    MetadataWithScore.x: False,
                    MetadataWithScore.y: False},
        height=600,
    )

    fig.update_layout(
        yaxis=dict(showgrid=False, showticklabels=True, showline=False, zeroline=False), yaxis_title=None,
        xaxis=dict(showgrid=False, showticklabels=True, showline=False, zeroline=False), xaxis_title=None,
        yaxis_range=[min_y, max_y],
        xaxis_range=[min_x, max_x],
    )

    top_words_topic[TopWordsPositionsCluster.cluster] = top_words_topic[TopWordsPositionsCluster.cluster].astype(str)
    top_words_topic["counts"] = 1
    centroids_visu = top_words_topic.groupby(
        TopWordsPositionsCluster.cluster, as_index=False,
    ).agg({TopWordsPositionsCluster.x: "median", TopWordsPositionsCluster.y: "median",
           TopWordsPositionsCluster.top_words: "first", "counts": "count"}).sort_values(TopWordsPositionsCluster.cluster)
    fig2 = px.scatter(
        data_frame=centroids_visu,
        x=TopWordsPositionsCluster.x,
        y=TopWordsPositionsCluster.y,
        opacity=0.9,
        color=TopWordsPositionsCluster.cluster,
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data={TopWordsPositionsCluster.top_words: True,
                    TopWordsPositionsCluster.x: False,
                    TopWordsPositionsCluster.y: False},
        height=500,
        size="counts",
        size_max=30,
    )

    fig2.update_layout(
        yaxis=dict(showgrid=False, showticklabels=True, showline=False, zeroline=False), yaxis_title=None,
        xaxis=dict(showgrid=False, showticklabels=True, showline=False, zeroline=False), xaxis_title=None,
        yaxis_range=[min_y, max_y],
        xaxis_range=[min_x, max_x],
    )
    choice = st.radio("graph type", ["topic centroids", "all data"])
    if choice == "topic centroids":
        st.plotly_chart(figure_or_data=fig2)
    else:
        st.plotly_chart(figure_or_data=fig)




if __name__ == "__main__":
    main()
