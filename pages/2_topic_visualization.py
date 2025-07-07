import pandas as pd
import plotly.express as px
import streamlit as st

from data_models import MetadataWithCluster
from data_models import MetadataWithScore
from data_models import TopWordsCluster
from data_models import TopWordsPositionsCluster
from setup_streamlit import load_data

st.title("Topic Visualization")

_, _, metadata, top_words_topic = load_data()

_MIN_YEAR, _MAX_YEAR = metadata[MetadataWithCluster.year].min(), metadata[MetadataWithCluster.year].max()
min_year_selected, max_year_selected = st.slider("Select Year Range", value=(_MIN_YEAR, _MAX_YEAR),
                                                  min_value=_MIN_YEAR, max_value=_MAX_YEAR)

metadata_carto = metadata[
    (metadata[MetadataWithScore.year] >= min_year_selected) &
    (metadata[MetadataWithScore.year] <= max_year_selected)
]

metadata_carto[MetadataWithScore.cluster] = metadata_carto[MetadataWithScore.cluster].astype(str)
metadata_carto = metadata_carto.sort_values(TopWordsCluster.cluster)
metadata_carto["counts"] = 1

margins = 1.05
min_y, max_y = metadata[MetadataWithScore.y].min() * margins, metadata[MetadataWithScore.y].max() * margins
min_x, max_x = metadata[MetadataWithScore.x].min() * margins, metadata[MetadataWithScore.x].max() * margins

fig_all = px.scatter(
    metadata_carto,
    x=MetadataWithScore.x, y=MetadataWithScore.y,
    opacity=0.5,
    color=MetadataWithScore.cluster,
    color_discrete_sequence=px.colors.qualitative.Dark24,
    hover_data={MetadataWithCluster.title: True},
    height=600
)

fig_all.update_layout(
    xaxis_range=[min_x, max_x], yaxis_range=[min_y, max_y],
    xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
)

centroids_visu = metadata_carto.groupby(TopWordsPositionsCluster.cluster, as_index=False).agg({
    TopWordsPositionsCluster.x: "median",
    TopWordsPositionsCluster.y: "median",
    "counts": "count",
}).sort_values(TopWordsPositionsCluster.cluster)

top_words_topic[TopWordsCluster.cluster] = top_words_topic[TopWordsCluster.cluster].astype(str)
top_words_topic_display = pd.merge(
    top_words_topic, centroids_visu, on=TopWordsPositionsCluster.cluster, how="inner"
)

fig_centroids = px.scatter(
    top_words_topic_display,
    x=TopWordsPositionsCluster.x, y=TopWordsPositionsCluster.y,
    size="counts", size_max=30,
    color=TopWordsPositionsCluster.cluster,
    hover_data={TopWordsPositionsCluster.top_words: True},
    color_discrete_sequence=px.colors.qualitative.Dark24,
    height=500,
    opacity=0.8
)

fig_centroids.update_layout(
    xaxis_range=[min_x, max_x], yaxis_range=[min_y, max_y],
    xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
)

choice = st.radio("Choose graph type", ["Topic Centroids", "All Data"])
st.plotly_chart(fig_centroids if choice == "Topic Centroids" else fig_all)
