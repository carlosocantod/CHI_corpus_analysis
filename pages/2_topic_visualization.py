import pandas as pd
import plotly.express as px
import streamlit as st

from data_models import MetadataWithCluster
from data_models import MetadataWithScore
from data_models import TopWordsCluster
from data_models import TopWordsPositionsCluster
from home import metadata
from home import top_words_topic
from settings import APP_NAME


st.set_page_config(page_title=APP_NAME, page_icon="🏗️", layout="wide")

st.title("Topic Visualization")


_MIN_YEAR, _MAX_YEAR = metadata[MetadataWithCluster.year].min(), metadata[MetadataWithCluster.year].max()
min_year_selected, max_year_selected = st.slider(
    "Select Year Range",
    value=(_MIN_YEAR, _MAX_YEAR),
    min_value=_MIN_YEAR,
    max_value=_MAX_YEAR,
)

n_results = metadata.shape[0]

metadata_carto = metadata[
    (metadata[MetadataWithScore.year] >= min_year_selected) &
    (metadata[MetadataWithScore.year] <= max_year_selected)
]

n_results_filtered = metadata_carto.shape[0]

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
if top_words_topic_display.shape[0]>1:
    fig_centroids = px.scatter(
        top_words_topic_display,
        x=TopWordsPositionsCluster.x,
        y=TopWordsPositionsCluster.y,
        size="counts",
        size_max=max(1, int(30*n_results_filtered/n_results)),
        color=TopWordsPositionsCluster.cluster,
        hover_data={TopWordsPositionsCluster.top_words: True,
                    TopWordsPositionsCluster.x: False,
                    TopWordsPositionsCluster.y: False,
                    TopWordsPositionsCluster.counts: False},
        color_discrete_sequence=px.colors.qualitative.Dark24,
        height=500,
        opacity=0.8,
    )

    fig_centroids.update_layout(
        xaxis_range=[min_x, max_x],
        yaxis_range=[min_y, max_y],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    st.plotly_chart(fig_centroids)

    st.dataframe(top_words_topic_display[[TopWordsPositionsCluster.cluster, TopWordsPositionsCluster.top_words, TopWordsPositionsCluster.counts]], hide_index=True)
