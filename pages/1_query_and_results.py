import math

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from data_models import Embeddings
from data_models import MetadataWithCluster
from data_models import MetadataWithScore
from home import embeddings
from home import metadata
from home import model
from settings import APP_NAME
from settings import DEFAULT_QUERY

st.set_page_config(page_title=APP_NAME, page_icon="🏗️", layout="wide")

st.title(f"{APP_NAME} – Query & Results")

min_score = 0.7


input_text = st.text_input("Enter search query", value=DEFAULT_QUERY)
if input_text:

    input_embeddings = model.encode([input_text], show_progress_bar=False)
    similarity_scores = cosine_similarity(input_embeddings, embeddings.drop(columns=Embeddings.doi).to_numpy()).ravel()
    metadata[MetadataWithScore.score] = similarity_scores

    metadata_display = metadata.sort_values(MetadataWithScore.score, ascending=False, ignore_index=True)
    metadata_display = metadata_display[metadata_display[MetadataWithScore.score] >= min_score]

    st.markdown(f"Number of results: **{len(metadata_display)}**")

    # Parameters
    ITEMS_PER_PAGE = 5
    columns = st.columns(4)
    with columns[0]:
        # Pagination controls
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=math.ceil(len(metadata_display) / ITEMS_PER_PAGE),
            step=1
        )

    # Filter data for current page
    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    page_data = metadata_display.iloc[start_idx:end_idx]

    # Display each entry in markdown
    for _, row in page_data.iterrows():
        st.markdown(f"""
    ### 📄 {row[MetadataWithCluster.title]}
    
    **Year:** {row[MetadataWithCluster.year]}  
    **Score:** {row["score"]:.3f}  

    **DOI:** [{row[MetadataWithCluster.doi]}](https://doi.org/{row[MetadataWithCluster.doi]})  
    **Scholar Link:** [View on Google Scholar]({row[MetadataWithCluster.scholar_link]})
    
    **Abstract:**  
    {row[MetadataWithCluster.abstract]}
    
    ---
    """)
    st.markdown("## Metadata full results")
    st.dataframe(metadata_display.drop(columns=[MetadataWithCluster.x, MetadataWithCluster.y]))
