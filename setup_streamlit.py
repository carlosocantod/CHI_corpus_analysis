import pandas as pd
import streamlit as st
from pandera import check_types
from pandera.typing import DataFrame
from sentence_transformers import SentenceTransformer

from data_models import Embeddings
from data_models import MetadataWithCluster
from settings import PATH_CLEAN_CHI_METADATA_CLUSTERS
from settings import PATH_EMBEDDINGS
from settings import SBERT_MODEL_NAME


@check_types()
@st.cache_data()
def load_data() -> tuple[SentenceTransformer, DataFrame[Embeddings], DataFrame[MetadataWithCluster]]:
    """
    Load heavy items only once then cache
    :return: model, embeddings, metadata
    """
    model = SentenceTransformer(SBERT_MODEL_NAME)
    embeddings = pd.read_parquet(PATH_EMBEDDINGS)
    metadata = pd.read_parquet(PATH_CLEAN_CHI_METADATA_CLUSTERS)
    return model, embeddings, metadata
