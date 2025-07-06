"""
Created on Tue Dec 20 15:34:41 2022

@author: carlos
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from data_models import Embeddings
from data_models import Metadata
from data_models import MetadataWithPositions
from settings import DISTANCE_METRIC
from settings import PATH_CLEAN_CHI_METADATA
from settings import PATH_CLEAN_CHI_METADATA_POSITIONS
from settings import PATH_EMBEDDINGS
from settings import PATH_EMBEDDINGS_10d
from settings import PATH_RAW_CHI_METADATA
from settings import SBERT_MODEL_NAME
from utils import get_embeddings_from_dataframe


def main() -> None:
    """
    Simple script to process raw CHI extract. remove nans, and compute embeddings
    :return:
    """
    if PATH_CLEAN_CHI_METADATA.is_file():
        df_text = pd.read_parquet(PATH_CLEAN_CHI_METADATA)
        Metadata.validate(df_text, inplace=True)
    else:
        df_text = pd.read_excel(PATH_RAW_CHI_METADATA)
        df_text.columns = [c.lower() for c in df_text.columns]
        # we need an abstract in our analysis
        df_text = df_text[~df_text[Metadata.abstract].isnull()]
        # TODO: relaunch with clearner doi
        #df_text[Metadata.doi] = df_text[Metadata.doi].str.lstrip('https://')
        Metadata.validate(df_text, inplace=True)
        df_text.to_parquet(PATH_CLEAN_CHI_METADATA, index=False)

    if PATH_EMBEDDINGS.is_file():
        print("embeddings file exists")
    else:
        # only import sbert if file does not exist
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SBERT_MODEL_NAME)
        embeddings = model.encode(df_text[Metadata.abstract].tolist(), show_progress_bar=True)
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings[Embeddings.doi] = df_text[Metadata.doi]
        Embeddings.validate(df_embeddings, inplace=True)
        df_embeddings.to_parquet(PATH_EMBEDDINGS, index=False)

    if not PATH_EMBEDDINGS_10d.is_file():
        print("performing dim reduction")
        df_embeddings = pd.read_parquet(PATH_EMBEDDINGS)
        df_text = pd.read_parquet(PATH_CLEAN_CHI_METADATA)

        _input_embeddings_to_scale = StandardScaler().fit_transform(get_embeddings_from_dataframe(df_embeddings))

        dim_reduction_cluster = UMAP(
            n_components=10,
            random_state=0,
            min_dist=0.001,
            n_neighbors=10,
            metric=DISTANCE_METRIC,
        )

        embeddings_10d = pd.DataFrame(dim_reduction_cluster.fit_transform(X=_input_embeddings_to_scale))
        embeddings_10d[Embeddings.doi] = df_embeddings[Embeddings.doi]
        embeddings_10d.to_parquet(PATH_EMBEDDINGS_10d, index=False)

        dim_reduction_visu = UMAP(n_components=2, random_state=0, min_dist=0.99, metric=DISTANCE_METRIC)
        df_text[[MetadataWithPositions.x, MetadataWithPositions.y]] = pd.DataFrame(
            dim_reduction_visu.fit_transform(X=get_embeddings_from_dataframe(embeddings_10d))
        )
        MetadataWithPositions.validate(df_text, inplace=True)
        df_text.to_parquet(PATH_CLEAN_CHI_METADATA_POSITIONS, index=False)


if __name__ == "__main__":
    main()
