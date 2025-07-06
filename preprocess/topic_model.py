import pandas as pd
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction

from data_models import Embeddings
from data_models import MetadataWithPositions
from settings import PATH_CLEAN_CHI_METADATA_POSITIONS
from settings import PATH_EMBEDDINGS_10d, PATH_EMBEDDINGS
from utils import get_embeddings_from_dataframe
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from data_models import MetadataWithCluster
from settings import PATH_CLEAN_CHI_METADATA_CLUSTERS

def main():

    df_metadata = MetadataWithPositions.validate(pd.read_parquet(PATH_CLEAN_CHI_METADATA_POSITIONS))
    df_embeddings = Embeddings.validate(pd.read_parquet(PATH_EMBEDDINGS))

    if any(df_metadata[MetadataWithPositions.doi] != df_embeddings[MetadataWithPositions.doi]):
        raise ValueError("Non consistent ids, metadata vs embeddings")

    embeddings = get_embeddings_from_dataframe(df_embeddings)

    topic_model = BERTopic(
        # umap_model=BaseDimensionalityReduction(),
        top_n_words=15,
        hdbscan_model=HDBSCAN(min_cluster_size=50, min_samples=5,))
    labels, probs = topic_model.fit_transform(embeddings=embeddings, documents=df_metadata[MetadataWithPositions.abstract].tolist())
    labels = np.array(labels)
    print(f"{(labels == -1).sum() / labels.shape[0]:.2f} {len(set(labels))}")
    df_metadata[MetadataWithCluster.cluster] = labels
    MetadataWithCluster.validate(df_metadata)
    df_metadata.to_parquet(path=PATH_CLEAN_CHI_METADATA_CLUSTERS)




    calis = {}
    for K in [5, 10, 25, 50, 75, 100, 110]:
        model = KMeans(random_state=1, n_clusters=K)
        model.fit(embeddings)
        calis[K] = silhouette_score(embeddings, labels=model.labels_)
    print(calis)


if __name__ == "__main__":
    main()
