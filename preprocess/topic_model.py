import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.cluster import silhouette_score
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from data_models import Embeddings
from data_models import MetadataWithCluster
from data_models import MetadataWithPositions, TopWordsCluster
from settings import PATH_CLEAN_CHI_METADATA_CLUSTERS
from settings import PATH_CLEAN_CHI_METADATA_POSITIONS
from settings import PATH_EMBEDDINGS_10d, PATH_CLEAN_CHI_CLUSTERS_TOP_WORDS
from settings import SBERT_MODEL_NAME
from utils import get_embeddings_from_dataframe
from simplemma import text_lemmatizer


Ks = [25, 40, 50, 75, 90, 90]

vectorizer_model = CountVectorizer(stop_words="english")
representation_model = KeyBERTInspired()
model = SentenceTransformer(SBERT_MODEL_NAME)


def simple_lemmatize(text: str, greedy=None, lang="en") -> str:
    lemmas = text_lemmatizer(text=text, lang=lang, greedy=greedy)
    out_text = " ".join(lemma for lemma in lemmas if lemma.isalnum())
    return out_text


def main() -> None:
    """

    :return:
    """
    df_metadata = MetadataWithPositions.validate(pd.read_parquet(PATH_CLEAN_CHI_METADATA_POSITIONS))
    df_embeddings = Embeddings.validate(pd.read_parquet(PATH_EMBEDDINGS_10d))

    if any(df_metadata[MetadataWithPositions.doi] != df_embeddings[MetadataWithPositions.doi]):
        raise ValueError("Non consistent ids, metadata vs embeddings")

    embeddings = get_embeddings_from_dataframe(df_embeddings)
    linkage_matric = linkage(embeddings, method='ward')

    labels_metrics_dict = dict(K=list(), labels=list(), score=list())
    for K in tqdm(Ks):
        labels = fcluster(linkage_matric, t=K, criterion='maxclust')  # t = desired number of clusters
        labels_metrics_dict["K"].append(K)
        labels_metrics_dict["labels"].append(labels)
        labels_metrics_dict["score"].append(silhouette_score(embeddings, labels))

    best_idx = np.argmax(labels_metrics_dict["score"])
    best_K = labels_metrics_dict["K"][best_idx]
    print(f"Best K {best_K}")
    lemmatized_text = df_metadata[MetadataWithPositions.abstract].apply(simple_lemmatize).tolist()

    # for convenience we will use Bert Topic. We can get top words and such from this
    # TODO: change top n words display
    topic_model = BERTopic(
        top_n_words=20,
        hdbscan_model=AgglomerativeClustering(n_clusters=best_K),
        vectorizer_model=vectorizer_model,
        embedding_model=model,
        representation_model=representation_model,
    )

    labels, _ = topic_model.fit_transform(embeddings=embeddings,
                                          documents=lemmatized_text)
    labels = np.array(labels)
    df_metadata[MetadataWithCluster.cluster] = labels

    # get metadata clusters
    top_words_cluster = list()
    for label in set(labels):
        top_words = "; ".join([x[0] for x in topic_model.get_topic(label)])
        top_words_cluster.append({TopWordsCluster.cluster: label, TopWordsCluster.top_words: top_words})

    top_words_cluster = pd.DataFrame(top_words_cluster)

    # save output dataframes
    TopWordsCluster.validate(top_words_cluster)
    top_words_cluster.to_parquet(path=PATH_CLEAN_CHI_CLUSTERS_TOP_WORDS)

    MetadataWithCluster.validate(df_metadata)
    df_metadata.to_parquet(path=PATH_CLEAN_CHI_METADATA_CLUSTERS)


if __name__ == "__main__":
    main()
