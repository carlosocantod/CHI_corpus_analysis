from itertools import batched

from fastembed import SparseEmbedding
from fastembed import SparseTextEmbedding
from fastembed import TextEmbedding
from numpy import ndarray
from pandera import check_types
from pandera.typing import DataFrame
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models
from tqdm import tqdm

from data_models import Embeddings
from settings import SBERT_MODEL_NAME
from settings import SPARSE_MODEL_NAME


class HybridSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = QdrantClient(url="http://localhost:6333")
        self.model_dense = TextEmbedding(model_name=SBERT_MODEL_NAME)
        self.model_sparse = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    def search(
            self,
            documents: str | list[str],
            limit: int = 10,
            limit_dense: int = 2_000,
            score_threshold_dense: float = 0.7,
            limit_sparse: int = 1_000,
            query_filter: None | BaseModel = None,
    ):
        dense_embeddings = list(self.model_dense.embed(documents=documents))[0]
        sparse_embeddings = list(self.model_sparse.embed(documents=documents))[0]
        sparse_embeddings = models.SparseVector(
            indices=sparse_embeddings.indices.tolist(),
            values=sparse_embeddings.values.tolist(),
        )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF  # we are using reciprocal rank fusion here
            ),
            prefetch=[
                models.Prefetch(
                    query=dense_embeddings,
                    score_threshold=score_threshold_dense,
                    limit=limit_dense,
                    using="dense",

                ),
                models.Prefetch(
                    query=sparse_embeddings,
                    limit=limit_sparse,
                    using="sparse",
                ),
            ],
            query_filter=query_filter,  # If you don't want any filters for now
            limit=limit,  # 5 the closest results
        ).points

        return search_result



@check_types()
def get_embeddings_from_dataframe(dataframe_embeddings: DataFrame[Embeddings]) -> ndarray:
    """

    :param dataframe_embeddings:
    :return:
    """
    return dataframe_embeddings.drop(columns=[Embeddings.doi]).to_numpy()


def obtain_sparse_vectors(
        model_sparse: SparseTextEmbedding,
        documents: list[str],
        batch_size: int = 32,
) -> list[SparseEmbedding]:
    """

    :param model_sparse:
    :param documents:
    :param batch_size:
    :return:
    """
    sparse_embeddings = []
    batch_documents = batched(documents, batch_size)
    with tqdm(total=len(documents), desc="Encoding") as pbar:
        for batch in batch_documents:
            batch_embeddings = list(model_sparse.embed(batch))
            sparse_embeddings.extend(batch_embeddings)
            pbar.update(len(batch))
    return sparse_embeddings
