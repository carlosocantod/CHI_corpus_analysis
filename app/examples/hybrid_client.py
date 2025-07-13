from fastembed import SparseTextEmbedding
from fastembed import TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models

from home import embeddings
from settings import SBERT_MODEL_NAME
from settings import SPARSE_MODEL_NAME, COLLECTION_HYBRID_NAME


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


searcher = HybridSearcher(collection_name=COLLECTION_HYBRID_NAME)


text = "Principles of information-oriented graphic design have been utilized in redesigning the interface for a large information management system. These principles are explained and examples of typical screen formats are shown to indicate the nature of improvements."
results = searcher.search(documents=[text])

print(results)
print(1)

model_dense = TextEmbedding(model_name=SBERT_MODEL_NAME)

embedding = list(model_dense.embed(documents=text))[0]
client = QdrantClient(url="http://localhost:6333")
client.query_points(collection_name=COLLECTION_HYBRID_NAME,
                    query =embedding,
                    using="dense",
                    )


