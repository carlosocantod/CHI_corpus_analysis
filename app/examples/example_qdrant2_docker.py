from typing import Any
from typing import Generator

from fastembed import SparseEmbedding
from pandas import read_parquet
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.models import SparseVector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data_models import Embeddings
from data_models import Metadata
from data_models import SparseEmbeddingsDataModel
from settings import COLLECTION_HYBRID_NAME
from settings import DENSE_VECTOR_NAME
from settings import PATH_CLEAN_CHI_METADATA
from settings import PATH_EMBEDDINGS
from settings import PATH_SPARSE_EMBEDDINGS
from settings import SBERT_MODEL_NAME
from settings import SPARSE_VECTOR_NAME
from utils import get_embeddings_from_dataframe


def generate_named_vectors(
        vectors: list[float],
        sparse_vectors: list[models.SparseVector],
        dense_vector_name: str,
        sparse_vector_name: str,
) -> Generator[dict[str, float | SparseVector], Any, None]:
    for vector, sparse_vector in zip(vectors, sparse_vectors):
        yield {
            dense_vector_name: vector,
            sparse_vector_name: models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values
            ),
        }


def main() -> None:
    """
    Simple script for persistent local Qdrant DB
    """
    encoder = SentenceTransformer(SBERT_MODEL_NAME)

    client = QdrantClient(url="http://localhost:6333")
    # why is v2 not supported?
    # ValueError: Model prithivida/Splade_PP_en_v2 is not supported in SparseTextEmbedding.Please check the supported models using `SparseTextEmbedding.list_supported_models()`

    if not client.collection_exists(COLLECTION_HYBRID_NAME):
        client.create_collection(
            collection_name=COLLECTION_HYBRID_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                    distance=models.Distance.COSINE
                )
            },  # size and distance are model dependent
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )
        print(f"Created collection {COLLECTION_HYBRID_NAME}")

    embeddings = read_parquet(PATH_EMBEDDINGS)
    metadata = read_parquet(PATH_CLEAN_CHI_METADATA)
    sparse_embeddings = read_parquet(PATH_SPARSE_EMBEDDINGS)

    assert all(metadata[Metadata.doi] == embeddings[Embeddings.doi]) and all(metadata[Metadata.doi] == sparse_embeddings[SparseEmbeddingsDataModel.doi]) , "there is no correspondance ids embeddings and metadata"

    embeddings_raw = get_embeddings_from_dataframe(embeddings).tolist()
    payload = metadata[[Metadata.year, Metadata.doi]].to_dict(orient="records")

    embeddings_raw_sparse = [
        SparseEmbedding(
            indices=row[SparseEmbeddingsDataModel.sparse_indices],
            values=row[SparseEmbeddingsDataModel.sparse_values],
        ) for _, row in sparse_embeddings.iterrows()
    ]

    client.upload_collection(
        collection_name=COLLECTION_HYBRID_NAME,
        vectors=tqdm(generate_named_vectors(vectors=embeddings_raw,
                                            sparse_vectors=embeddings_raw_sparse,
                                            dense_vector_name=DENSE_VECTOR_NAME,
                                            sparse_vector_name=SPARSE_VECTOR_NAME),
                     desc="getting_named_vectors"),
        payload=payload,
        parallel=4,
        max_retries=3,
        ids=list(range(metadata.shape[0]))
    )
    print("uploaded my thing")


if __name__ == "__main__":
    main()
