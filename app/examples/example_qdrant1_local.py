from pandas import read_parquet
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from data_models import Embeddings
from data_models import Metadata
from settings import COLLECTION_NAME
from settings import PATH_CLEAN_CHI_METADATA
from settings import PATH_EMBEDDINGS
from settings import QDRANT_DB_PATH
from settings import SBERT_MODEL_NAME
from utils import get_embeddings_from_dataframe


def main() -> None:
    """
    Simple script for persistent local Qdrant DB
    """
    encoder = SentenceTransformer(SBERT_MODEL_NAME)

    client = QdrantClient(path=QDRANT_DB_PATH)

    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created collection {COLLECTION_NAME}")

        embeddings = read_parquet(PATH_EMBEDDINGS)
        metadata = read_parquet(PATH_CLEAN_CHI_METADATA)

        assert all(metadata[Metadata.doi] == embeddings[Embeddings.doi]), "there is no correspondance ids embeddings and metadata"

        embeddings_raw = get_embeddings_from_dataframe(embeddings).tolist()
        years = metadata[[Metadata.year, Metadata.doi]].to_dict(orient="records")

        client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=embeddings_raw,
            payload=years,
            parallel=4,
            max_retries=3,
        )
        print("uploaded my thing")


if __name__ == "__main__":
    main()
