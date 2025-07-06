import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from data_models import Embeddings
from data_models import MetadataWithPositions
from settings import CHROMA_COLLECTION_NAME
from settings import CHROMA_DB_PATH
from settings import PATH_CLEAN_CHI_METADATA_POSITIONS
from settings import PATH_EMBEDDINGS
from settings import SBERT_MODEL_NAME
from utils import get_embeddings_from_dataframe


def main() -> None:
    """
    Create a persistent chroma DB vectorstore with the previously computed embeddings

    :return: None
    """
    df_metadata = MetadataWithPositions.validate(pd.read_parquet(PATH_CLEAN_CHI_METADATA_POSITIONS))
    df_embeddings = Embeddings.validate(pd.read_parquet(PATH_EMBEDDINGS))

    if any(df_metadata[MetadataWithPositions.doi] != df_embeddings[MetadataWithPositions.doi]):
        raise ValueError("Non consistent ids, metadata vs embeddings")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH.__str__())
    collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME)
    collection.add(
        documents=df_metadata[MetadataWithPositions.abstract].tolist(),
        metadatas=df_metadata[[MetadataWithPositions.doi, MetadataWithPositions.year]].to_dict(orient="records"),
        embeddings=get_embeddings_from_dataframe(df_embeddings),
        ids=df_metadata[MetadataWithPositions.doi].tolist()
    )

    # ----------------------------------- example on how to use with Langchain ----------------------------------------
    langchain_chroma_db = Chroma(client=chroma_client,
                                 collection_name=CHROMA_COLLECTION_NAME,
                                 embedding_function=HuggingFaceEmbeddings(model_name=SBERT_MODEL_NAME))
    # Perform a similarity search
    query = "climate change impact on agriculture"
    results = langchain_chroma_db.similarity_search_with_score(query, k=5, filter={
        "$and": [
            {"year": {"$gte": 2015}},
            {"year": {"$lte": 2016}}
        ]
    })
    for doc in results:
        print(doc)


if __name__ == "__main__":
    main()
