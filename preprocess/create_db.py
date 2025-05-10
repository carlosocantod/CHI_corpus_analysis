
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from settings import PATH_CLEAN_CHI_METADATA_POSITIONS, PATH_EMBEDDINGS
import pandas as pd
from data_models import Embeddings, MetadataWithPositions
from settings import SBERT_MODEL_NAME
from utils import get_embeddings_from_dataframe

df_metadata = MetadataWithPositions.validate(pd.read_parquet(PATH_CLEAN_CHI_METADATA_POSITIONS))
df_embeddings = Embeddings.validate(pd.read_parquet(PATH_EMBEDDINGS))

if any(df_metadata[MetadataWithPositions.doi] != df_embeddings[MetadataWithPositions.doi]):
    raise ValueError("Non consistent ids, metadata vs embeddings")

chroma_client = chromadb.PersistentClient()
collection = chroma_client.create_collection(name="my_collection")


collection.add(
    documents=df_metadata["abstract"].tolist(),
    metadatas=df_metadata[["year", "doi"]].to_dict(orient="records"),
    embeddings=get_embeddings_from_dataframe(df_embeddings),
    ids=df_metadata["doi"].tolist()
)

from langchain_community.vectorstores import Chroma

langchainChroma = Chroma(client=chroma_client,
                         collection_name="my_collection",
                         persist_directory="./chroma_langchain_db",
                         embedding_function=HuggingFaceEmbeddings(model_name=SBERT_MODEL_NAME))


# Perform a similarity search
query = "climate change impact on agriculture"
results = langchainChroma.similarity_search_with_score(query, k=5, filter={
    "$and": [
        {"year": {"$gte": 2015}},
        {"year": {"$lte": 2016}}
    ]
})

for doc in results:
    print(doc.page_content)
    print(doc.metadata)







