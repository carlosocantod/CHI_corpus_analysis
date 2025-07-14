from pathlib import Path

PATH_DATA = Path.home() / "git_repos" / "CHI_corpus_analysis" / "data"
PATH_RAW_CHI_METADATA = PATH_DATA / "CHI_raw.xls"
PATH_CLEAN_CHI_METADATA = PATH_DATA / "CHI_filtered.parquet"
PATH_CLEAN_CHI_METADATA_POSITIONS = PATH_DATA / "CHI_metadata.parquet"
PATH_CLEAN_CHI_METADATA_CLUSTERS = PATH_DATA / "CHI_metadata_clusters.parquet"
PATH_CLEAN_CHI_CLUSTERS_TOP_WORDS = PATH_DATA / "CHI_cluster_top_words.parquet"

PATH_EMBEDDINGS = PATH_DATA / "embeddings.parquet"
PATH_EMBEDDINGS_10d = PATH_DATA / "embeddings_10d.parquet"
PATH_SPARSE_EMBEDDINGS = PATH_DATA / "sparse_embeddings.parquet"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# why is v2 not supported?
# ValueError: Model prithivida/Splade_PP_en_v2 is not supported in SparseTextEmbedding.Please check the supported models using `SparseTextEmbedding.list_supported_models()`
SBERT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"

DEFAULT_QUERY = "doctors in participatory design"
APP_NAME = "CHI papers search engine"
CHROMA_DB_PATH = PATH_DATA / "chroma_db_vectors"
QDRANT_DB_PATH = PATH_DATA / "qdrant_db_vectors"
COLLECTION_NAME = "chi_collection"
COLLECTION_HYBRID_NAME = "hybrid_collection"

CHROMA_COLLECTION_NAME = "chi_collection"
SQL_DB_PATH_LOCAL = PATH_DATA / "metadata.db"
METADATA_TABLE_NAME = "METADATA_TABLE"
CLUSTER_TOP_WORDS_TABLE_NAME = "CLUSTER_TOP_WORDS"


DISTANCE_METRIC = "cosine"
