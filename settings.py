from pathlib import Path

PATH_DATA = Path.home() / "git_repos" / "CHI_corpus_analysis" / "data"
PATH_RAW_CHI_METADATA = PATH_DATA / "CHI_raw.xls"
PATH_CLEAN_CHI_METADATA = PATH_DATA / "CHI_filtered.parquet"
PATH_CLEAN_CHI_METADATA_POSITIONS = PATH_DATA / "CHI_metadata.parquet"
PATH_CLEAN_CHI_METADATA_CLUSTERS = PATH_DATA / "CHI_metadata_clusters.parquet"
PATH_EMBEDDINGS = PATH_DATA / "embeddings.parquet"
PATH_EMBEDDINGS_10d = PATH_DATA / "embeddings_10d.parquet"
SBERT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_QUERY = "doctors in participatory design"
APP_NAME = "CHI papers search engine"
CHROMA_DB_PATH = PATH_DATA / "chroma_db_vectors"
CHROMA_COLLECTION_NAME = "chi_collection"

DISTANCE_METRIC = "cosine"
