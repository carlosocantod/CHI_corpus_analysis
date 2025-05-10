from pathlib import Path

PATH_DATA = Path.home() / "git_repos" / "CHI_corpus_analysis" / "data"
PATH_RAW_CHI_METADATA = PATH_DATA / "CHI_raw.xls"
PATH_CLEAN_CHI_METADATA = PATH_DATA / "CHI_filtered.parquet"
PATH_EMBEDDINGS = PATH_DATA / "embeddings.parquet"
SBERT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_QUERY = "doctors in participatory design"
APP_NAME = "CHI papers search engine"
