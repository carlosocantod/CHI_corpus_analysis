from pathlib import Path

PATH_DATA = Path.home() / "Downloads"
PATH_RAW_CHI_METADATA = PATH_DATA / "CHI_raw.xls"
PATH_CLEAN_CHI_METADATA = PATH_DATA / "CHI_filtered.csv"
PATH_EMBEDDINGS = PATH_DATA / "embeddings.csv"
COL_ABSTRACT = "Abstracts"
COL_DOI = "DOI"
COL_COSINE_SIMILARITY = "cosine_score"
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
DEFAULT_QUERY = "doctors in participatory design"
APP_NAME = "CHI papers search engine"
