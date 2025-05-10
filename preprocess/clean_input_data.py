"""
Created on Tue Dec 20 15:34:41 2022

@author: carlos
"""
import pandas as pd

from data_models import Embeddings
from data_models import Metadata
from settings import PATH_CLEAN_CHI_METADATA
from settings import PATH_EMBEDDINGS
from settings import PATH_RAW_CHI_METADATA
from settings import SBERT_MODEL_NAME


def main() -> None:
    """
    Simple script to process raw CHI extract. remove nans, and compute embeddings
    :return:
    """
    if PATH_CLEAN_CHI_METADATA.is_file():
        df_text = pd.read_parquet(PATH_CLEAN_CHI_METADATA)
        Metadata.validate(df_text, inplace=True)
    else:
        df_text = pd.read_excel(PATH_RAW_CHI_METADATA)
        df_text.columns = [c.lower() for c in df_text.columns]
        # we need an abstract in our analysis
        df_text = df_text[~df_text[Metadata.abstract].isnull()]
        Metadata.validate(df_text, inplace=True)
        df_text.to_parquet(PATH_CLEAN_CHI_METADATA, index=False)

    if PATH_EMBEDDINGS.is_file():
        print("embeddings file exists")
    else:
        # only import sbert if file does not exist
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SBERT_MODEL_NAME)
        embeddings = model.encode(df_text[Metadata.abstract].tolist(), show_progress_bar=True)
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings[Embeddings.doi] = df_text[Metadata.doi]
        Embeddings.validate(df_embeddings, inplace=True)
        df_embeddings.to_parquet(PATH_EMBEDDINGS, index=False)


if __name__ == "__main__":
    main()
