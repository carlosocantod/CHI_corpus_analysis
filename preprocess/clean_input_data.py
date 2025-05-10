"""
Created on Tue Dec 20 15:34:41 2022

@author: carlos
"""
import pandas as pd

from settings import COL_ABSTRACT
from settings import COL_DOI
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
        df = pd.read_csv(PATH_CLEAN_CHI_METADATA)
    else:
        df = pd.read_excel(PATH_RAW_CHI_METADATA)
        df = df[~df[COL_ABSTRACT].isnull()]
        df.to_csv(PATH_CLEAN_CHI_METADATA, index=False)

    if PATH_EMBEDDINGS.is_file():
        print("embeddings file exists")
    else:
        # only import sbert if file does not exist
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SBERT_MODEL_NAME)
        embeddings = model.encode(df[COL_ABSTRACT].tolist(), show_progress_bar=True)
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings[COL_DOI] = df[COL_DOI]
        df_embeddings.to_csv(PATH_EMBEDDINGS, index=False)


if __name__ == "__main__":
    main()
