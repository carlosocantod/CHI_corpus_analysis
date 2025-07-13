from itertools import batched

from fastembed import SparseEmbedding
from fastembed import SparseTextEmbedding
from numpy import ndarray
from pandera import check_types
from pandera.typing import DataFrame
from tqdm import tqdm

from data_models import Embeddings


@check_types()
def get_embeddings_from_dataframe(dataframe_embeddings: DataFrame[Embeddings]) -> ndarray:
    """

    :param dataframe_embeddings:
    :return:
    """
    return dataframe_embeddings.drop(columns=[Embeddings.doi]).to_numpy()


def obtain_sparse_vectors(
        model_sparse: SparseTextEmbedding,
        documents: list[str],
        batch_size: int = 32,
) -> list[SparseEmbedding]:
    """

    :param model_sparse:
    :param documents:
    :param batch_size:
    :return:
    """
    sparse_embeddings = []
    batch_documents = batched(documents, batch_size)
    with tqdm(total=len(documents), desc="Encoding") as pbar:
        for batch in batch_documents:
            batch_embeddings = list(model_sparse.embed(batch))
            sparse_embeddings.extend(batch_embeddings)
            pbar.update(len(batch))
    return sparse_embeddings
