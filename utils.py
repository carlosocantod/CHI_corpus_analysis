from pandera import check_types
from data_models import Embeddings
from pandera.typing import DataFrame
from numpy import ndarray


@check_types()
def get_embeddings_from_dataframe(dataframe_embeddings: DataFrame[Embeddings]) -> ndarray:
    """

    :param dataframe_embeddings:
    :return:
    """
    return dataframe_embeddings.drop(columns=[Embeddings.doi]).to_numpy()
