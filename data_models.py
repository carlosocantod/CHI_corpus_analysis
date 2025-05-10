from pandera import DataFrameModel
from functools import partial
from pandera import Field

CheckNameField = partial(Field, check_name=True)
NullableField = partial(CheckNameField, nullable=True)
CheckListField = partial(CheckNameField, str_matches=r'\[.*\]')


class DataFrameBaseModel(DataFrameModel):

    class Config:
        """
        Default configuration class for dataframe models.

        Attributes:
            strict: make sure all specified columns are in the validated dataframe
            coerce: coerce types of all dataframe components
        """
        strict = True
        coerce = True


class _DOI(DataFrameBaseModel):
    doi: str = CheckNameField()


class Metadata(_DOI):
    title: str = CheckNameField()
    abstract: str = CheckNameField()
    doi: str = CheckNameField(unique=True)
    year: int = CheckNameField()
    scholar_link: str = CheckNameField()


class MetadataWithPositions(Metadata):
    x: float = CheckNameField()
    y: float = CheckNameField()


class MetadataWithScore(MetadataWithPositions):
    score: float = CheckNameField(le=-1.0, ge=1.0)


class Embeddings(_DOI):
    numbers: float = CheckNameField(regex=True, alias=r'\d{1,3}')
