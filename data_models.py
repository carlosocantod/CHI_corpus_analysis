from functools import partial

from pandera import DataFrameModel
from pandera import Field
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base

from settings import CLUSTER_TOP_WORDS_TABLE_NAME
from settings import METADATA_TABLE_NAME

Base = declarative_base()

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


class ClusterPositions(DataFrameBaseModel):
    x: float = CheckNameField()
    y: float = CheckNameField()
    cluster: str = CheckNameField()
    doi: str = CheckNameField()


class MetadataWithCluster(Metadata, ClusterPositions):
    cluster: str = CheckNameField()


class MetadataWithScore(MetadataWithCluster):
    score: float = CheckNameField(le=-1.0, ge=1.0)


class Embeddings(_DOI):
    numbers: float = CheckNameField(regex=True, alias=r'\d{1,3}')


class TopWordsCluster(DataFrameBaseModel):
    cluster: str = CheckNameField(unique=True)
    top_words: str = CheckNameField()


class TopWordsPositionsCluster(TopWordsCluster):
    counts: int = CheckNameField()
    x: float = CheckNameField()
    y: float = CheckNameField()


class SparseEmbeddingsDataModel(_DOI):
    sparse_indices: str = CheckNameField()
    sparse_values: str = CheckNameField()


db_meta = MetaData()


class MetadataDB(Base):
    __tablename__ = METADATA_TABLE_NAME
    doi = Column(String, primary_key=True)
    title = Column(String)
    abstract = Column(String)
    cluster = Column(String)
    year = Column(Integer)
    x = Column(Float)
    y = Column(Float)
    scholar_link = Column(String)


class ClusterWordsDB(Base):
    __tablename__ = CLUSTER_TOP_WORDS_TABLE_NAME
    cluster = Column(String, primary_key=True)
    top_words = Column(String)
