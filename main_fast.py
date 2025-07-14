from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pandera import check_types
from pandera.typing import DataFrame
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import load_only

from data_models import ClusterPositions
from data_models import MetadataDB
from settings import COLLECTION_HYBRID_NAME
from settings import SQL_DB_PATH_LOCAL
from utils import HybridSearcher

app = FastAPI()


from fastapi import FastAPI

client_resources = {}
persistent_data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    client_resources["hybrid_searcher"] = HybridSearcher(collection_name=COLLECTION_HYBRID_NAME)

    # 2. Create a database engine (SQLite in this example)
    engine = create_engine(f'sqlite:///{SQL_DB_PATH_LOCAL}')  # Creates a file named metadata.db

    # 1. Create session
    session = Session(engine)
    client_resources["session_sql"] = session

    results = session.query(MetadataDB).options(load_only(MetadataDB.x, MetadataDB.y, MetadataDB.cluster)).all()

    dataframe_positions = pd.DataFrame(
        [d.__dict__ for d in results]).drop(columns="_sa_instance_state", errors="ignore")
    print(dataframe_positions)
    ClusterPositions.validate(dataframe_positions, inplace=True)
    persistent_data["cluster_positions"] = dataframe_positions

    yield
    # Clean up the ML models and release the resources
    session.close()
    client_resources.clear()

app = FastAPI(lifespan=lifespan)


@check_types()
@app.get("/cluster_positions")
async def get_cluster_positions() -> DataFrame[ClusterPositions]:
    # Use the model for prediction
    return persistent_data["cluster_positions"]



