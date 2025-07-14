from sqlalchemy import create_engine

from data_models import Base
from settings import METADATA_TABLE_NAME
from settings import SQL_DB_PATH_LOCAL
from setup_streamlit import load_data


def main() -> None:
    engine = create_engine(f'sqlite:///{SQL_DB_PATH_LOCAL}')
    _, _, metadata, top_words_topics = load_data()
    # Create all tables at once
    Base.metadata.create_all(engine)
    metadata.to_sql(
        name=METADATA_TABLE_NAME,
        con=engine,
        if_exists='replace',
        index=False
    )


if __name__ == "__main__":
    main()
