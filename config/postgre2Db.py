from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.declarative import declarative_base
from .setting import env
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

metadata = MetaData()
Base = declarative_base(metadata=metadata)

SYNC_DB_URL = (
    f"postgresql+psycopg2://{env.db_user}:{env.db_password}@"
    f"{env.db_host}:5431/postgres"
)
sync_engine = create_engine(SYNC_DB_URL, echo=False)
