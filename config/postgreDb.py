from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.declarative import declarative_base
from .setting import env
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# --- Required for Alembic autogenerate support (metadata, Base and SYNC_DB_URL) --- 
metadata = MetaData()
Base = declarative_base(metadata=metadata)
SYNC_DB_URL = (
    f"postgresql+psycopg2://{env.db_user}:{env.db_password}@"
    f"{env.db_host}:{env.db_port}/{env.db_name}"
)

ASYNC_DB_URL = (
    f"postgresql+asyncpg://{env.db_user}:{env.db_password}@"
    f"{env.db_host}:{env.db_port}/{env.db_name}"
)

sync_engine = create_engine(SYNC_DB_URL, echo=False)
async_engine = create_async_engine(ASYNC_DB_URL, echo=False)

sync_session = sessionmaker(sync_engine, autocommit=False, autoflush=False)
async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# --- Session Helper Functions ---
def get_sync_db():
    """
    Provides a synchronous database session.
    """
    db = sync_session()
    try:
        yield db
    finally:
        db.close()

async def get_async_db(): # Renamed from get_db for clarity
    """
    Provides an asynchronous database session.
    """
    async with async_session() as session:
        yield session

