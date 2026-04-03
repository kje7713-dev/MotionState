"""Database engine and session factory."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from libs.config import settings

_PLAIN_POSTGRES_PREFIX = "postgresql://"
_ASYNCPG_PREFIX = "postgresql+asyncpg://"


def normalize_database_url(url: str) -> str:
    """Rewrite a plain ``postgresql://`` URL to use the asyncpg driver.

    Railway and similar platforms expose ``DATABASE_URL`` as a plain
    ``postgresql://...`` connection string.  SQLAlchemy's async engine
    requires the ``postgresql+asyncpg://`` scheme; without it SQLAlchemy
    tries to import *psycopg2* and crashes at deploy time.

    Rules:
    - ``postgresql://...``         → ``postgresql+asyncpg://...``
    - ``postgresql+asyncpg://...`` → unchanged
    - anything else                → unchanged
    """
    if url.startswith(_PLAIN_POSTGRES_PREFIX) and not url.startswith(_ASYNCPG_PREFIX):
        return _ASYNCPG_PREFIX + url[len(_PLAIN_POSTGRES_PREFIX):]
    return url


engine = create_async_engine(normalize_database_url(settings.database_url), echo=settings.debug)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
