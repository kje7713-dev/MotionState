"""Database engine and session factory."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from libs.config import settings


def normalize_database_url(url: str) -> str:
    """Normalize a Postgres URL for async SQLAlchemy (asyncpg driver).

    Railway and similar platforms expose plain ``postgresql://…`` URLs.
    SQLAlchemy's async engine requires the ``postgresql+asyncpg://`` scheme.
    This helper rewrites the scheme when needed so callers never have to
    remember to do it manually.

    Rules:
    - ``postgresql://…``            → ``postgresql+asyncpg://…``
    - ``postgresql+asyncpg://…``    → unchanged
    - anything else                 → unchanged
    """
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://"):]
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
