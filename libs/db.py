"""Database engine and session factory."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

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


def make_engine(database_url: str, debug: bool) -> AsyncEngine:
    """Create and return an async SQLAlchemy engine for *database_url*.

    The URL is normalized via :func:`normalize_database_url` so callers may
    pass plain ``postgresql://…`` URLs without needing to specify the asyncpg
    driver explicitly.
    """
    return create_async_engine(normalize_database_url(database_url), echo=debug)


engine = make_engine(settings.database_url, settings.debug)

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
