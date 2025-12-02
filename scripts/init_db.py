"""Database initialization script"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import init_db, engine
from app.models.database import Base
from app.config import settings
from app.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


async def create_tables():
    """Create all database tables"""
    logger.info("Creating database tables...")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created successfully")
        return True

    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        return False


async def create_directories():
    """Create necessary application directories"""
    logger.info("Creating application directories...")

    directories = [
        settings.upload_dir,
        settings.temp_dir,
        settings.ml_model_dir,
        settings.c2pa_trust_anchors_path,
        "logs",
    ]

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

    logger.info("Application directories created successfully")


async def main():
    """Main initialization function"""
    logger.info("Starting Verity initialization...")

    # Create directories
    await create_directories()

    # Create database tables
    success = await create_tables()

    if success:
        logger.info("Verity initialization completed successfully")
        return 0
    else:
        logger.error("Verity initialization failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
