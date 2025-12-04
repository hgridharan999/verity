"""Celery background tasks"""
from app.celery_app import celery_app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@celery_app.task(name="app.services.tasks.cleanup_old_files")
def cleanup_old_files():
    """
    Background task to clean up old uploaded files.

    This task runs periodically to delete files that have exceeded
    their retention period.
    """
    logger.info("Starting cleanup of old files...")
    # TODO: Implement file cleanup logic
    logger.info("File cleanup completed")
    return {"status": "completed"}


@celery_app.task(name="app.services.tasks.process_verification")
def process_verification(verification_id: str):
    """
    Background task to process a verification request.

    Args:
        verification_id: The ID of the verification to process
    """
    logger.info(f"Processing verification: {verification_id}")
    # TODO: Implement verification processing logic
    logger.info(f"Verification {verification_id} completed")
    return {"verification_id": verification_id, "status": "completed"}
