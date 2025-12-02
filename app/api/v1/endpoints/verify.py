"""Verification endpoints"""
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Depends, BackgroundTasks, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.models.schemas import (
    VerificationRequest,
    VerificationResponse,
    VerificationOptions,
    URLVerificationRequest,
)
from app.db.session import get_db
from app.services.ingestion import ingest_file
from app.services.verification_pipeline import VerificationPipeline
from app.core.security import generate_verification_id
from app.core.exceptions import ValidationError, NotFoundError
from app.utils.logging import get_logger, log_verification_started
from app.models.database import Verification, VerificationStatus

router = APIRouter()
logger = get_logger(__name__)


@router.post("/verify", response_model=VerificationResponse, status_code=status.HTTP_200_OK)
async def verify_content(
    file: UploadFile = File(...),
    priority: str = "standard",
    include_detailed_report: bool = True,
    vertical: Optional[str] = None,
    force_full_pipeline: bool = False,
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = None,
):
    """
    Verify uploaded content for authenticity.

    This endpoint accepts an image or video file and runs it through
    the multi-stage verification pipeline to determine if it's authentic
    or AI-generated.

    Args:
        file: The file to verify (image or video)
        priority: Processing priority (standard|expedited)
        include_detailed_report: Include detailed evidence breakdown
        vertical: Industry vertical (insurance|legal|ecommerce|news)
        force_full_pipeline: Force all stages even if C2PA verified
        db: Database session
        background_tasks: Background tasks

    Returns:
        VerificationResponse with trust score and detailed analysis
    """
    try:
        # Generate verification ID
        verification_id = generate_verification_id()

        # Read file content
        file_content = await file.read()

        # Create verification options
        options = VerificationOptions(
            priority=priority,
            include_detailed_report=include_detailed_report,
            vertical=vertical,
            force_full_pipeline=force_full_pipeline,
        )

        # Log verification started
        log_verification_started(
            verification_id=verification_id,
            content_type="unknown",  # Will be determined during ingestion
            file_size=len(file_content)
        )

        # Ingest file
        ingestion_result = await ingest_file(
            file_content=file_content,
            filename=file.filename,
            verification_id=verification_id,
        )

        # Create verification record in database
        verification = Verification(
            id=verification_id,
            file_hash=ingestion_result["file_hash"],
            file_size=ingestion_result["file_size"],
            content_type=ingestion_result["content_type"],
            file_format=ingestion_result["file_format"],
            original_filename=file.filename,
            storage_path=ingestion_result["storage_path"],
            api_key_id="default",  # TODO: Get from authentication
            vertical=vertical,
            status=VerificationStatus.PROCESSING,
            processing_started_at=datetime.utcnow(),
            options=options.model_dump(),
        )

        db.add(verification)
        await db.commit()
        await db.refresh(verification)

        # Run verification pipeline
        pipeline = VerificationPipeline(
            verification_id=verification_id,
            file_path=ingestion_result["storage_path"],
            content_type=ingestion_result["content_type"],
            file_format=ingestion_result["file_format"],
            options=options,
            db=db,
        )

        result = await pipeline.run()

        return result

    except Exception as e:
        logger.error("Verification failed", error=str(e), verification_id=verification_id)
        raise


@router.get("/verify/{verification_id}", response_model=VerificationResponse)
async def get_verification_result(
    verification_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get verification result by ID.

    Args:
        verification_id: The verification ID
        db: Database session

    Returns:
        VerificationResponse with verification results
    """
    from sqlalchemy import select

    # Query verification from database
    result = await db.execute(
        select(Verification).where(Verification.id == verification_id)
    )
    verification = result.scalar_one_or_none()

    if not verification:
        raise NotFoundError(f"Verification not found: {verification_id}")

    # Convert database record to response
    from app.services.verification_pipeline import VerificationPipeline
    response = VerificationPipeline.db_to_response(verification)

    return response


@router.delete("/verify/{verification_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_verification(
    verification_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete verification result (GDPR compliance).

    Args:
        verification_id: The verification ID
        db: Database session
    """
    from sqlalchemy import select, update

    # Check if verification exists
    result = await db.execute(
        select(Verification).where(Verification.id == verification_id)
    )
    verification = result.scalar_one_or_none()

    if not verification:
        raise NotFoundError(f"Verification not found: {verification_id}")

    # Soft delete
    await db.execute(
        update(Verification)
        .where(Verification.id == verification_id)
        .values(deleted_at=datetime.utcnow())
    )

    # Delete file from storage
    if verification.storage_path:
        from app.utils.file_handler import delete_file
        await delete_file(verification.storage_path)

    await db.commit()

    return None


@router.post("/verify/url", response_model=VerificationResponse)
async def verify_from_url(
    request: URLVerificationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Verify content from URL.

    Downloads content from the provided URL and verifies it.

    Args:
        request: URL verification request
        db: Database session

    Returns:
        VerificationResponse with verification results
    """
    # TODO: Implement URL fetching and verification
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="URL verification not yet implemented"
    )


@router.post("/verify/batch")
async def verify_batch(
    db: AsyncSession = Depends(get_db),
):
    """
    Batch verification endpoint.

    Accepts multiple files for verification and returns a job ID.

    Args:
        db: Database session

    Returns:
        Batch job status
    """
    # TODO: Implement batch verification
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Batch verification not yet implemented"
    )
