"""Content ingestion service"""
from typing import Dict, Any
from app.utils.file_handler import (
    save_upload_file,
    detect_file_type,
    validate_file_size,
    scan_for_malware,
)
from app.core.exceptions import FileProcessingError
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def ingest_file(
    file_content: bytes,
    filename: str,
    verification_id: str,
) -> Dict[str, Any]:
    """
    Ingest and validate uploaded file.

    Steps:
    1. Validate file size
    2. Detect file type
    3. Save file to storage
    4. Scan for malware
    5. Return file metadata

    Args:
        file_content: File bytes
        filename: Original filename
        verification_id: Verification ID

    Returns:
        Dictionary with file metadata:
        - file_hash: SHA-256 hash
        - file_size: Size in bytes
        - content_type: "image" or "video"
        - file_format: File extension
        - storage_path: Path where file is stored

    Raises:
        UnsupportedFileTypeError: If file type not supported
        FileTooLargeError: If file exceeds size limit
        FileProcessingError: If any processing step fails
    """
    logger.info(
        "Starting file ingestion",
        verification_id=verification_id,
        filename=filename,
        file_size=len(file_content)
    )

    # Step 1: Validate file size
    validate_file_size(file_content)
    logger.debug("File size validation passed", verification_id=verification_id)

    # Step 2: Detect file type using magic numbers
    content_type, file_format = detect_file_type(file_content)
    logger.info(
        "File type detected",
        verification_id=verification_id,
        content_type=content_type,
        file_format=file_format
    )

    # Step 3: Save file to storage
    storage_path, file_hash = await save_upload_file(file_content, filename)
    logger.info(
        "File saved to storage",
        verification_id=verification_id,
        storage_path=storage_path,
        file_hash=file_hash
    )

    # Step 4: Scan for malware
    is_clean = await scan_for_malware(storage_path)
    if not is_clean:
        # Delete the file
        from app.utils.file_handler import delete_file
        await delete_file(storage_path)
        raise FileProcessingError("Malware detected in uploaded file")

    logger.debug("Malware scan passed", verification_id=verification_id)

    # Return file metadata
    return {
        "file_hash": file_hash,
        "file_size": len(file_content),
        "content_type": content_type,
        "file_format": file_format,
        "storage_path": storage_path,
    }
