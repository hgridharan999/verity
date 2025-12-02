"""File handling utilities"""
import os
import aiofiles
import filetype
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import hashlib
from app.config import settings
from app.core.exceptions import (
    UnsupportedFileTypeError,
    FileTooLargeError,
    FileProcessingError
)


async def save_upload_file(file_data: bytes, original_filename: str) -> Tuple[str, str]:
    """
    Save uploaded file to storage.

    Args:
        file_data: File bytes
        original_filename: Original filename

    Returns:
        Tuple of (storage_path, file_hash)
    """
    # Generate file hash
    file_hash = hashlib.sha256(file_data).hexdigest()

    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate storage path using hash to avoid collisions
    file_extension = Path(original_filename).suffix
    storage_filename = f"{file_hash}{file_extension}"
    storage_path = upload_dir / storage_filename

    # Save file
    async with aiofiles.open(storage_path, 'wb') as f:
        await f.write(file_data)

    return str(storage_path), f"sha256:{file_hash}"


async def delete_file(file_path: str) -> bool:
    """
    Delete file from storage.

    Args:
        file_path: Path to file

    Returns:
        True if deleted, False if file not found
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    except Exception as e:
        raise FileProcessingError(f"Failed to delete file: {str(e)}")


async def read_file(file_path: str) -> bytes:
    """
    Read file from storage.

    Args:
        file_path: Path to file

    Returns:
        File bytes
    """
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    except FileNotFoundError:
        raise FileProcessingError(f"File not found: {file_path}")
    except Exception as e:
        raise FileProcessingError(f"Failed to read file: {str(e)}")


def detect_file_type(file_data: bytes) -> Tuple[str, str]:
    """
    Detect file type from bytes.

    Args:
        file_data: File bytes

    Returns:
        Tuple of (content_type, file_format)

    Raises:
        UnsupportedFileTypeError: If file type is not supported
    """
    # Use filetype library for magic number detection
    kind = filetype.guess(file_data)

    if kind is None:
        raise UnsupportedFileTypeError(
            "unknown",
            settings.allowed_image_formats_list + settings.allowed_video_formats_list
        )

    mime_type = kind.mime
    extension = kind.extension.lower()

    # Determine content type
    if mime_type.startswith('image/'):
        content_type = "image"
        if extension not in settings.allowed_image_formats_list:
            raise UnsupportedFileTypeError(extension, settings.allowed_image_formats_list)
    elif mime_type.startswith('video/'):
        content_type = "video"
        if extension not in settings.allowed_video_formats_list:
            raise UnsupportedFileTypeError(extension, settings.allowed_video_formats_list)
    else:
        raise UnsupportedFileTypeError(
            extension,
            settings.allowed_image_formats_list + settings.allowed_video_formats_list
        )

    return content_type, extension


def validate_file_size(file_data: bytes) -> None:
    """
    Validate file size.

    Args:
        file_data: File bytes

    Raises:
        FileTooLargeError: If file exceeds maximum size
    """
    file_size = len(file_data)
    max_size = settings.max_upload_size_bytes

    if file_size > max_size:
        raise FileTooLargeError(file_size, max_size)


def get_image_dimensions(file_path: str) -> Tuple[int, int]:
    """
    Get image dimensions.

    Args:
        file_path: Path to image file

    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception as e:
        raise FileProcessingError(f"Failed to get image dimensions: {str(e)}")


def create_thumbnail(file_path: str, max_size: Tuple[int, int] = (200, 200)) -> Optional[str]:
    """
    Create thumbnail of image.

    Args:
        file_path: Path to source image
        max_size: Maximum thumbnail size (width, height)

    Returns:
        Path to thumbnail or None if failed
    """
    try:
        with Image.open(file_path) as img:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Generate thumbnail path
            path = Path(file_path)
            thumbnail_path = path.parent / f"{path.stem}_thumb{path.suffix}"

            img.save(thumbnail_path)
            return str(thumbnail_path)
    except Exception:
        return None


async def scan_for_malware(file_path: str) -> bool:
    """
    Scan file for malware.

    Note: This is a placeholder. In production, integrate with ClamAV or similar.

    Args:
        file_path: Path to file

    Returns:
        True if clean, False if malware detected
    """
    # TODO: Integrate with ClamAV
    # For now, just check file exists
    return Path(file_path).exists()


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure directory exists, create if not.

    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.

    Args:
        filename: Filename

    Returns:
        File extension (lowercase, without dot)
    """
    return Path(filename).suffix.lstrip('.').lower()


def is_image_file(filename: str) -> bool:
    """
    Check if filename appears to be an image.

    Args:
        filename: Filename

    Returns:
        True if image file extension
    """
    ext = get_file_extension(filename)
    return ext in settings.allowed_image_formats_list


def is_video_file(filename: str) -> bool:
    """
    Check if filename appears to be a video.

    Args:
        filename: Filename

    Returns:
        True if video file extension
    """
    ext = get_file_extension(filename)
    return ext in settings.allowed_video_formats_list
