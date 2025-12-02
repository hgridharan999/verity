"""Custom exceptions for Verity application"""
from typing import Any, Dict, Optional


class VerityException(Exception):
    """Base exception for all Verity errors"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(VerityException):
    """Raised when input validation fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class AuthenticationError(VerityException):
    """Raised when authentication fails"""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(VerityException):
    """Raised when authorization fails"""

    def __init__(self, message: str = "Not authorized", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=403, details=details)


class NotFoundError(VerityException):
    """Raised when a resource is not found"""

    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=404, details=details)


class RateLimitError(VerityException):
    """Raised when rate limit is exceeded"""

    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=429, details=details)


class FileProcessingError(VerityException):
    """Raised when file processing fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class UnsupportedFileTypeError(FileProcessingError):
    """Raised when file type is not supported"""

    def __init__(self, file_type: str, supported_types: list):
        super().__init__(
            f"Unsupported file type: {file_type}",
            details={"file_type": file_type, "supported_types": supported_types}
        )


class FileTooLargeError(FileProcessingError):
    """Raised when file is too large"""

    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            f"File size {file_size} bytes exceeds maximum {max_size} bytes",
            details={"file_size": file_size, "max_size": max_size}
        )


class VerificationError(VerityException):
    """Raised when verification process fails"""

    def __init__(self, message: str, stage: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if stage:
            details["stage"] = stage
        super().__init__(message, status_code=500, details=details)


class C2PAVerificationError(VerificationError):
    """Raised when C2PA verification fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, stage="c2pa_verification", details=details)


class MetadataExtractionError(VerificationError):
    """Raised when metadata extraction fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, stage="metadata_analysis", details=details)


class MLInferenceError(VerificationError):
    """Raised when ML inference fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, stage="ml_detection", details=details)


class ExternalServiceError(VerityException):
    """Raised when external service call fails"""

    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["service"] = service
        super().__init__(f"{service} error: {message}", status_code=502, details=details)


class DatabaseError(VerityException):
    """Raised when database operation fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Database error: {message}", status_code=500, details=details)


class ConfigurationError(VerityException):
    """Raised when configuration is invalid"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Configuration error: {message}", status_code=500, details=details)
