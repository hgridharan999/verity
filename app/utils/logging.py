"""Logging configuration using structlog"""
import logging
import sys
from typing import Any, Dict
import structlog
from app.config import settings


def configure_logging():
    """Configure structured logging"""

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.is_development
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


def log_verification_started(verification_id: str, content_type: str, file_size: int):
    """Log verification started event"""
    logger = get_logger("verification")
    logger.info(
        "verification_started",
        verification_id=verification_id,
        content_type=content_type,
        file_size=file_size,
    )


def log_verification_completed(
    verification_id: str,
    trust_score: float,
    risk_category: str,
    processing_time_ms: int
):
    """Log verification completed event"""
    logger = get_logger("verification")
    logger.info(
        "verification_completed",
        verification_id=verification_id,
        trust_score=trust_score,
        risk_category=risk_category,
        processing_time_ms=processing_time_ms,
    )


def log_verification_failed(verification_id: str, error: str, stage: str = None):
    """Log verification failed event"""
    logger = get_logger("verification")
    logger.error(
        "verification_failed",
        verification_id=verification_id,
        error=error,
        stage=stage,
    )


def log_stage_completed(
    verification_id: str,
    stage_name: str,
    status: str,
    duration_ms: int,
    contribution: float
):
    """Log verification stage completed"""
    logger = get_logger("verification.stage")
    logger.info(
        "stage_completed",
        verification_id=verification_id,
        stage=stage_name,
        status=status,
        duration_ms=duration_ms,
        contribution=contribution,
    )


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: int,
    user_id: str = None
):
    """Log API request"""
    logger = get_logger("api")
    logger.info(
        "api_request",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
        user_id=user_id,
    )


def log_security_event(
    event_type: str,
    severity: str,
    details: Dict[str, Any],
    user_id: str = None,
    ip_address: str = None
):
    """Log security event"""
    logger = get_logger("security")
    logger.warning(
        "security_event",
        event_type=event_type,
        severity=severity,
        details=details,
        user_id=user_id,
        ip_address=ip_address,
    )


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log error with context"""
    logger = get_logger("error")
    logger.error(
        "error_occurred",
        error=str(error),
        error_type=type(error).__name__,
        context=context or {},
        exc_info=True,
    )
