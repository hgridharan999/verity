"""SQLAlchemy database models"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, JSON, Enum as SQLEnum, Index, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


class VerificationStatus(str, enum.Enum):
    """Verification status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskCategory(str, enum.Enum):
    """Risk category enum"""
    VERIFIED = "verified"
    AUTHENTIC_HIGH_CONFIDENCE = "authentic_high_confidence"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    LIKELY_SYNTHETIC = "likely_synthetic"
    SYNTHETIC_HIGH_CONFIDENCE = "synthetic_high_confidence"
    FRAUDULENT = "fraudulent"


class ContentType(str, enum.Enum):
    """Content type enum"""
    IMAGE = "image"
    VIDEO = "video"


class Verification(Base):
    """Main verification record"""
    __tablename__ = "verifications"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # File information
    file_hash = Column(String(128), nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    content_type = Column(SQLEnum(ContentType), nullable=False)
    file_format = Column(String(20), nullable=False)
    original_filename = Column(String(255))
    storage_path = Column(String(512))

    # User/API information
    api_key_id = Column(String(36), ForeignKey("api_keys.id"), nullable=False, index=True)
    user_id = Column(String(36), index=True)
    vertical = Column(String(50), index=True)  # insurance, legal, ecommerce, news

    # Processing information
    status = Column(SQLEnum(VerificationStatus), default=VerificationStatus.PENDING, nullable=False, index=True)
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    processing_time_ms = Column(Integer)

    # Verification results
    trust_score = Column(Float)
    confidence = Column(Float)
    risk_category = Column(SQLEnum(RiskCategory), index=True)

    # Stage results (stored as JSON)
    c2pa_result = Column(JSON)
    hardware_auth_result = Column(JSON)
    metadata_result = Column(JSON)
    contextual_result = Column(JSON)
    ml_result = Column(JSON)

    # Aggregated results
    key_findings = Column(JSON)
    risk_factors = Column(JSON)
    recommendation = Column(Text)

    # Metadata
    options = Column(JSON)  # Request options
    error_message = Column(Text)
    error_details = Column(JSON)

    # Retention
    expires_at = Column(DateTime, index=True)
    deleted_at = Column(DateTime, index=True)

    # Relationships
    api_key = relationship("APIKey", back_populates="verifications")
    audit_logs = relationship("AuditLog", back_populates="verification", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_user_created", "user_id", "created_at"),
        Index("idx_status_created", "status", "created_at"),
        Index("idx_risk_category_created", "risk_category", "created_at"),
    )


class APIKey(Base):
    """API key management"""
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Key information
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(10), nullable=False)  # First few chars for identification
    name = Column(String(100))
    description = Column(Text)

    # User/Organization
    user_id = Column(String(36), nullable=False, index=True)
    organization_id = Column(String(36), index=True)

    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime)
    revoked_reason = Column(Text)

    # Expiry
    expires_at = Column(DateTime, index=True)

    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=1000)
    rate_limit_per_day = Column(Integer, default=10000)

    # Usage tracking
    total_requests = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime)

    # Permissions
    allowed_verticals = Column(JSON)  # List of allowed verticals
    max_file_size_mb = Column(Integer, default=500)

    # Relationships
    verifications = relationship("Verification", back_populates="api_key")

    # Indexes
    __table_args__ = (
        Index("idx_user_active", "user_id", "is_active"),
    )


class CreatorReputation(Base):
    """Creator reputation tracking"""
    __tablename__ = "creator_reputations"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Identity
    creator_id = Column(String(255), unique=True, nullable=False, index=True)
    identity_verified = Column(Boolean, default=False, nullable=False)
    verification_method = Column(String(50))

    # Statistics
    total_submissions = Column(Integer, default=0, nullable=False)
    verified_authentic = Column(Integer, default=0, nullable=False)
    flagged_synthetic = Column(Integer, default=0, nullable=False)
    fraud_incidents = Column(Integer, default=0, nullable=False)

    # Reputation score
    reputation_score = Column(Float, default=50.0, nullable=False, index=True)
    confidence_score = Column(Float, default=0.0, nullable=False)

    # Domain expertise
    domain_expertise = Column(JSON)  # {vertical: score}
    certifications = Column(JSON)  # Professional certifications

    # Activity
    first_submission_date = Column(DateTime, index=True)
    last_submission_date = Column(DateTime, index=True)
    last_score_update = Column(DateTime)

    # Flags
    is_flagged = Column(Boolean, default=False, nullable=False, index=True)
    flag_reason = Column(Text)
    flagged_at = Column(DateTime)


class AuditLog(Base):
    """Audit log for all significant events"""
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Event information
    event_type = Column(String(50), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)
    event_description = Column(Text)

    # Related entities
    verification_id = Column(String(36), ForeignKey("verifications.id"), index=True)
    user_id = Column(String(36), index=True)
    api_key_id = Column(String(36), index=True)

    # Event details
    event_data = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(String(255))

    # Security
    is_security_event = Column(Boolean, default=False, nullable=False, index=True)
    severity = Column(String(20))  # info, warning, error, critical

    # Relationships
    verification = relationship("Verification", back_populates="audit_logs")

    # Indexes
    __table_args__ = (
        Index("idx_event_type_created", "event_type", "created_at"),
        Index("idx_security_created", "is_security_event", "created_at"),
    )


class MLModel(Base):
    """ML model registry"""
    __tablename__ = "ml_models"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Model information
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # spatial, frequency, etc.
    description = Column(Text)

    # Storage
    storage_path = Column(String(512), nullable=False)
    model_size_mb = Column(Float)
    checksum = Column(String(128))

    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)

    # Status
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    is_default = Column(Boolean, default=False, nullable=False)

    # Training information
    training_date = Column(DateTime)
    training_dataset_size = Column(Integer)
    training_config = Column(JSON)

    # Deployment
    deployed_at = Column(DateTime)
    deprecated_at = Column(DateTime)

    # Indexes
    __table_args__ = (
        Index("idx_name_version", "name", "version", unique=True),
        Index("idx_type_active", "model_type", "is_active"),
    )


class Webhook(Base):
    """Webhook configuration"""
    __tablename__ = "webhooks"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Owner
    user_id = Column(String(36), nullable=False, index=True)
    api_key_id = Column(String(36), index=True)

    # Configuration
    url = Column(String(512), nullable=False)
    secret = Column(String(255), nullable=False)
    events = Column(JSON, nullable=False)  # List of event types to subscribe to

    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Statistics
    total_deliveries = Column(Integer, default=0, nullable=False)
    successful_deliveries = Column(Integer, default=0, nullable=False)
    failed_deliveries = Column(Integer, default=0, nullable=False)
    last_delivery_at = Column(DateTime)
    last_success_at = Column(DateTime)
    last_failure_at = Column(DateTime)
    consecutive_failures = Column(Integer, default=0, nullable=False)

    # Automatic disabling
    max_consecutive_failures = Column(Integer, default=10)
    disabled_at = Column(DateTime)
    disabled_reason = Column(Text)


class WebhookDelivery(Base):
    """Webhook delivery log"""
    __tablename__ = "webhook_deliveries"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Webhook information
    webhook_id = Column(String(36), ForeignKey("webhooks.id"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    verification_id = Column(String(36), index=True)

    # Delivery information
    payload = Column(JSON, nullable=False)
    response_status_code = Column(Integer)
    response_body = Column(Text)
    delivery_duration_ms = Column(Integer)

    # Status
    is_successful = Column(Boolean, nullable=False, index=True)
    error_message = Column(Text)

    # Retry information
    attempt_number = Column(Integer, default=1, nullable=False)
    will_retry = Column(Boolean, default=False, nullable=False)
    next_retry_at = Column(DateTime)

    # Indexes
    __table_args__ = (
        Index("idx_webhook_created", "webhook_id", "created_at"),
        Index("idx_verification_created", "verification_id", "created_at"),
    )
