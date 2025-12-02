"""Pydantic schemas for request/response validation"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# Enums
class VerificationStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskCategoryEnum(str, Enum):
    VERIFIED = "verified"
    AUTHENTIC_HIGH_CONFIDENCE = "authentic_high_confidence"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    LIKELY_SYNTHETIC = "likely_synthetic"
    SYNTHETIC_HIGH_CONFIDENCE = "synthetic_high_confidence"
    FRAUDULENT = "fraudulent"


class ContentTypeEnum(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class PriorityEnum(str, Enum):
    STANDARD = "standard"
    EXPEDITED = "expedited"


class VerticalEnum(str, Enum):
    INSURANCE = "insurance"
    LEGAL = "legal"
    ECOMMERCE = "ecommerce"
    NEWS = "news"
    GENERAL = "general"


# Request Schemas
class VerificationOptions(BaseModel):
    """Options for verification request"""
    priority: PriorityEnum = Field(default=PriorityEnum.STANDARD, description="Processing priority")
    include_detailed_report: bool = Field(default=True, description="Include detailed evidence in response")
    vertical: Optional[VerticalEnum] = Field(default=None, description="Industry vertical for specialized analysis")
    force_full_pipeline: bool = Field(default=False, description="Force execution of all stages even if C2PA verified")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for context")


class VerificationRequest(BaseModel):
    """Request to verify content"""
    options: VerificationOptions = Field(default_factory=VerificationOptions)


class URLVerificationRequest(BaseModel):
    """Request to verify content from URL"""
    url: str = Field(..., description="URL of the content to verify")
    options: VerificationOptions = Field(default_factory=VerificationOptions)


class BatchVerificationRequest(BaseModel):
    """Request for batch verification"""
    options: VerificationOptions = Field(default_factory=VerificationOptions)


# Stage Result Schemas
class C2PAResult(BaseModel):
    """C2PA verification result"""
    stage: str = Field(default="c2pa_verification")
    status: str = Field(..., description="verified|invalid|not_present")
    details: Dict[str, Any] = Field(default_factory=dict)
    trust_contribution: float = Field(..., ge=0, le=100)
    evidence_weight: str = Field(..., description="very_high|high|medium|low")


class HardwareAuthResult(BaseModel):
    """Hardware authentication result"""
    stage: str = Field(default="hardware_authentication")
    status: str = Field(..., description="authenticated|untrusted|none")
    details: Dict[str, Any] = Field(default_factory=dict)
    trust_contribution: float = Field(..., ge=-100, le=100)
    evidence_weight: str = Field(..., description="very_high|high|medium|low")


class MetadataResult(BaseModel):
    """Metadata analysis result"""
    stage: str = Field(default="metadata_analysis")
    status: str = Field(..., description="consistent|suspicious|anomalous")
    details: Dict[str, Any] = Field(default_factory=dict)
    trust_contribution: float = Field(..., ge=-100, le=100)
    evidence_weight: str = Field(..., description="very_high|high|medium|low")


class ContextualResult(BaseModel):
    """Contextual verification result"""
    stage: str = Field(default="contextual_verification")
    status: str = Field(..., description="trusted|unknown|suspicious")
    details: Dict[str, Any] = Field(default_factory=dict)
    trust_contribution: float = Field(..., ge=-100, le=100)
    evidence_weight: str = Field(..., description="very_high|high|medium|low")


class MLResult(BaseModel):
    """ML detection result"""
    stage: str = Field(default="ml_detection")
    status: str = Field(..., description="ai_likely|uncertain|authentic_likely")
    details: Dict[str, Any] = Field(default_factory=dict)
    trust_contribution: float = Field(..., ge=-100, le=100)
    evidence_weight: str = Field(..., description="very_high|high|medium|low")


# Response Schemas
class VerificationStage(BaseModel):
    """Individual verification stage result"""
    stage: int = Field(..., ge=1, le=5)
    name: str
    status: str
    duration_ms: int
    contribution: float


class VisualEvidence(BaseModel):
    """Visual evidence URLs"""
    original_image_url: Optional[str] = None
    annotated_image_url: Optional[str] = None
    heatmap_url: Optional[str] = None


class VerificationResponse(BaseModel):
    """Complete verification response"""
    verification_id: str
    timestamp: datetime
    file_hash: str
    trust_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100)
    risk_category: RiskCategoryEnum
    processing_time_ms: int

    verification_stages: List[VerificationStage]
    key_findings: List[str]
    risk_factors: List[str]
    recommendation: str

    # Detailed stage results (optional)
    c2pa_result: Optional[C2PAResult] = None
    hardware_auth_result: Optional[HardwareAuthResult] = None
    metadata_result: Optional[MetadataResult] = None
    contextual_result: Optional[ContextualResult] = None
    ml_result: Optional[MLResult] = None

    visual_evidence: Optional[VisualEvidence] = None

    class Config:
        json_schema_extra = {
            "example": {
                "verification_id": "ver_abc123def456",
                "timestamp": "2025-12-02T15:30:00Z",
                "file_hash": "sha256:abc123...",
                "trust_score": 78,
                "confidence": 85,
                "risk_category": "likely_authentic",
                "processing_time_ms": 2847,
                "verification_stages": [
                    {
                        "stage": 1,
                        "name": "C2PA Verification",
                        "status": "not_present",
                        "duration_ms": 245,
                        "contribution": 0
                    }
                ],
                "key_findings": [
                    "Valid hardware signature from Canon EOS R5",
                    "Consistent metadata across all fields"
                ],
                "risk_factors": [],
                "recommendation": "Content appears authentic with high confidence"
            }
        }


class BatchVerificationStatus(BaseModel):
    """Batch verification status"""
    job_id: str
    status: VerificationStatusEnum
    total_items: int
    completed_items: int
    failed_items: int
    created_at: datetime
    updated_at: datetime
    results: Optional[List[VerificationResponse]] = None


class APIKeyCreate(BaseModel):
    """Request to create an API key"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    expires_in_days: Optional[int] = Field(default=365, ge=1, le=3650)
    rate_limit_per_minute: int = Field(default=60, ge=1)
    rate_limit_per_hour: int = Field(default=1000, ge=1)


class APIKeyResponse(BaseModel):
    """API key response"""
    id: str
    key: str  # Only returned on creation
    key_prefix: str
    name: str
    description: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]  # {service_name: status}


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None


class WebhookCreate(BaseModel):
    """Create webhook"""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(default=None, description="Webhook secret for signature verification")


class WebhookResponse(BaseModel):
    """Webhook response"""
    id: str
    url: str
    events: List[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int


class WebhookEvent(BaseModel):
    """Webhook event payload"""
    event: str
    timestamp: datetime
    verification_id: str
    trust_score: float
    category: RiskCategoryEnum
    data: VerificationResponse
