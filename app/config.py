"""Application configuration management using Pydantic Settings"""
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Application Settings
    app_name: str = Field(default="Verity", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # API Configuration
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    secret_key: str = Field(default="change-me-in-production", description="Secret key for encryption")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of workers")

    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://verity:verity@localhost:5432/verity",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=10, description="Database connection pool size")
    database_max_overflow: int = Field(default=20, description="Database max overflow connections")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_max_connections: int = Field(default=50, description="Redis max connections")

    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/1", description="Celery broker URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", description="Celery result backend")

    # File Upload Configuration
    max_upload_size_mb: int = Field(default=500, description="Max upload size in MB")
    allowed_image_formats: str = Field(
        default="jpg,jpeg,png,webp,tiff,heic",
        description="Allowed image formats (comma-separated)"
    )
    allowed_video_formats: str = Field(
        default="mp4,mov,avi,mkv,webm",
        description="Allowed video formats (comma-separated)"
    )
    upload_dir: str = Field(default="./uploads", description="Upload directory")
    temp_dir: str = Field(default="./temp", description="Temporary files directory")

    # Storage Configuration
    storage_backend: str = Field(default="local", description="Storage backend (local, s3, gcs)")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")
    aws_region: str = Field(default="us-east-1", description="AWS region")
    s3_bucket_name: Optional[str] = Field(default=None, description="S3 bucket name")
    gcp_project_id: Optional[str] = Field(default=None, description="GCP project ID")
    gcp_bucket_name: Optional[str] = Field(default=None, description="GCP bucket name")

    # Content Retention
    default_file_retention_hours: int = Field(default=24, description="Default file retention in hours")
    default_result_retention_days: int = Field(default=90, description="Default result retention in days")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    rate_limit_per_hour: int = Field(default=1000, description="Rate limit per hour")

    # C2PA Configuration
    c2pa_trust_anchors_path: str = Field(
        default="./config/trust_anchors",
        description="Path to C2PA trust anchors"
    )
    c2pa_verify_certificates: bool = Field(default=True, description="Verify C2PA certificates")

    # ML Models Configuration
    ml_model_dir: str = Field(default="./models", description="ML models directory")
    ml_device: str = Field(default="cpu", description="ML device (cpu, cuda)")
    ml_batch_size: int = Field(default=8, description="ML batch size")
    ml_num_workers: int = Field(default=4, description="ML number of workers")
    huggingface_cache_dir: str = Field(default="./cache/huggingface", description="HuggingFace cache dir")

    # External Services
    reverse_image_search_api_key: Optional[str] = Field(
        default=None,
        description="Reverse image search API key"
    )
    google_vision_api_key: Optional[str] = Field(default=None, description="Google Vision API key")

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus port")

    # Security
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: str = Field(default='["http://localhost:3000"]', description="CORS origins (JSON array)")
    api_key_expiry_days: int = Field(default=365, description="API key expiry days")
    enable_api_key_rotation: bool = Field(default=True, description="Enable API key rotation")

    # Webhooks
    webhook_timeout_seconds: int = Field(default=30, description="Webhook timeout in seconds")
    webhook_max_retries: int = Field(default=3, description="Webhook max retries")
    webhook_retry_delay_seconds: int = Field(default=60, description="Webhook retry delay in seconds")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from JSON string or list"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def allowed_image_formats_list(self) -> List[str]:
        """Get allowed image formats as list"""
        return [fmt.strip().lower() for fmt in self.allowed_image_formats.split(",")]

    @property
    def allowed_video_formats_list(self) -> List[str]:
        """Get allowed video formats as list"""
        return [fmt.strip().lower() for fmt in self.allowed_video_formats.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes"""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()
