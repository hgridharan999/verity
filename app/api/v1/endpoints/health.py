"""Health check endpoints"""
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.models.schemas import HealthResponse
from app.config import settings
from app.db.session import get_db
import redis.asyncio as aioredis

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint.

    Returns system status and health of all services.
    """
    services = {}

    # Check database
    try:
        result = await db.execute(text("SELECT 1"))
        result.fetchone()
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        redis_client = aioredis.from_url(settings.redis_url)
        await redis_client.ping()
        await redis_client.close()
        services["redis"] = "healthy"
    except Exception as e:
        services["redis"] = f"unhealthy: {str(e)}"

    # Check file storage
    try:
        from pathlib import Path
        upload_dir = Path(settings.upload_dir)
        if upload_dir.exists() and upload_dir.is_dir():
            services["storage"] = "healthy"
        else:
            services["storage"] = "unhealthy: directory not found"
    except Exception as e:
        services["storage"] = f"unhealthy: {str(e)}"

    # Determine overall status
    overall_status = "healthy" if all("healthy" in s for s in services.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        timestamp=datetime.utcnow(),
        services=services
    )


@router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Readiness check for Kubernetes.

    Returns 200 if service is ready to accept traffic.
    """
    try:
        # Check database connectivity
        result = await db.execute(text("SELECT 1"))
        result.fetchone()
        return {"status": "ready"}
    except Exception:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check for Kubernetes.

    Returns 200 if service is alive.
    """
    return {"status": "alive"}
