"""Security utilities for authentication and authorization"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Algorithm for JWT
ALGORITHM = "HS256"


def generate_api_key() -> str:
    """Generate a secure random API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage"""
    return pwd_context.hash(api_key)


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash"""
    return pwd_context.verify(plain_key, hashed_key)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.api_key_expiry_days)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT access token"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def generate_verification_id() -> str:
    """Generate a unique verification ID"""
    random_bytes = secrets.token_bytes(16)
    hash_obj = hashlib.sha256(random_bytes)
    return f"ver_{hash_obj.hexdigest()[:24]}"


def generate_file_hash(file_bytes: bytes) -> str:
    """Generate SHA-256 hash of file contents"""
    hash_obj = hashlib.sha256(file_bytes)
    return f"sha256:{hash_obj.hexdigest()}"


def verify_webhook_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify webhook signature"""
    expected_signature = hashlib.sha256(
        f"{payload}{secret}".encode()
    ).hexdigest()
    return secrets.compare_digest(signature, expected_signature)


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self):
        self.requests = {}

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed based on rate limit"""
        now = datetime.utcnow()

        if key not in self.requests:
            self.requests[key] = []

        # Remove expired requests
        cutoff = now - timedelta(seconds=window_seconds)
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > cutoff
        ]

        # Check if limit exceeded
        if len(self.requests[key]) >= max_requests:
            return False

        # Add current request
        self.requests[key].append(now)
        return True

    def get_remaining(self, key: str, max_requests: int) -> int:
        """Get remaining requests for key"""
        if key not in self.requests:
            return max_requests
        return max(0, max_requests - len(self.requests[key]))

    def get_reset_time(self, key: str, window_seconds: int) -> Optional[datetime]:
        """Get time when rate limit resets"""
        if key not in self.requests or not self.requests[key]:
            return None
        oldest_request = min(self.requests[key])
        return oldest_request + timedelta(seconds=window_seconds)


# Global rate limiter instance
rate_limiter = RateLimiter()
