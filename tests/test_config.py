"""Tests for configuration"""
from app.config import Settings


def test_settings_initialization():
    """Test settings can be initialized"""
    settings = Settings()
    assert settings.app_name == "Verity"
    assert settings.app_version == "1.0.0"


def test_settings_properties():
    """Test settings computed properties"""
    settings = Settings()

    # Test allowed formats
    assert isinstance(settings.allowed_image_formats_list, list)
    assert isinstance(settings.allowed_video_formats_list, list)

    # Test max upload size
    assert settings.max_upload_size_bytes > 0
    assert settings.max_upload_size_bytes == settings.max_upload_size_mb * 1024 * 1024


def test_environment_detection():
    """Test environment detection"""
    settings = Settings(environment="production")
    assert settings.is_production is True
    assert settings.is_development is False

    settings = Settings(environment="development")
    assert settings.is_production is False
    assert settings.is_development is True
