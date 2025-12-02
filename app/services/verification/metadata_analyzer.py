"""Metadata analysis service"""
import time
from typing import Dict, Any, Optional
from datetime import datetime
import exifread
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from app.utils.logging import get_logger
from app.core.exceptions import MetadataExtractionError

logger = get_logger(__name__)


class MetadataAnalyzer:
    """Metadata extraction and analysis"""

    def __init__(self, file_path: str, content_type: str):
        self.file_path = file_path
        self.content_type = content_type

    async def analyze(self) -> Dict[str, Any]:
        """
        Extract and analyze metadata from file.

        Returns:
            Dictionary with analysis results:
            - stage: "metadata_analysis"
            - status: "consistent"|"suspicious"|"anomalous"
            - details: Dict with metadata
            - trust_contribution: float (-100 to 100)
            - evidence_weight: str
            - duration_ms: int
        """
        start_time = time.time()

        try:
            if self.content_type == "image":
                result = await self._analyze_image_metadata()
            else:
                # Video metadata analysis
                result = await self._analyze_video_metadata()

            duration_ms = int((time.time() - start_time) * 1000)
            result["duration_ms"] = duration_ms

            return result

        except Exception as e:
            logger.error("Metadata analysis failed", error=str(e))
            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "stage": "metadata_analysis",
                "status": "anomalous",
                "details": {
                    "exif_present": False,
                    "error": str(e),
                },
                "trust_contribution": -20,
                "evidence_weight": "medium",
                "duration_ms": duration_ms,
            }

    async def _analyze_image_metadata(self) -> Dict[str, Any]:
        """Analyze image metadata"""
        metadata = {}
        anomalies = []

        # Extract EXIF data using PIL
        try:
            with Image.open(self.file_path) as img:
                exif_data = img._getexif()

                if exif_data:
                    # Convert EXIF tags to readable names
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        metadata[tag_name] = str(value)

                    # Extract GPS data
                    gps_info = self._extract_gps_data(exif_data)
                    if gps_info:
                        metadata["gps"] = gps_info

        except Exception as e:
            logger.warning("Failed to extract EXIF with PIL", error=str(e))

        # Also try with exifread for more comprehensive extraction
        try:
            with open(self.file_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)

                for tag, value in tags.items():
                    if tag not in ('JPEGThumbnail', 'TIFFThumbnail'):
                        metadata[tag] = str(value)

        except Exception as e:
            logger.warning("Failed to extract EXIF with exifread", error=str(e))

        # Check if any metadata was found
        if not metadata:
            logger.info("No EXIF metadata found")
            return {
                "stage": "metadata_analysis",
                "status": "suspicious",
                "details": {
                    "exif_present": False,
                    "reason": "No metadata found (common in AI-generated or stripped images)",
                },
                "trust_contribution": -30,
                "evidence_weight": "medium",
            }

        # Analyze metadata for consistency
        consistency_checks = self._check_metadata_consistency(metadata)

        # Extract key information
        camera_model = metadata.get("Model") or metadata.get("Image Model")
        timestamp = metadata.get("DateTime") or metadata.get("EXIF DateTimeOriginal")
        software = metadata.get("Software") or metadata.get("Image Software")

        # Check for AI generation indicators
        ai_indicators = self._check_ai_indicators(metadata)

        if ai_indicators:
            anomalies.extend(ai_indicators)

        # Determine status
        if anomalies:
            status = "anomalous"
            trust_contribution = -40
        elif not consistency_checks["all_passed"]:
            status = "suspicious"
            trust_contribution = -20
        else:
            status = "consistent"
            trust_contribution = 70

        return {
            "stage": "metadata_analysis",
            "status": status,
            "details": {
                "exif_present": True,
                "exif_consistent": consistency_checks["all_passed"],
                "camera_model": camera_model,
                "timestamp": timestamp,
                "software_chain": [software] if software else [],
                "consistency_checks": consistency_checks,
                "camera_profile_match": "unknown",  # Would require database lookup
                "ai_watermarks": {
                    "synthid_detected": False,  # Requires SynthID SDK
                    "stable_signature_detected": False,
                    "visible_watermarks": [],
                },
                "anomalies": anomalies,
                "full_metadata_count": len(metadata),
            },
            "trust_contribution": trust_contribution,
            "evidence_weight": "medium",
        }

    async def _analyze_video_metadata(self) -> Dict[str, Any]:
        """Analyze video metadata"""
        # TODO: Implement video metadata analysis using ffmpeg or similar
        logger.info("Video metadata analysis not yet fully implemented")

        return {
            "stage": "metadata_analysis",
            "status": "uncertain",
            "details": {
                "metadata_present": False,
                "reason": "Video metadata analysis not yet implemented",
            },
            "trust_contribution": 0,
            "evidence_weight": "low",
        }

    def _extract_gps_data(self, exif_data: Dict) -> Optional[Dict[str, Any]]:
        """Extract GPS coordinates from EXIF"""
        try:
            gps_info = {}
            if 34853 in exif_data:  # GPSInfo tag
                gps_data = exif_data[34853]

                for tag_id, value in gps_data.items():
                    tag_name = GPSTAGS.get(tag_id, tag_id)
                    gps_info[tag_name] = value

                # Convert to decimal degrees if possible
                if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                    lat = self._convert_to_degrees(gps_info['GPSLatitude'])
                    lon = self._convert_to_degrees(gps_info['GPSLongitude'])

                    if 'GPSLatitudeRef' in gps_info and gps_info['GPSLatitudeRef'] == 'S':
                        lat = -lat
                    if 'GPSLongitudeRef' in gps_info and gps_info['GPSLongitudeRef'] == 'W':
                        lon = -lon

                    return {"latitude": lat, "longitude": lon}

            return None
        except Exception:
            return None

    def _convert_to_degrees(self, value) -> float:
        """Convert GPS coordinates to degrees"""
        try:
            d, m, s = value
            return float(d) + float(m) / 60.0 + float(s) / 3600.0
        except:
            return 0.0

    def _check_metadata_consistency(self, metadata: Dict[str, Any]) -> Dict[str, bool]:
        """Check metadata for consistency"""
        checks = {
            "timestamp_coherent": True,
            "settings_plausible": True,
            "gps_valid": True,
            "software_valid": True,
            "all_passed": True,
        }

        # Check timestamps
        try:
            datetime_str = metadata.get("DateTime") or metadata.get("EXIF DateTimeOriginal")
            if datetime_str:
                # Basic timestamp format check
                if not any(char.isdigit() for char in str(datetime_str)):
                    checks["timestamp_coherent"] = False
        except Exception:
            pass

        # Check GPS coordinates
        gps = metadata.get("gps")
        if gps:
            lat = gps.get("latitude", 0)
            lon = gps.get("longitude", 0)
            # Check for null island (0, 0) which is suspicious
            if lat == 0 and lon == 0:
                checks["gps_valid"] = False

        # Check for all passed
        checks["all_passed"] = all([
            checks["timestamp_coherent"],
            checks["settings_plausible"],
            checks["gps_valid"],
            checks["software_valid"],
        ])

        return checks

    def _check_ai_indicators(self, metadata: Dict[str, Any]) -> list:
        """Check for AI generation indicators in metadata"""
        indicators = []

        # Check software field for AI tools
        software = str(metadata.get("Software", "")).lower()
        ai_tools = [
            "midjourney", "dall-e", "dalle", "stable diffusion",
            "firefly", "photoshop beta", "generative fill"
        ]

        for tool in ai_tools:
            if tool in software:
                indicators.append(f"AI tool detected in metadata: {tool}")

        # Check for description/comment fields mentioning AI
        description = str(metadata.get("ImageDescription", "")).lower()
        comment = str(metadata.get("UserComment", "")).lower()

        ai_keywords = ["ai generated", "ai-generated", "artificial intelligence", "neural network"]
        for keyword in ai_keywords:
            if keyword in description or keyword in comment:
                indicators.append(f"AI keyword found in metadata: {keyword}")

        return indicators
