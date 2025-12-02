"""C2PA signature verification service"""
import time
from typing import Dict, Any, Optional
from pathlib import Path
from app.utils.logging import get_logger
from app.core.exceptions import C2PAVerificationError

logger = get_logger(__name__)


class C2PAVerifier:
    """C2PA signature verification"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    async def verify(self) -> Dict[str, Any]:
        """
        Verify C2PA signature and extract provenance data.

        Returns:
            Dictionary with verification results:
            - stage: "c2pa_verification"
            - status: "verified"|"invalid"|"not_present"
            - details: Dict with manifest data
            - trust_contribution: float (0-100)
            - evidence_weight: str
            - duration_ms: int
        """
        start_time = time.time()

        try:
            # Try to import c2pa library
            try:
                import c2pa
            except ImportError:
                logger.warning("c2pa-python not installed, C2PA verification unavailable")
                return self._no_signature_result(start_time)

            # Read the manifest from the file
            try:
                reader = c2pa.Reader(self.file_path)
                manifest_store = reader.get_manifest_store()

                if not manifest_store:
                    logger.info("No C2PA manifest found", file_path=self.file_path)
                    return self._no_signature_result(start_time)

                # Get active manifest
                active_manifest = manifest_store.get_active_manifest()

                if not active_manifest:
                    logger.info("No active C2PA manifest found")
                    return self._no_signature_result(start_time)

                # Verify signature
                validation_status = manifest_store.validation_status()

                if validation_status and len(validation_status) == 0:
                    # Signature is valid
                    logger.info("C2PA signature verified successfully")

                    # Extract details
                    details = self._extract_manifest_details(active_manifest)

                    duration_ms = int((time.time() - start_time) * 1000)

                    return {
                        "stage": "c2pa_verification",
                        "status": "verified",
                        "details": details,
                        "trust_contribution": 95,
                        "evidence_weight": "very_high",
                        "duration_ms": duration_ms,
                    }
                else:
                    # Signature is invalid or tampered
                    logger.warning(
                        "C2PA signature verification failed",
                        validation_status=validation_status
                    )

                    duration_ms = int((time.time() - start_time) * 1000)

                    return {
                        "stage": "c2pa_verification",
                        "status": "invalid",
                        "details": {
                            "manifest_found": True,
                            "signature_valid": False,
                            "validation_errors": validation_status,
                        },
                        "trust_contribution": -95,
                        "evidence_weight": "very_high",
                        "duration_ms": duration_ms,
                    }

            except Exception as e:
                logger.error("Error reading C2PA manifest", error=str(e))
                return self._no_signature_result(start_time)

        except Exception as e:
            logger.error("C2PA verification failed", error=str(e))
            raise C2PAVerificationError(f"C2PA verification error: {str(e)}")

    def _no_signature_result(self, start_time: float) -> Dict[str, Any]:
        """Return result for no signature found"""
        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "stage": "c2pa_verification",
            "status": "not_present",
            "details": {
                "manifest_found": False,
                "signature_valid": False,
            },
            "trust_contribution": 0,
            "evidence_weight": "high",
            "duration_ms": duration_ms,
        }

    def _extract_manifest_details(self, manifest: Any) -> Dict[str, Any]:
        """Extract details from C2PA manifest"""
        try:
            details = {
                "manifest_found": True,
                "signature_valid": True,
                "certificate_trusted": True,
                "tamper_detected": False,
            }

            # Extract claim generator (software that created the file)
            try:
                claim = manifest.claim()
                if claim:
                    claim_generator = claim.get("claim_generator")
                    if claim_generator:
                        details["issuer"] = claim_generator

                    # Get timestamp
                    dc_created = claim.get("dc:created")
                    if dc_created:
                        details["timestamp"] = dc_created

            except Exception as e:
                logger.debug("Error extracting claim data", error=str(e))

            # Extract actions (edit history)
            try:
                actions = manifest.get("actions", [])
                details["actions"] = actions
            except Exception:
                details["actions"] = []

            # Extract assertions
            try:
                assertions = {}

                # Look for location data
                location = manifest.get("stds.exif", {}).get("GPS", {})
                if location:
                    assertions["location"] = location

                # Look for device/camera info
                exif = manifest.get("stds.exif", {})
                if exif.get("Make") or exif.get("Model"):
                    assertions["device"] = f"{exif.get('Make', '')} {exif.get('Model', '')}"

                details["assertions"] = assertions
            except Exception:
                details["assertions"] = {}

            return details

        except Exception as e:
            logger.error("Error extracting manifest details", error=str(e))
            return {
                "manifest_found": True,
                "signature_valid": True,
                "error": "Failed to extract details",
            }
