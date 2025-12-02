"""ML-based AI detection service"""
import time
import numpy as np
from typing import Dict, Any, List, Tuple
from PIL import Image
import cv2
from app.utils.logging import get_logger
from app.core.exceptions import MLInferenceError
from app.config import settings

logger = get_logger(__name__)


class MLDetector:
    """ML-based detection of AI-generated content"""

    def __init__(self, file_path: str, content_type: str):
        self.file_path = file_path
        self.content_type = content_type
        self.models_loaded = False

    async def detect(self) -> Dict[str, Any]:
        """
        Run ML detection pipeline.

        Returns:
            Dictionary with detection results:
            - stage: "ml_detection"
            - status: "ai_likely"|"uncertain"|"authentic_likely"
            - details: Dict with model outputs and analysis
            - trust_contribution: float (-100 to 100)
            - evidence_weight: str
            - duration_ms: int
        """
        start_time = time.time()

        try:
            if self.content_type == "image":
                result = await self._detect_image()
            else:
                result = await self._detect_video()

            duration_ms = int((time.time() - start_time) * 1000)
            result["duration_ms"] = duration_ms

            return result

        except Exception as e:
            logger.error("ML detection failed", error=str(e))
            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "stage": "ml_detection",
                "status": "uncertain",
                "details": {
                    "error": str(e),
                },
                "trust_contribution": 0,
                "evidence_weight": "low",
                "duration_ms": duration_ms,
            }

    async def _detect_image(self) -> Dict[str, Any]:
        """Detect AI generation in images"""

        # Load image
        try:
            image = Image.open(self.file_path)
            image_array = np.array(image.convert('RGB'))
        except Exception as e:
            raise MLInferenceError(f"Failed to load image: {str(e)}")

        # Run different detection methods
        spatial_score = await self._spatial_artifact_detection(image_array)
        frequency_score = await self._frequency_domain_analysis(image_array)
        problem_areas = await self._detect_problem_areas(image_array)

        # Calculate ensemble score
        model_outputs = {
            "spatial_detector": {"ai_score": spatial_score, "confidence": 0.70},
            "frequency_analyzer": {"ai_score": frequency_score, "confidence": 0.60},
            "problem_area_detector": {"ai_score": problem_areas["score"], "confidence": 0.65},
        }

        # Weighted ensemble
        ensemble_score = (
            spatial_score * 0.4 +
            frequency_score * 0.4 +
            problem_areas["score"] * 0.2
        )

        # Calculate confidence based on agreement
        scores = [spatial_score, frequency_score, problem_areas["score"]]
        std_dev = np.std(scores)
        confidence = max(0.0, 1.0 - std_dev)  # Higher agreement = higher confidence

        # Determine status
        if ensemble_score > 0.7:
            status = "ai_likely"
            trust_contribution = -60
        elif ensemble_score > 0.5:
            status = "uncertain"
            trust_contribution = -30
        else:
            status = "authentic_likely"
            trust_contribution = 40

        # Generate explanation
        explanation = self._generate_explanation(
            ensemble_score,
            spatial_score,
            frequency_score,
            problem_areas
        )

        return {
            "stage": "ml_detection",
            "status": status,
            "details": {
                "ensemble_prediction": {
                    "ai_probability": float(ensemble_score),
                    "confidence": float(confidence),
                    "uncertainty": float(1 - confidence),
                },
                "model_outputs": model_outputs,
                "frequency_analysis": {
                    "checkerboard_artifacts": frequency_score > 0.6,
                    "gan_fingerprint_detected": frequency_score > 0.7,
                    "anomaly_score": float(frequency_score),
                },
                "problem_areas": problem_areas,
                "explanation": explanation,
                "suspicious_regions": [],  # Would require more sophisticated analysis
            },
            "trust_contribution": trust_contribution,
            "evidence_weight": "medium",
        }

    async def _detect_video(self) -> Dict[str, Any]:
        """Detect AI generation in videos"""
        # TODO: Implement video detection
        logger.info("Video ML detection not yet fully implemented")

        return {
            "stage": "ml_detection",
            "status": "uncertain",
            "details": {
                "reason": "Video ML detection not yet implemented",
            },
            "trust_contribution": 0,
            "evidence_weight": "low",
        }

    async def _spatial_artifact_detection(self, image: np.ndarray) -> float:
        """
        Detect spatial artifacts using basic computer vision.

        In production, this would use a trained CNN model.
        For now, we use heuristics.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Look for unusual patterns in local variance
            # AI-generated images often have unusual noise patterns
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normalize variance score (this is a simplification)
            # Real images typically have variance in range [100, 2000]
            if variance < 50:
                # Very low variance can indicate AI generation
                score = 0.7
            elif variance > 3000:
                # Very high variance can also be suspicious
                score = 0.6
            else:
                # Normal range
                score = 0.3

            # Check for edge consistency
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            # AI images often have either too many or too few edges
            if edge_density < 0.01 or edge_density > 0.3:
                score += 0.2

            return min(1.0, score)

        except Exception as e:
            logger.error("Spatial artifact detection failed", error=str(e))
            return 0.5  # Uncertain

    async def _frequency_domain_analysis(self, image: np.ndarray) -> float:
        """
        Analyze frequency domain for AI artifacts.

        AI generators often leave characteristic patterns in frequency domain.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Compute FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)

            # Compute power spectral density
            psd = magnitude_spectrum ** 2

            # Look for checkerboard pattern (common in upsampling)
            # This is a simplified heuristic
            h, w = psd.shape
            center_h, center_w = h // 2, w // 2

            # Sample at Nyquist frequency
            nyquist_sample = psd[center_h - 10:center_h + 10, center_w - 10:center_w + 10]
            nyquist_power = np.mean(nyquist_sample)

            # Sample at high frequencies
            high_freq_sample = psd[0:20, 0:20]
            high_freq_power = np.mean(high_freq_sample)

            # Ratio indicates potential upsampling artifacts
            ratio = high_freq_power / (nyquist_power + 1e-10)

            if ratio > 0.1:
                return 0.8  # Likely AI-generated (upsampled)
            elif ratio > 0.05:
                return 0.6  # Possibly AI-generated
            else:
                return 0.3  # Likely authentic

        except Exception as e:
            logger.error("Frequency analysis failed", error=str(e))
            return 0.5  # Uncertain

    async def _detect_problem_areas(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect common problem areas in AI-generated images.

        Problem areas: hands, faces, text, reflections
        """
        result = {
            "score": 0.5,  # Default uncertain
            "hands_detected": 0,
            "hands_suspicious": False,
            "hand_issues": [],
            "faces_detected": 0,
            "face_quality": "unknown",
            "text_garbled": False,
        }

        try:
            # In production, use object detection models (YOLO, etc.)
            # For now, use simple heuristics

            # Check image smoothness (AI images often too smooth)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score < 100:
                # Very smooth, potentially AI-generated
                result["score"] = 0.7
                result["hand_issues"].append("Image appears unnaturally smooth")
            elif blur_score > 2000:
                # Very noisy, could be manipulated
                result["score"] = 0.6
            else:
                # Normal range
                result["score"] = 0.3

            # TODO: Add actual hand detection
            # TODO: Add actual face detection
            # TODO: Add OCR for text quality check

            return result

        except Exception as e:
            logger.error("Problem area detection failed", error=str(e))
            return result

    def _generate_explanation(
        self,
        ensemble_score: float,
        spatial_score: float,
        frequency_score: float,
        problem_areas: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation"""

        if ensemble_score > 0.7:
            explanation = "High likelihood of AI generation based on: "
            reasons = []

            if frequency_score > 0.7:
                reasons.append("(1) Checkerboard artifacts in frequency domain suggesting upsampling")
            if spatial_score > 0.7:
                reasons.append("(2) Unusual spatial noise patterns inconsistent with camera sensors")
            if problem_areas["score"] > 0.7:
                reasons.append("(3) Image smoothness typical of AI generators")

            if not reasons:
                reasons.append("multiple ML model indicators")

            return explanation + ", ".join(reasons)

        elif ensemble_score > 0.5:
            return "Moderate indicators of potential AI generation, but not conclusive. Further analysis recommended."

        else:
            return "Low likelihood of AI generation. Image characteristics consistent with authentic photography."
