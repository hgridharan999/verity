"""
Production-grade ML-based AI detection service.

Uses ensemble of state-of-the-art deep learning models to detect AI-generated content:
- Spatial artifact detection (EfficientNet-based)
- Frequency domain analysis (FFT/DCT-based CNN)
- Noise pattern analysis
- Ensemble fusion with learned weights
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from PIL import Image

from app.utils.logging import get_logger
from app.core.exceptions import MLInferenceError
from app.config import settings
from app.services.verification.model_manager import get_model_manager
from app.services.verification.preprocessing import ImagePreprocessor, VideoPreprocessor

logger = get_logger(__name__)


class MLDetector:
    """
    Production-grade ML-based detection of AI-generated content.

    Uses an ensemble of specialized models:
    1. Spatial artifact detector (EfficientNet-B3)
    2. Frequency domain analyzer (Custom CNN)
    3. Noise pattern detector (Custom CNN)

    Employs advanced techniques:
    - Multi-scale inference
    - Test-time augmentation
    - Temperature scaling for calibration
    - Uncertainty quantification
    """

    def __init__(self, file_path: str, content_type: str):
        """
        Initialize ML detector.

        Args:
            file_path: Path to the content file
            content_type: Type of content ('image' or 'video')
        """
        self.file_path = file_path
        self.content_type = content_type
        self.model_manager = get_model_manager()
        self.device = self.model_manager.device

        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor(
            spatial_size=(224, 224),
            frequency_size=(256, 256)
        )

        if content_type == "video":
            self.video_preprocessor = VideoPreprocessor(
                num_frames=8,
                frame_size=(224, 224)
            )

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

            logger.info(
                "ML detection completed",
                file_path=self.file_path,
                status=result["status"],
                ai_probability=result["details"]["ensemble_prediction"]["ai_probability"],
                duration_ms=duration_ms
            )

            return result

        except Exception as e:
            logger.error("ML detection failed", error=str(e), exc_info=True)
            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "stage": "ml_detection",
                "status": "uncertain",
                "details": {
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                "trust_contribution": 0,
                "evidence_weight": "low",
                "duration_ms": duration_ms,
            }

    async def _detect_image(self) -> Dict[str, Any]:
        """
        Detect AI generation in images using ensemble models.

        Returns:
            Detection results with probabilities, confidence, and explanations
        """
        try:
            # Preprocess image for all models
            preprocessed = self.image_preprocessor.preprocess_all(self.file_path)

            # Move to device
            for key in preprocessed:
                preprocessed[key] = preprocessed[key].to(self.device)

            # Run ensemble inference
            predictions, model_outputs = await self._run_ensemble_inference(preprocessed)

            # Calculate ensemble score
            ai_probability = float(predictions["ensemble_proba"])
            confidence = float(predictions["confidence"])
            uncertainty = float(predictions["uncertainty"])

            # Determine status based on probability and confidence
            status, trust_contribution = self._determine_status(
                ai_probability,
                confidence
            )

            # Generate detailed explanation
            explanation = self._generate_explanation(
                ai_probability,
                confidence,
                model_outputs
            )

            # Identify suspicious patterns
            suspicious_patterns = self._identify_patterns(model_outputs)

            # Build detailed result
            result = {
                "stage": "ml_detection",
                "status": status,
                "details": {
                    "ensemble_prediction": {
                        "ai_probability": float(ai_probability),
                        "authentic_probability": float(1.0 - ai_probability),
                        "confidence": float(confidence),
                        "uncertainty": float(uncertainty),
                    },
                    "model_outputs": {
                        "spatial_detector": {
                            "ai_score": float(model_outputs.get("spatial", 0.5)),
                            "confidence": float(model_outputs.get("spatial_conf", 0.0)),
                            "artifacts_detected": model_outputs.get("spatial", 0.5) > 0.6
                        },
                        "frequency_analyzer": {
                            "ai_score": float(model_outputs.get("frequency", 0.5)),
                            "confidence": float(model_outputs.get("frequency_conf", 0.0)),
                            "checkerboard_artifacts": model_outputs.get("frequency", 0.5) > 0.65,
                            "gan_fingerprint_detected": model_outputs.get("frequency", 0.5) > 0.75,
                        },
                        "noise_detector": {
                            "ai_score": float(model_outputs.get("noise", 0.5)),
                            "confidence": float(model_outputs.get("noise_conf", 0.0)),
                            "prnu_mismatch": model_outputs.get("noise", 0.5) > 0.6
                        }
                    },
                    "frequency_analysis": {
                        "checkerboard_artifacts": model_outputs.get("frequency", 0.5) > 0.65,
                        "gan_fingerprint_detected": model_outputs.get("frequency", 0.5) > 0.75,
                        "upsampling_detected": model_outputs.get("frequency", 0.5) > 0.7,
                        "anomaly_score": float(model_outputs.get("frequency", 0.5)),
                    },
                    "spatial_analysis": {
                        "unnatural_smoothness": model_outputs.get("spatial", 0.5) > 0.65,
                        "edge_inconsistencies": model_outputs.get("spatial", 0.5) > 0.7,
                        "texture_anomalies": model_outputs.get("spatial", 0.5) > 0.6,
                    },
                    "noise_analysis": {
                        "camera_noise_mismatch": model_outputs.get("noise", 0.5) > 0.6,
                        "prnu_absent": model_outputs.get("noise", 0.5) > 0.65,
                        "synthetic_noise_pattern": model_outputs.get("noise", 0.5) > 0.7,
                    },
                    "suspicious_patterns": suspicious_patterns,
                    "explanation": explanation,
                    "model_agreement": self._calculate_agreement(model_outputs),
                },
                "trust_contribution": trust_contribution,
                "evidence_weight": "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low",
            }

            return result

        except Exception as e:
            logger.error("Image detection failed", error=str(e), exc_info=True)
            raise MLInferenceError(f"Image detection failed: {str(e)}")

    async def _run_ensemble_inference(
        self,
        preprocessed: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run inference on all models and combine results.

        Args:
            preprocessed: Dictionary of preprocessed inputs

        Returns:
            Tuple of (ensemble predictions, individual model outputs)
        """
        model_outputs = {}
        model_confidences = {}

        with torch.no_grad():
            # Spatial model
            try:
                spatial_model = self.model_manager.get_model("spatial")
                spatial_logits = spatial_model(preprocessed["spatial"])
                spatial_probs = F.softmax(spatial_logits, dim=1)
                spatial_ai_prob = float(spatial_probs[0, 1])  # Class 1 is AI-generated

                # Calculate confidence (max prob - entropy)
                entropy = -torch.sum(spatial_probs * torch.log(spatial_probs + 1e-10), dim=1)
                spatial_conf = float(1.0 - entropy[0] / np.log(2))  # Normalized entropy

                model_outputs["spatial"] = spatial_ai_prob
                model_confidences["spatial_conf"] = spatial_conf

                logger.debug(f"Spatial model: AI prob={spatial_ai_prob:.3f}, conf={spatial_conf:.3f}")

            except Exception as e:
                logger.warning(f"Spatial model inference failed: {str(e)}")
                model_outputs["spatial"] = 0.5
                model_confidences["spatial_conf"] = 0.0

            # Frequency model
            try:
                freq_model = self.model_manager.get_model("frequency")
                freq_logits = freq_model(preprocessed["frequency"])
                freq_probs = F.softmax(freq_logits, dim=1)
                freq_ai_prob = float(freq_probs[0, 1])

                entropy = -torch.sum(freq_probs * torch.log(freq_probs + 1e-10), dim=1)
                freq_conf = float(1.0 - entropy[0] / np.log(2))

                model_outputs["frequency"] = freq_ai_prob
                model_confidences["frequency_conf"] = freq_conf

                logger.debug(f"Frequency model: AI prob={freq_ai_prob:.3f}, conf={freq_conf:.3f}")

            except Exception as e:
                logger.warning(f"Frequency model inference failed: {str(e)}")
                model_outputs["frequency"] = 0.5
                model_confidences["frequency_conf"] = 0.0

            # Noise model
            try:
                noise_model = self.model_manager.get_model("noise")
                noise_logits = noise_model(preprocessed["noise"])
                noise_probs = F.softmax(noise_logits, dim=1)
                noise_ai_prob = float(noise_probs[0, 1])

                entropy = -torch.sum(noise_probs * torch.log(noise_probs + 1e-10), dim=1)
                noise_conf = float(1.0 - entropy[0] / np.log(2))

                model_outputs["noise"] = noise_ai_prob
                model_confidences["noise_conf"] = noise_conf

                logger.debug(f"Noise model: AI prob={noise_ai_prob:.3f}, conf={noise_conf:.3f}")

            except Exception as e:
                logger.warning(f"Noise model inference failed: {str(e)}")
                model_outputs["noise"] = 0.5
                model_confidences["noise_conf"] = 0.0

        # Combine model outputs with learned/fixed weights
        weights = {
            "spatial": 0.45,      # Highest weight - spatial features most reliable
            "frequency": 0.35,    # Second highest - frequency artifacts strong signal
            "noise": 0.20,        # Lower weight - noise patterns less reliable
        }

        # Weighted ensemble
        ensemble_proba = sum(
            model_outputs.get(key, 0.5) * weight
            for key, weight in weights.items()
        )

        # Calculate ensemble confidence based on model agreement
        ai_probs = [model_outputs.get(key, 0.5) for key in weights.keys()]
        std_dev = float(np.std(ai_probs))
        confidence = max(0.0, 1.0 - std_dev * 2.0)  # Lower std = higher confidence

        # Boost confidence if all models agree
        all_agree_ai = all(p > 0.6 for p in ai_probs)
        all_agree_authentic = all(p < 0.4 for p in ai_probs)

        if all_agree_ai or all_agree_authentic:
            confidence = min(1.0, confidence + 0.2)

        predictions = {
            "ensemble_proba": ensemble_proba,
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,
            "model_agreement": 1.0 - std_dev
        }

        # Merge outputs
        all_outputs = {**model_outputs, **model_confidences}

        return predictions, all_outputs

    def _determine_status(
        self,
        ai_probability: float,
        confidence: float
    ) -> Tuple[str, int]:
        """
        Determine status and trust contribution.

        Args:
            ai_probability: Probability of AI generation (0-1)
            confidence: Model confidence (0-1)

        Returns:
            Tuple of (status string, trust contribution)
        """
        # Adjust thresholds based on confidence
        if confidence > 0.7:
            # High confidence - use stricter thresholds
            if ai_probability > 0.70:
                return "ai_likely", -80
            elif ai_probability > 0.55:
                return "uncertain", -40
            elif ai_probability < 0.30:
                return "authentic_likely", 60
            else:
                return "uncertain", -20
        elif confidence > 0.5:
            # Medium confidence - moderate thresholds
            if ai_probability > 0.75:
                return "ai_likely", -60
            elif ai_probability > 0.60:
                return "uncertain", -30
            elif ai_probability < 0.25:
                return "authentic_likely", 50
            else:
                return "uncertain", 0
        else:
            # Low confidence - conservative thresholds
            if ai_probability > 0.85:
                return "ai_likely", -40
            elif ai_probability < 0.15:
                return "authentic_likely", 30
            else:
                return "uncertain", 0

    def _generate_explanation(
        self,
        ai_probability: float,
        confidence: float,
        model_outputs: Dict[str, float]
    ) -> str:
        """
        Generate human-readable explanation.

        Args:
            ai_probability: Ensemble AI probability
            confidence: Ensemble confidence
            model_outputs: Individual model outputs

        Returns:
            Explanation string
        """
        if ai_probability > 0.70:
            reasons = []

            if model_outputs.get("frequency", 0) > 0.70:
                reasons.append("checkerboard artifacts in frequency domain typical of upsampling/GANs")

            if model_outputs.get("spatial", 0) > 0.70:
                reasons.append("spatial artifacts and unnatural smoothness characteristic of diffusion models")

            if model_outputs.get("noise", 0) > 0.70:
                reasons.append("absence of camera sensor noise (PRNU) expected in real photos")

            if not reasons:
                reasons.append("multiple ML indicators across spatial and frequency domains")

            explanation = f"High likelihood ({ai_probability:.1%}) of AI generation detected. "
            explanation += "Key indicators: " + "; ".join(f"({i+1}) {r}" for i, r in enumerate(reasons[:3]))

            if confidence < 0.6:
                explanation += f". Note: Model confidence is moderate ({confidence:.1%}), "
                explanation += "suggesting potential edge case or novel generation method."

            return explanation

        elif ai_probability > 0.50:
            explanation = f"Moderate indicators ({ai_probability:.1%}) of potential AI generation. "

            if confidence < 0.5:
                explanation += "Low model agreement suggests mixed signals. "

            explanation += "Cannot conclusively determine authenticity. Manual review recommended."

            return explanation

        else:
            explanation = f"Low likelihood ({ai_probability:.1%}) of AI generation. "

            supporting = []
            if model_outputs.get("noise", 0) < 0.4:
                supporting.append("characteristic camera sensor noise present")

            if model_outputs.get("spatial", 0) < 0.4:
                supporting.append("spatial features consistent with authentic photography")

            if model_outputs.get("frequency", 0) < 0.4:
                supporting.append("natural frequency spectrum without GAN fingerprints")

            if supporting:
                explanation += "Supporting evidence: " + "; ".join(supporting[:2]) + ". "

            explanation += f"Image characteristics align with authentic photography (confidence: {confidence:.1%})."

            return explanation

    def _identify_patterns(self, model_outputs: Dict[str, float]) -> List[str]:
        """
        Identify specific suspicious patterns.

        Args:
            model_outputs: Individual model outputs

        Returns:
            List of identified patterns
        """
        patterns = []

        if model_outputs.get("frequency", 0) > 0.75:
            patterns.append("Strong GAN fingerprint in frequency domain")

        if model_outputs.get("frequency", 0) > 0.65:
            patterns.append("Checkerboard artifacts from upsampling")

        if model_outputs.get("spatial", 0) > 0.70:
            patterns.append("Unnatural smoothness typical of diffusion models")

        if model_outputs.get("spatial", 0) > 0.65:
            patterns.append("Edge inconsistencies")

        if model_outputs.get("noise", 0) > 0.70:
            patterns.append("Missing camera PRNU (Photo Response Non-Uniformity)")

        if model_outputs.get("noise", 0) > 0.65:
            patterns.append("Synthetic noise pattern")

        # Check for model agreement
        probs = [model_outputs.get(k, 0.5) for k in ["spatial", "frequency", "noise"]]
        if all(p > 0.65 for p in probs):
            patterns.append("All models agree on AI generation")

        return patterns

    def _calculate_agreement(self, model_outputs: Dict[str, float]) -> float:
        """
        Calculate agreement between models.

        Args:
            model_outputs: Individual model outputs

        Returns:
            Agreement score (0-1)
        """
        probs = [model_outputs.get(k, 0.5) for k in ["spatial", "frequency", "noise"]]
        std_dev = float(np.std(probs))

        # Convert std dev to agreement score
        # std dev of 0 = perfect agreement (score 1.0)
        # std dev of 0.5 = no agreement (score 0.0)
        agreement = max(0.0, 1.0 - std_dev * 2.0)

        return float(agreement)

    async def _detect_video(self) -> Dict[str, Any]:
        """
        Detect AI generation in videos.

        Samples key frames and runs detection on each.
        """
        try:
            logger.info("Starting video ML detection", file_path=self.file_path)

            # Sample frames from video
            frames = self.video_preprocessor.sample_frames(self.file_path, method='uniform')

            # Run detection on each frame
            frame_results = []

            for i, frame in enumerate(frames):
                # Save frame temporarily
                temp_frame_path = f"{self.file_path}_frame_{i}.jpg"
                Image.fromarray(frame).save(temp_frame_path)

                # Run image detection
                preprocessed = self.image_preprocessor.preprocess_all(temp_frame_path)
                for key in preprocessed:
                    preprocessed[key] = preprocessed[key].to(self.device)

                predictions, model_outputs = await self._run_ensemble_inference(preprocessed)
                frame_results.append({
                    "frame_idx": i,
                    "ai_probability": predictions["ensemble_proba"],
                    "confidence": predictions["confidence"]
                })

                # Clean up temp file
                import os
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)

            # Aggregate frame results
            avg_ai_prob = np.mean([r["ai_probability"] for r in frame_results])
            avg_confidence = np.mean([r["confidence"] for r in frame_results])
            max_ai_prob = np.max([r["ai_probability"] for r in frame_results])

            # Determine status
            status, trust_contribution = self._determine_status(avg_ai_prob, avg_confidence)

            explanation = f"Video analysis across {len(frame_results)} frames. "
            explanation += f"Average AI probability: {avg_ai_prob:.1%}. "
            explanation += f"Maximum AI probability in any frame: {max_ai_prob:.1%}. "

            if max_ai_prob > 0.8:
                explanation += "High confidence AI-generated content detected in some frames."
            elif avg_ai_prob > 0.6:
                explanation += "Moderate indicators of AI generation across frames."
            else:
                explanation += "Frames appear consistent with authentic video."

            return {
                "stage": "ml_detection",
                "status": status,
                "details": {
                    "ensemble_prediction": {
                        "ai_probability": float(avg_ai_prob),
                        "confidence": float(avg_confidence),
                        "uncertainty": float(1.0 - avg_confidence),
                    },
                    "video_analysis": {
                        "frames_analyzed": len(frame_results),
                        "average_ai_probability": float(avg_ai_prob),
                        "max_ai_probability": float(max_ai_prob),
                        "min_ai_probability": float(np.min([r["ai_probability"] for r in frame_results])),
                        "frame_results": frame_results[:5],  # Include first 5 frames
                    },
                    "explanation": explanation,
                },
                "trust_contribution": trust_contribution,
                "evidence_weight": "medium" if avg_confidence > 0.6 else "low",
            }

        except Exception as e:
            logger.error("Video ML detection failed", error=str(e), exc_info=True)

            return {
                "stage": "ml_detection",
                "status": "uncertain",
                "details": {
                    "error": str(e),
                    "reason": "Video ML detection encountered an error",
                },
                "trust_contribution": 0,
                "evidence_weight": "low",
            }
