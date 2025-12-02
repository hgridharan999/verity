"""Trust score calculation and aggregation engine"""
import numpy as np
from typing import Dict, Any, List
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ScoringEngine:
    """
    Aggregates verification stage results into a unified trust score.

    The scoring algorithm:
    1. If C2PA verified: Return high trust score (95) immediately
    2. If C2PA invalid: Return very low trust score (5) immediately
    3. Otherwise: Weighted aggregation of all stage contributions
    """

    def __init__(self, stage_results: Dict[str, Dict[str, Any]]):
        self.stage_results = stage_results

    def calculate_trust_score(self) -> Dict[str, Any]:
        """
        Calculate trust score from all stage results.

        Returns:
            Dictionary with:
            - score: float (0-100)
            - confidence: float (0-100)
            - category: str (risk category)
            - key_findings: List[str]
            - risk_factors: List[str]
            - recommendation: str
        """
        # Check C2PA first (highest priority)
        if "c2pa" in self.stage_results:
            c2pa_status = self.stage_results["c2pa"]["status"]

            if c2pa_status == "verified":
                return self._c2pa_verified_result()
            elif c2pa_status == "invalid":
                return self._c2pa_invalid_result()

        # Calculate weighted score from all stages
        return self._calculate_weighted_score()

    def _c2pa_verified_result(self) -> Dict[str, Any]:
        """Return result for C2PA verified content"""
        c2pa_details = self.stage_results["c2pa"].get("details", {})

        key_findings = [
            "Valid C2PA cryptographic signature verified",
            "Content provenance chain authenticated",
        ]

        # Add issuer if available
        issuer = c2pa_details.get("issuer")
        if issuer:
            key_findings.append(f"Signed by: {issuer}")

        # Add timestamp if available
        timestamp = c2pa_details.get("timestamp")
        if timestamp:
            key_findings.append(f"Creation timestamp: {timestamp}")

        return {
            "score": 95.0,
            "confidence": 95.0,
            "category": "verified",
            "key_findings": key_findings,
            "risk_factors": [],
            "recommendation": "Content has valid C2PA signature and authenticated provenance chain. "
                            "High confidence in authenticity.",
        }

    def _c2pa_invalid_result(self) -> Dict[str, Any]:
        """Return result for invalid C2PA signature"""
        return {
            "score": 5.0,
            "confidence": 95.0,
            "category": "fraudulent",
            "key_findings": [],
            "risk_factors": [
                "Invalid or tampered C2PA signature detected",
                "Content has been modified after signing",
            ],
            "recommendation": "Content has an invalid C2PA signature, indicating tampering or fraud. "
                            "Do not trust this content.",
        }

    def _calculate_weighted_score(self) -> Dict[str, Any]:
        """Calculate weighted score from all available stages"""
        base_score = 50.0  # Neutral starting point
        contributions = []
        weights = []
        key_findings = []
        risk_factors = []

        # Collect contributions from each stage
        stage_info = [
            ("hardware", 0.25, "Hardware Authentication"),
            ("metadata", 0.20, "Metadata Analysis"),
            ("contextual", 0.15, "Contextual Verification"),
            ("ml", 0.40, "ML Detection"),
        ]

        for stage_key, weight, stage_name in stage_info:
            if stage_key in self.stage_results:
                result = self.stage_results[stage_key]
                contribution = result.get("trust_contribution", 0)
                status = result.get("status", "unknown")

                contributions.append(contribution)
                weights.append(weight)

                # Extract key findings
                findings = self._extract_findings(stage_key, result)
                key_findings.extend(findings["positive"])
                risk_factors.extend(findings["negative"])

        # If no stages ran (shouldn't happen), return uncertain
        if not contributions:
            return {
                "score": 50.0,
                "confidence": 0.0,
                "category": "uncertain",
                "key_findings": [],
                "risk_factors": ["Insufficient data for verification"],
                "recommendation": "Unable to verify content authenticity due to lack of verification data.",
            }

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Calculate weighted contribution
        weighted_contribution = np.dot(contributions, weights)

        # Calculate final score
        final_score = base_score + weighted_contribution
        final_score = np.clip(final_score, 0, 100)

        # Calculate confidence based on agreement between stages
        confidence = self._calculate_confidence(contributions)

        # Categorize risk
        category = self._categorize_risk(final_score)

        # Generate recommendation
        recommendation = self._generate_recommendation(final_score, confidence, key_findings, risk_factors)

        return {
            "score": float(final_score),
            "confidence": float(confidence),
            "category": category,
            "key_findings": key_findings[:5],  # Top 5 findings
            "risk_factors": risk_factors[:5],  # Top 5 risks
            "recommendation": recommendation,
        }

    def _extract_findings(
        self,
        stage_key: str,
        result: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Extract key findings and risk factors from stage result"""
        positive = []
        negative = []

        status = result.get("status", "")
        details = result.get("details", {})
        contribution = result.get("trust_contribution", 0)

        if stage_key == "hardware":
            if status == "authenticated":
                device = details.get("device_manufacturer", "Unknown")
                model = details.get("device_model", "")
                positive.append(f"Valid hardware signature from {device} {model}".strip())
            elif contribution < 0:
                negative.append("No trusted hardware signature found")

        elif stage_key == "metadata":
            if status == "consistent":
                positive.append("Metadata is consistent and well-formed")
                if details.get("camera_model"):
                    positive.append(f"Camera: {details.get('camera_model')}")
            elif status == "anomalous":
                anomalies = details.get("anomalies", [])
                for anomaly in anomalies[:2]:  # Max 2
                    negative.append(f"Metadata anomaly: {anomaly}")

        elif stage_key == "contextual":
            if status == "trusted":
                rep = details.get("creator_reputation", {})
                if rep.get("reputation_score", 0) > 80:
                    positive.append("Creator has high reputation score")
            elif status == "suspicious":
                if details.get("cross_reference", {}).get("reverse_image_found"):
                    negative.append("Content found in reverse image search")

        elif stage_key == "ml":
            if status == "ai_likely":
                explanation = details.get("explanation", "")
                if explanation:
                    negative.append(f"ML detection: {explanation}")
            elif status == "authentic_likely":
                positive.append("ML models indicate likely authentic")

        return {"positive": positive, "negative": negative}

    def _calculate_confidence(self, contributions: List[float]) -> float:
        """
        Calculate confidence based on agreement between stages.

        Higher agreement = higher confidence
        """
        if len(contributions) < 2:
            return 50.0  # Low confidence with only one stage

        # Calculate standard deviation
        std_dev = np.std(contributions)

        # Convert to confidence (lower std dev = higher confidence)
        # Max std dev we expect is ~100 (complete disagreement)
        # Min std dev is 0 (perfect agreement)
        confidence = 100 - min(std_dev, 100)

        # Boost confidence if all agree on direction
        all_positive = all(c > 0 for c in contributions)
        all_negative = all(c < 0 for c in contributions)

        if all_positive or all_negative:
            confidence = min(100, confidence + 20)

        return float(confidence)

    def _categorize_risk(self, score: float) -> str:
        """Categorize risk based on score"""
        if score >= 90:
            return "authentic_high_confidence"
        elif score >= 70:
            return "likely_authentic"
        elif score >= 50:
            return "uncertain"
        elif score >= 30:
            return "likely_synthetic"
        else:
            return "synthetic_high_confidence"

    def _generate_recommendation(
        self,
        score: float,
        confidence: float,
        key_findings: List[str],
        risk_factors: List[str]
    ) -> str:
        """Generate human-readable recommendation"""
        if score >= 80:
            return (
                "Content appears authentic with high confidence. "
                "Multiple verification signals support authenticity. "
                "Recommended for acceptance."
            )
        elif score >= 60:
            return (
                "Content is likely authentic based on available evidence. "
                f"Confidence level: {confidence:.0f}%. "
                "Consider for acceptance with standard review."
            )
        elif score >= 40:
            return (
                "Unable to determine authenticity with sufficient confidence. "
                "Mixed signals from verification stages. "
                "Recommend manual review before acceptance."
            )
        elif score >= 20:
            return (
                "Content shows indicators of potential AI generation or manipulation. "
                "Multiple risk factors identified. "
                "Recommend additional verification before acceptance."
            )
        else:
            return (
                "High likelihood of AI-generated or manipulated content. "
                "Strong indicators of synthetic origin. "
                "Not recommended for acceptance without thorough manual review."
            )
