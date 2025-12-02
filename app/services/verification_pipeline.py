"""Main verification pipeline orchestrator"""
import time
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update

from app.models.schemas import (
    VerificationResponse,
    VerificationOptions,
    VerificationStage,
    RiskCategoryEnum,
)
from app.models.database import Verification, VerificationStatus
from app.services.verification.c2pa_verifier import C2PAVerifier
from app.services.verification.hardware_auth import HardwareAuthenticator
from app.services.verification.metadata_analyzer import MetadataAnalyzer
from app.services.verification.contextual_verifier import ContextualVerifier
from app.services.verification.ml_detector import MLDetector
from app.services.scoring_engine import ScoringEngine
from app.utils.logging import (
    get_logger,
    log_stage_completed,
    log_verification_completed,
    log_verification_failed,
)

logger = get_logger(__name__)


class VerificationPipeline:
    """
    Main verification pipeline that orchestrates all verification stages.

    The pipeline runs 5 stages in order:
    1. C2PA Signature Verification
    2. Hardware Authentication
    3. Metadata Analysis
    4. Contextual Verification
    5. ML Detection

    If C2PA verification succeeds and force_full_pipeline=False, the pipeline
    exits early with high trust score.
    """

    def __init__(
        self,
        verification_id: str,
        file_path: str,
        content_type: str,
        file_format: str,
        options: VerificationOptions,
        db: AsyncSession,
        user_id: Optional[str] = None,
    ):
        self.verification_id = verification_id
        self.file_path = file_path
        self.content_type = content_type
        self.file_format = file_format
        self.options = options
        self.db = db
        self.user_id = user_id

        self.stage_results = {}
        self.metadata_cache = {}

    async def run(self) -> VerificationResponse:
        """
        Run the complete verification pipeline.

        Returns:
            VerificationResponse with complete verification results
        """
        pipeline_start = time.time()

        try:
            logger.info(
                "Starting verification pipeline",
                verification_id=self.verification_id,
                content_type=self.content_type,
                vertical=self.options.vertical,
            )

            # Stage 1: C2PA Verification
            c2pa_result = await self._run_stage_1_c2pa()
            self.stage_results["c2pa"] = c2pa_result

            # Early exit if C2PA verified and not forcing full pipeline
            if (
                c2pa_result["status"] == "verified"
                and not self.options.force_full_pipeline
            ):
                logger.info(
                    "Early exit: C2PA verified",
                    verification_id=self.verification_id
                )
                return await self._create_response(pipeline_start, early_exit=True)

            # Stage 2: Hardware Authentication (needs metadata)
            # First get metadata for subsequent stages
            metadata_result = await self._run_stage_3_metadata()
            self.stage_results["metadata"] = metadata_result
            self.metadata_cache = metadata_result.get("details", {})

            # Now run hardware auth with metadata
            hardware_result = await self._run_stage_2_hardware()
            self.stage_results["hardware"] = hardware_result

            # Stage 4: Contextual Verification
            contextual_result = await self._run_stage_4_contextual()
            self.stage_results["contextual"] = contextual_result

            # Stage 5: ML Detection
            ml_result = await self._run_stage_5_ml()
            self.stage_results["ml"] = ml_result

            # Create final response
            response = await self._create_response(pipeline_start, early_exit=False)

            log_verification_completed(
                verification_id=self.verification_id,
                trust_score=response.trust_score,
                risk_category=response.risk_category.value,
                processing_time_ms=response.processing_time_ms,
            )

            return response

        except Exception as e:
            logger.error(
                "Verification pipeline failed",
                verification_id=self.verification_id,
                error=str(e),
            )
            log_verification_failed(
                verification_id=self.verification_id,
                error=str(e),
            )

            # Update database with failure
            await self.db.execute(
                update(Verification)
                .where(Verification.id == self.verification_id)
                .values(
                    status=VerificationStatus.FAILED,
                    error_message=str(e),
                    processing_completed_at=datetime.utcnow(),
                )
            )
            await self.db.commit()

            raise

    async def _run_stage_1_c2pa(self) -> Dict[str, Any]:
        """Run Stage 1: C2PA Verification"""
        logger.info("Running Stage 1: C2PA Verification", verification_id=self.verification_id)

        verifier = C2PAVerifier(self.file_path)
        result = await verifier.verify()

        log_stage_completed(
            verification_id=self.verification_id,
            stage_name="c2pa_verification",
            status=result["status"],
            duration_ms=result["duration_ms"],
            contribution=result["trust_contribution"],
        )

        return result

    async def _run_stage_2_hardware(self) -> Dict[str, Any]:
        """Run Stage 2: Hardware Authentication"""
        logger.info("Running Stage 2: Hardware Authentication", verification_id=self.verification_id)

        authenticator = HardwareAuthenticator(self.file_path, self.metadata_cache)
        result = await authenticator.authenticate()

        log_stage_completed(
            verification_id=self.verification_id,
            stage_name="hardware_authentication",
            status=result["status"],
            duration_ms=result["duration_ms"],
            contribution=result["trust_contribution"],
        )

        return result

    async def _run_stage_3_metadata(self) -> Dict[str, Any]:
        """Run Stage 3: Metadata Analysis"""
        logger.info("Running Stage 3: Metadata Analysis", verification_id=self.verification_id)

        analyzer = MetadataAnalyzer(self.file_path, self.content_type)
        result = await analyzer.analyze()

        log_stage_completed(
            verification_id=self.verification_id,
            stage_name="metadata_analysis",
            status=result["status"],
            duration_ms=result["duration_ms"],
            contribution=result["trust_contribution"],
        )

        return result

    async def _run_stage_4_contextual(self) -> Dict[str, Any]:
        """Run Stage 4: Contextual Verification"""
        logger.info("Running Stage 4: Contextual Verification", verification_id=self.verification_id)

        # Get file hash from database
        from sqlalchemy import select
        result = await self.db.execute(
            select(Verification.file_hash).where(Verification.id == self.verification_id)
        )
        file_hash = result.scalar_one_or_none()

        verifier = ContextualVerifier(file_hash, self.user_id)
        result = await verifier.verify()

        log_stage_completed(
            verification_id=self.verification_id,
            stage_name="contextual_verification",
            status=result["status"],
            duration_ms=result["duration_ms"],
            contribution=result["trust_contribution"],
        )

        return result

    async def _run_stage_5_ml(self) -> Dict[str, Any]:
        """Run Stage 5: ML Detection"""
        logger.info("Running Stage 5: ML Detection", verification_id=self.verification_id)

        detector = MLDetector(self.file_path, self.content_type)
        result = await detector.detect()

        log_stage_completed(
            verification_id=self.verification_id,
            stage_name="ml_detection",
            status=result["status"],
            duration_ms=result["duration_ms"],
            contribution=result["trust_contribution"],
        )

        return result

    async def _create_response(
        self,
        pipeline_start: float,
        early_exit: bool = False
    ) -> VerificationResponse:
        """Create verification response from stage results"""

        processing_time_ms = int((time.time() - pipeline_start) * 1000)

        # Use scoring engine to calculate trust score
        scoring_engine = ScoringEngine(self.stage_results)
        trust_result = scoring_engine.calculate_trust_score()

        # Build verification stages list
        verification_stages = []

        # Add stages that were run
        stage_mapping = {
            "c2pa": (1, "C2PA Verification"),
            "hardware": (2, "Hardware Authentication"),
            "metadata": (3, "Metadata Analysis"),
            "contextual": (4, "Contextual Verification"),
            "ml": (5, "ML Detection"),
        }

        for key, (stage_num, stage_name) in stage_mapping.items():
            if key in self.stage_results:
                result = self.stage_results[key]
                verification_stages.append(
                    VerificationStage(
                        stage=stage_num,
                        name=stage_name,
                        status=result.get("status", "unknown"),
                        duration_ms=result.get("duration_ms", 0),
                        contribution=result.get("trust_contribution", 0),
                    )
                )

        # Get file hash from database
        from sqlalchemy import select
        db_result = await self.db.execute(
            select(Verification.file_hash).where(Verification.id == self.verification_id)
        )
        file_hash = db_result.scalar_one_or_none() or "unknown"

        # Create response
        response = VerificationResponse(
            verification_id=self.verification_id,
            timestamp=datetime.utcnow(),
            file_hash=file_hash,
            trust_score=trust_result["score"],
            confidence=trust_result["confidence"],
            risk_category=RiskCategoryEnum(trust_result["category"]),
            processing_time_ms=processing_time_ms,
            verification_stages=verification_stages,
            key_findings=trust_result["key_findings"],
            risk_factors=trust_result["risk_factors"],
            recommendation=trust_result["recommendation"],
        )

        # Add detailed results if requested
        if self.options.include_detailed_report:
            from app.models.schemas import (
                C2PAResult,
                HardwareAuthResult,
                MetadataResult,
                ContextualResult,
                MLResult,
            )

            if "c2pa" in self.stage_results:
                response.c2pa_result = C2PAResult(**self.stage_results["c2pa"])
            if "hardware" in self.stage_results:
                response.hardware_auth_result = HardwareAuthResult(**self.stage_results["hardware"])
            if "metadata" in self.stage_results:
                response.metadata_result = MetadataResult(**self.stage_results["metadata"])
            if "contextual" in self.stage_results:
                response.contextual_result = ContextualResult(**self.stage_results["contextual"])
            if "ml" in self.stage_results:
                response.ml_result = MLResult(**self.stage_results["ml"])

        # Update database
        await self.db.execute(
            update(Verification)
            .where(Verification.id == self.verification_id)
            .values(
                status=VerificationStatus.COMPLETED,
                processing_completed_at=datetime.utcnow(),
                processing_time_ms=processing_time_ms,
                trust_score=trust_result["score"],
                confidence=trust_result["confidence"],
                risk_category=trust_result["category"],
                c2pa_result=self.stage_results.get("c2pa"),
                hardware_auth_result=self.stage_results.get("hardware"),
                metadata_result=self.stage_results.get("metadata"),
                contextual_result=self.stage_results.get("contextual"),
                ml_result=self.stage_results.get("ml"),
                key_findings=trust_result["key_findings"],
                risk_factors=trust_result["risk_factors"],
                recommendation=trust_result["recommendation"],
            )
        )
        await self.db.commit()

        return response

    @staticmethod
    def db_to_response(verification: Verification) -> VerificationResponse:
        """Convert database Verification record to VerificationResponse"""
        # Build verification stages from stored results
        verification_stages = []

        stage_mapping = [
            ("c2pa_result", 1, "C2PA Verification"),
            ("hardware_auth_result", 2, "Hardware Authentication"),
            ("metadata_result", 3, "Metadata Analysis"),
            ("contextual_result", 4, "Contextual Verification"),
            ("ml_result", 5, "ML Detection"),
        ]

        for field, stage_num, stage_name in stage_mapping:
            result = getattr(verification, field)
            if result:
                verification_stages.append(
                    VerificationStage(
                        stage=stage_num,
                        name=stage_name,
                        status=result.get("status", "unknown"),
                        duration_ms=result.get("duration_ms", 0),
                        contribution=result.get("trust_contribution", 0),
                    )
                )

        return VerificationResponse(
            verification_id=verification.id,
            timestamp=verification.created_at,
            file_hash=verification.file_hash,
            trust_score=verification.trust_score or 0,
            confidence=verification.confidence or 0,
            risk_category=RiskCategoryEnum(verification.risk_category.value),
            processing_time_ms=verification.processing_time_ms or 0,
            verification_stages=verification_stages,
            key_findings=verification.key_findings or [],
            risk_factors=verification.risk_factors or [],
            recommendation=verification.recommendation or "No recommendation available",
        )
