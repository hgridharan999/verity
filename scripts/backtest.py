"""
Backtesting framework for Verity verification system.

This script allows you to test the verification pipeline against labeled datasets
and measure performance metrics.
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import time
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.verification_pipeline import VerificationPipeline
from app.models.schemas import VerificationOptions


@dataclass
class TestCase:
    """Single test case"""
    file_path: str
    true_label: str  # "authentic" or "ai_generated"
    metadata: Dict[str, Any]  # Additional info (generator, source, etc.)


@dataclass
class BacktestResult:
    """Results for a single verification"""
    file_path: str
    true_label: str
    predicted_label: str  # Based on risk_category
    trust_score: float
    risk_category: str
    processing_time_ms: int
    stage_results: Dict[str, Any]
    correct: bool
    confidence: float


class BacktestRunner:
    """Run backtests on the verification system"""

    def __init__(self, dataset_dir: str, results_dir: str = "./backtest_results"):
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def load_dataset(self, manifest_file: str) -> List[TestCase]:
        """
        Load dataset from manifest file.

        Manifest format (JSON):
        {
            "test_cases": [
                {
                    "file_path": "path/to/image.jpg",
                    "true_label": "authentic",  # or "ai_generated"
                    "metadata": {
                        "source": "camera_model",
                        "generator": null,
                        "notes": "..."
                    }
                },
                ...
            ]
        }
        """
        manifest_path = self.dataset_dir / manifest_file

        with open(manifest_path, 'r') as f:
            data = json.load(f)

        test_cases = []
        for case in data["test_cases"]:
            test_cases.append(TestCase(
                file_path=str(self.dataset_dir / case["file_path"]),
                true_label=case["true_label"],
                metadata=case.get("metadata", {})
            ))

        return test_cases

    async def run_backtest(
        self,
        test_cases: List[TestCase],
        options: VerificationOptions = None
    ) -> List[BacktestResult]:
        """Run verification on all test cases"""

        if options is None:
            options = VerificationOptions(
                priority="standard",
                include_detailed_report=True,
                force_full_pipeline=True  # Run all stages
            )

        results = []

        for i, test_case in enumerate(test_cases):
            print(f"Processing {i+1}/{len(test_cases)}: {Path(test_case.file_path).name}")

            try:
                # Run verification
                pipeline = VerificationPipeline(
                    file_path=test_case.file_path,
                    content_type="image",  # Assume images for now
                    verification_id=f"backtest_{i}",
                    options=options
                )

                verification_result = await pipeline.run()

                # Map risk_category to binary prediction
                predicted_label = self._map_risk_to_label(verification_result.risk_category)

                # Create result
                result = BacktestResult(
                    file_path=test_case.file_path,
                    true_label=test_case.true_label,
                    predicted_label=predicted_label,
                    trust_score=verification_result.trust_score,
                    risk_category=verification_result.risk_category,
                    processing_time_ms=verification_result.processing_time_ms,
                    stage_results={
                        stage.name: {
                            "status": stage.status,
                            "contribution": stage.contribution
                        }
                        for stage in verification_result.stages
                    },
                    correct=(predicted_label == test_case.true_label),
                    confidence=verification_result.confidence
                )

                results.append(result)

            except Exception as e:
                print(f"Error processing {test_case.file_path}: {str(e)}")
                continue

        return results

    def _map_risk_to_label(self, risk_category: str) -> str:
        """Map risk category to binary label"""
        ai_categories = ["likely_synthetic", "synthetic_high_confidence", "fraudulent"]
        authentic_categories = ["verified", "authentic_high_confidence", "likely_authentic"]

        if risk_category in ai_categories:
            return "ai_generated"
        elif risk_category in authentic_categories:
            return "authentic"
        else:
            return "uncertain"

    def calculate_metrics(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Calculate performance metrics"""

        # Filter out uncertain predictions for binary metrics
        binary_results = [r for r in results if r.predicted_label != "uncertain"]

        # True positives, false positives, etc.
        tp = sum(1 for r in binary_results if r.true_label == "ai_generated" and r.predicted_label == "ai_generated")
        tn = sum(1 for r in binary_results if r.true_label == "authentic" and r.predicted_label == "authentic")
        fp = sum(1 for r in binary_results if r.true_label == "authentic" and r.predicted_label == "ai_generated")
        fn = sum(1 for r in binary_results if r.true_label == "ai_generated" and r.predicted_label == "authentic")

        # Calculate metrics
        accuracy = (tp + tn) / len(binary_results) if binary_results else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # False positive/negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Per-stage analysis
        stage_performance = self._analyze_stage_performance(results)

        # Trust score distribution
        ai_scores = [r.trust_score for r in results if r.true_label == "ai_generated"]
        auth_scores = [r.trust_score for r in results if r.true_label == "authentic"]

        return {
            "overall": {
                "total_samples": len(results),
                "binary_samples": len(binary_results),
                "uncertain_samples": len(results) - len(binary_results),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
            },
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            },
            "trust_scores": {
                "ai_generated": {
                    "mean": sum(ai_scores) / len(ai_scores) if ai_scores else 0,
                    "min": min(ai_scores) if ai_scores else 0,
                    "max": max(ai_scores) if ai_scores else 0,
                },
                "authentic": {
                    "mean": sum(auth_scores) / len(auth_scores) if auth_scores else 0,
                    "min": min(auth_scores) if auth_scores else 0,
                    "max": max(auth_scores) if auth_scores else 0,
                },
            },
            "stage_performance": stage_performance,
            "avg_processing_time_ms": sum(r.processing_time_ms for r in results) / len(results) if results else 0,
        }

    def _analyze_stage_performance(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Analyze performance of individual stages"""

        stage_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "uncertain": 0})

        for result in results:
            for stage_name, stage_data in result.stage_results.items():
                # Simplified: just track if stage contributed correctly
                if result.correct:
                    stage_stats[stage_name]["correct"] += 1
                else:
                    stage_stats[stage_name]["incorrect"] += 1

        return dict(stage_stats)

    def save_results(self, results: List[BacktestResult], metrics: Dict[str, Any], run_name: str):
        """Save results to JSON files"""

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / f"{run_name}_{timestamp}"
        run_dir.mkdir(exist_ok=True)

        # Save detailed results
        with open(run_dir / "results.json", 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        # Save metrics
        with open(run_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save summary
        summary = {
            "run_name": run_name,
            "timestamp": timestamp,
            "total_samples": len(results),
            "accuracy": metrics["overall"]["accuracy"],
            "precision": metrics["overall"]["precision"],
            "recall": metrics["overall"]["recall"],
            "f1_score": metrics["overall"]["f1_score"],
        }

        with open(run_dir / "summary.txt", 'w') as f:
            f.write(f"Backtest Results: {run_name}\n")
            f.write(f"=" * 60 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n" + "=" * 60 + "\n")
            f.write(f"\nConfusion Matrix:\n")
            for key, value in metrics["confusion_matrix"].items():
                f.write(f"  {key}: {value}\n")

        print(f"\nResults saved to: {run_dir}")
        return run_dir


async def main():
    """Example usage"""

    # Initialize runner
    runner = BacktestRunner(dataset_dir="./test_datasets/ai_detection_v1")

    # Load dataset
    print("Loading dataset...")
    test_cases = runner.load_dataset("manifest.json")
    print(f"Loaded {len(test_cases)} test cases")

    # Run backtest
    print("\nRunning backtest...")
    results = await runner.run_backtest(test_cases)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = runner.calculate_metrics(results)

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['overall']['accuracy']:.2%}")
    print(f"Precision: {metrics['overall']['precision']:.2%}")
    print(f"Recall:    {metrics['overall']['recall']:.2%}")
    print(f"F1 Score:  {metrics['overall']['f1_score']:.2%}")
    print(f"FPR:       {metrics['overall']['false_positive_rate']:.2%}")
    print(f"FNR:       {metrics['overall']['false_negative_rate']:.2%}")
    print("=" * 60)

    # Save results
    runner.save_results(results, metrics, run_name="baseline")


if __name__ == "__main__":
    asyncio.run(main())
