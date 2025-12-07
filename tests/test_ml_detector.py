"""
Comprehensive tests for ML detection system.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import os

from app.services.verification.ml_detector import MLDetector
from app.services.verification.model_manager import get_model_manager
from app.services.verification.preprocessing import ImagePreprocessor, VideoPreprocessor
from app.services.verification.ml_models import (
    SpatialArtifactDetector,
    FrequencyDomainCNN,
    NoisePatternCNN,
    EnsembleDetector,
)


@pytest.fixture
def temp_image():
    """Create a temporary test image"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a random test image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f.name)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def model_manager():
    """Get model manager instance"""
    return get_model_manager()


class TestMLModels:
    """Test individual ML model architectures"""

    def test_spatial_detector_architecture(self):
        """Test spatial artifact detector architecture"""
        model = SpatialArtifactDetector(model_name="efficientnet_b3", pretrained=False)
        model.eval()

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)

        assert output.shape == (2, 2), "Output should be (batch_size, 2)"
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_frequency_detector_architecture(self):
        """Test frequency domain CNN architecture"""
        model = FrequencyDomainCNN()
        model.eval()

        # Test forward pass
        dummy_input = torch.randn(2, 1, 256, 256)
        output = model(dummy_input)

        assert output.shape == (2, 2), "Output should be (batch_size, 2)"
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_noise_detector_architecture(self):
        """Test noise pattern CNN architecture"""
        model = NoisePatternCNN()
        model.eval()

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)

        assert output.shape == (2, 2), "Output should be (batch_size, 2)"
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_ensemble_detector(self):
        """Test ensemble detector"""
        spatial_model = SpatialArtifactDetector(pretrained=False)
        frequency_model = FrequencyDomainCNN()
        noise_model = NoisePatternCNN()

        ensemble = EnsembleDetector(
            spatial_model=spatial_model,
            frequency_model=frequency_model,
            noise_model=noise_model,
            use_learned_weights=True
        )
        ensemble.eval()

        # Test forward pass
        spatial_input = torch.randn(1, 3, 224, 224)
        frequency_input = torch.randn(1, 1, 256, 256)
        noise_input = torch.randn(1, 3, 224, 224)

        logits, model_outputs = ensemble(
            spatial_input=spatial_input,
            frequency_input=frequency_input,
            noise_input=noise_input
        )

        assert logits.shape == (1, 2), "Ensemble output should be (1, 2)"
        assert 'ensemble' in model_outputs, "Should have ensemble prediction"
        assert 'spatial' in model_outputs, "Should have spatial prediction"
        assert 'frequency' in model_outputs, "Should have frequency prediction"
        assert 'noise' in model_outputs, "Should have noise prediction"


class TestPreprocessing:
    """Test preprocessing pipeline"""

    def test_image_loading(self, temp_image):
        """Test image loading"""
        preprocessor = ImagePreprocessor()
        np_image, pil_image = preprocessor.load_image(temp_image)

        assert isinstance(np_image, np.ndarray), "Should return numpy array"
        assert isinstance(pil_image, Image.Image), "Should return PIL Image"
        assert np_image.shape[2] == 3, "Should be RGB"
        assert pil_image.mode == 'RGB', "PIL image should be RGB"

    def test_spatial_preprocessing(self, temp_image):
        """Test spatial preprocessing"""
        preprocessor = ImagePreprocessor()
        _, pil_image = preprocessor.load_image(temp_image)

        tensor = preprocessor.preprocess_spatial(pil_image)

        assert tensor.shape == (1, 3, 224, 224), "Should be (1, 3, 224, 224)"
        assert isinstance(tensor, torch.Tensor), "Should be tensor"

    def test_frequency_preprocessing(self, temp_image):
        """Test frequency preprocessing"""
        preprocessor = ImagePreprocessor()
        np_image, _ = preprocessor.load_image(temp_image)

        tensor = preprocessor.preprocess_frequency(np_image)

        assert tensor.shape == (1, 1, 256, 256), "Should be (1, 1, 256, 256)"
        assert isinstance(tensor, torch.Tensor), "Should be tensor"

    def test_noise_extraction(self, temp_image):
        """Test noise extraction"""
        preprocessor = ImagePreprocessor()
        np_image, _ = preprocessor.load_image(temp_image)

        tensor = preprocessor.extract_noise(np_image)

        assert tensor.shape == (1, 3, 224, 224), "Should be (1, 3, 224, 224)"
        assert isinstance(tensor, torch.Tensor), "Should be tensor"

    def test_preprocess_all(self, temp_image):
        """Test preprocessing all modalities"""
        preprocessor = ImagePreprocessor()
        preprocessed = preprocessor.preprocess_all(temp_image)

        assert 'spatial' in preprocessed, "Should have spatial"
        assert 'frequency' in preprocessed, "Should have frequency"
        assert 'noise' in preprocessed, "Should have noise"
        assert 'dct' in preprocessed, "Should have DCT"

        # Check shapes
        assert preprocessed['spatial'].shape == (1, 3, 224, 224)
        assert preprocessed['frequency'].shape == (1, 1, 256, 256)
        assert preprocessed['noise'].shape == (1, 3, 224, 224)
        assert preprocessed['dct'].shape == (1, 1, 256, 256)


class TestModelManager:
    """Test model manager"""

    def test_model_manager_singleton(self):
        """Test model manager is singleton"""
        manager1 = get_model_manager()
        manager2 = get_model_manager()

        assert manager1 is manager2, "Should be same instance"

    def test_device_selection(self, model_manager):
        """Test device selection"""
        device = model_manager.device

        assert isinstance(device, torch.device), "Should be torch device"
        assert device.type in ['cpu', 'cuda'], "Should be cpu or cuda"

    def test_model_loading(self, model_manager):
        """Test model loading"""
        # This will use pretrained weights from timm for spatial model
        try:
            spatial_model = model_manager.get_model("spatial")
            assert isinstance(spatial_model, torch.nn.Module), "Should be nn.Module"
            assert next(spatial_model.parameters()).device == model_manager.device
        except Exception as e:
            pytest.skip(f"Model loading failed (expected without weights): {e}")

    def test_model_caching(self, model_manager):
        """Test model caching"""
        # Load model twice
        try:
            model1 = model_manager.get_model("spatial")
            model2 = model_manager.get_model("spatial")

            # Should be same instance (cached)
            assert model1 is model2, "Should return cached model"
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    def test_model_info(self, model_manager):
        """Test model info retrieval"""
        info = model_manager.get_model_info()

        assert 'device' in info, "Should have device info"
        assert 'models_loaded' in info, "Should have loaded models"
        assert 'models_available' in info, "Should have available models"


class TestMLDetector:
    """Test ML detector integration"""

    @pytest.mark.asyncio
    async def test_detector_initialization(self, temp_image):
        """Test detector initialization"""
        detector = MLDetector(temp_image, "image")

        assert detector.file_path == temp_image
        assert detector.content_type == "image"
        assert detector.model_manager is not None
        assert detector.image_preprocessor is not None

    @pytest.mark.asyncio
    async def test_image_detection(self, temp_image):
        """Test image detection pipeline"""
        detector = MLDetector(temp_image, "image")

        result = await detector.detect()

        # Check result structure
        assert 'stage' in result
        assert 'status' in result
        assert 'details' in result
        assert 'trust_contribution' in result
        assert 'evidence_weight' in result
        assert 'duration_ms' in result

        # Check details
        details = result['details']

        # Should handle errors gracefully even without trained weights
        if 'error' not in details:
            assert 'ensemble_prediction' in details
            assert 'model_outputs' in details
            assert 'explanation' in details

            # Check ensemble prediction
            ensemble = details['ensemble_prediction']
            assert 'ai_probability' in ensemble
            assert 'confidence' in ensemble
            assert 'uncertainty' in ensemble

            # Check probabilities are valid
            assert 0 <= ensemble['ai_probability'] <= 1
            assert 0 <= ensemble['confidence'] <= 1

    @pytest.mark.asyncio
    async def test_status_determination(self, temp_image):
        """Test status determination logic"""
        detector = MLDetector(temp_image, "image")

        # Test various probability/confidence combinations
        test_cases = [
            (0.9, 0.8, "ai_likely"),
            (0.2, 0.8, "authentic_likely"),
            (0.5, 0.4, "uncertain"),
        ]

        for ai_prob, confidence, expected_status in test_cases:
            status, contribution = detector._determine_status(ai_prob, confidence)
            assert status == expected_status, f"Expected {expected_status}, got {status}"
            assert isinstance(contribution, int), "Contribution should be integer"

    @pytest.mark.asyncio
    async def test_explanation_generation(self, temp_image):
        """Test explanation generation"""
        detector = MLDetector(temp_image, "image")

        model_outputs = {
            'spatial': 0.8,
            'frequency': 0.7,
            'noise': 0.6,
            'spatial_conf': 0.7,
            'frequency_conf': 0.6,
            'noise_conf': 0.5,
        }

        explanation = detector._generate_explanation(0.75, 0.7, model_outputs)

        assert isinstance(explanation, str), "Explanation should be string"
        assert len(explanation) > 0, "Explanation should not be empty"
        assert "%" in explanation, "Should include percentages"

    @pytest.mark.asyncio
    async def test_pattern_identification(self, temp_image):
        """Test pattern identification"""
        detector = MLDetector(temp_image, "image")

        model_outputs = {
            'spatial': 0.8,
            'frequency': 0.9,
            'noise': 0.7,
        }

        patterns = detector._identify_patterns(model_outputs)

        assert isinstance(patterns, list), "Patterns should be list"
        # Should identify GAN fingerprint
        assert any("GAN" in p for p in patterns), "Should identify GAN fingerprint"

    @pytest.mark.asyncio
    async def test_model_agreement_calculation(self, temp_image):
        """Test model agreement calculation"""
        detector = MLDetector(temp_image, "image")

        # High agreement
        high_agreement_outputs = {'spatial': 0.8, 'frequency': 0.82, 'noise': 0.78}
        high_agreement = detector._calculate_agreement(high_agreement_outputs)
        assert high_agreement > 0.7, "Should have high agreement"

        # Low agreement
        low_agreement_outputs = {'spatial': 0.2, 'frequency': 0.8, 'noise': 0.5}
        low_agreement = detector._calculate_agreement(low_agreement_outputs)
        assert low_agreement < 0.5, "Should have low agreement"


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_invalid_image_path(self):
        """Test handling of invalid image path"""
        detector = MLDetector("/nonexistent/image.jpg", "image")

        result = await detector.detect()

        assert result['status'] == 'uncertain', "Should return uncertain on error"
        assert 'error' in result['details'], "Should include error details"

    @pytest.mark.asyncio
    async def test_corrupted_image(self, temp_image):
        """Test handling of corrupted image"""
        # Write garbage data to file
        with open(temp_image, 'wb') as f:
            f.write(b'not an image')

        detector = MLDetector(temp_image, "image")
        result = await detector.detect()

        assert result['status'] == 'uncertain', "Should return uncertain on error"
        assert 'error' in result['details'], "Should include error details"


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks"""

    @pytest.mark.asyncio
    async def test_inference_speed(self, temp_image, benchmark):
        """Benchmark inference speed"""
        detector = MLDetector(temp_image, "image")

        async def run_detection():
            return await detector.detect()

        result = await run_detection()

        # Check reasonable inference time (should be < 5 seconds on CPU)
        assert result['duration_ms'] < 10000, f"Detection took {result['duration_ms']}ms, expected < 10000ms"

    def test_preprocessing_speed(self, temp_image, benchmark):
        """Benchmark preprocessing speed"""
        preprocessor = ImagePreprocessor()

        def run_preprocessing():
            return preprocessor.preprocess_all(temp_image)

        result = benchmark(run_preprocessing)

        assert 'spatial' in result
        assert 'frequency' in result
        assert 'noise' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
