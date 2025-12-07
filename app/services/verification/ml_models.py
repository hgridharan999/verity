"""
Production-grade ML models for AI-generated image detection.

This module implements state-of-the-art deep learning models for detecting
AI-generated images from various sources (Stable Diffusion, DALL-E, Midjourney, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm
from typing import Dict, Optional, Tuple
import numpy as np
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FrequencyDomainCNN(nn.Module):
    """
    CNN for detecting AI artifacts in frequency domain.

    AI-generated images often have characteristic patterns in DCT/FFT space
    that are not visible in spatial domain.
    """

    def __init__(self):
        super().__init__()

        # Frequency domain feature extractor
        self.freq_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.freq_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.freq_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.freq_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Frequency domain representation (batch, 1, H, W)

        Returns:
            Logits (batch, 2) for [authentic, ai-generated]
        """
        x = self.relu(self.freq_conv1(x))
        x = F.max_pool2d(x, 2)

        x = self.relu(self.freq_conv2(x))
        x = F.max_pool2d(x, 2)

        x = self.relu(self.freq_conv3(x))
        x = self.freq_pool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class NoisePatternCNN(nn.Module):
    """
    CNN specialized for detecting noise patterns and artifacts.

    Real cameras have characteristic noise patterns (PRNU) that differ
    from AI-generated images.
    """

    def __init__(self):
        super().__init__()

        # High-pass filter to extract noise
        self.noise_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual blocks for pattern analysis
        self.res_blocks = nn.Sequential(
            self._make_res_block(64, 128),
            self._make_res_block(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)

    def _make_res_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (batch, 3, H, W)

        Returns:
            Logits (batch, 2) for [authentic, ai-generated]
        """
        # Extract noise patterns
        noise = self.noise_extractor(x)

        # Analyze patterns
        features = self.res_blocks(noise)
        features = self.pool(features)
        features = torch.flatten(features, 1)

        logits = self.fc(features)
        return logits


class SpatialArtifactDetector(nn.Module):
    """
    EfficientNetV2-based detector for spatial artifacts.

    Uses transfer learning from ImageNet with fine-tuning for AI detection.
    Detects artifacts like:
    - Unnatural smoothness
    - Edge inconsistencies
    - Checkerboard patterns from upsampling
    - GAN fingerprints
    """

    def __init__(self, model_name: str = "efficientnet_b3", pretrained: bool = True):
        super().__init__()

        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg"
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_dim = self.backbone(dummy_input).shape[1]

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (batch, 3, H, W)

        Returns:
            Logits (batch, 2) for [authentic, ai-generated]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class CLIPBasedDetector(nn.Module):
    """
    CLIP-based semantic detector for AI-generated content.

    Uses CLIP's vision encoder to detect semantic inconsistencies and
    artifacts that indicate AI generation.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()

        # We'll load CLIP separately due to transformers dependency
        # This is a placeholder for the architecture
        self.feature_dim = 512

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: CLIP image features (batch, feature_dim)

        Returns:
            Logits (batch, 2) for [authentic, ai-generated]
        """
        logits = self.classifier(x)
        return logits


class EnsembleDetector(nn.Module):
    """
    Ensemble model combining multiple detection approaches.

    Combines:
    1. Spatial artifact detection (EfficientNet)
    2. Frequency domain analysis (Frequency CNN)
    3. Noise pattern analysis (Noise CNN)
    4. Semantic analysis (CLIP-based)

    Uses learned weights for optimal combination.
    """

    def __init__(
        self,
        spatial_model: Optional[nn.Module] = None,
        frequency_model: Optional[nn.Module] = None,
        noise_model: Optional[nn.Module] = None,
        clip_model: Optional[nn.Module] = None,
        use_learned_weights: bool = True
    ):
        super().__init__()

        self.spatial_model = spatial_model
        self.frequency_model = frequency_model
        self.noise_model = noise_model
        self.clip_model = clip_model

        # Ensemble weights (can be learned or fixed)
        if use_learned_weights:
            num_models = sum([
                spatial_model is not None,
                frequency_model is not None,
                noise_model is not None,
                clip_model is not None
            ])
            self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        else:
            # Fixed weights (spatial gets highest weight)
            self.register_buffer('weights', torch.tensor([0.4, 0.3, 0.2, 0.1]))

    def forward(
        self,
        spatial_input: Optional[torch.Tensor] = None,
        frequency_input: Optional[torch.Tensor] = None,
        noise_input: Optional[torch.Tensor] = None,
        clip_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through ensemble.

        Args:
            spatial_input: Spatial domain image (batch, 3, H, W)
            frequency_input: Frequency domain representation (batch, 1, H, W)
            noise_input: Noise-extracted image (batch, 3, H, W)
            clip_features: Pre-computed CLIP features (batch, feature_dim)

        Returns:
            Tuple of:
                - Final ensemble logits (batch, 2)
                - Dictionary of individual model outputs
        """
        outputs = []
        model_outputs = {}
        weights = []

        idx = 0

        # Spatial model
        if self.spatial_model is not None and spatial_input is not None:
            spatial_logits = self.spatial_model(spatial_input)
            outputs.append(spatial_logits)
            model_outputs['spatial'] = F.softmax(spatial_logits, dim=1)
            weights.append(self.weights[idx])
            idx += 1

        # Frequency model
        if self.frequency_model is not None and frequency_input is not None:
            freq_logits = self.frequency_model(frequency_input)
            outputs.append(freq_logits)
            model_outputs['frequency'] = F.softmax(freq_logits, dim=1)
            weights.append(self.weights[idx])
            idx += 1

        # Noise model
        if self.noise_model is not None and noise_input is not None:
            noise_logits = self.noise_model(noise_input)
            outputs.append(noise_logits)
            model_outputs['noise'] = F.softmax(noise_logits, dim=1)
            weights.append(self.weights[idx])
            idx += 1

        # CLIP model
        if self.clip_model is not None and clip_features is not None:
            clip_logits = self.clip_model(clip_features)
            outputs.append(clip_logits)
            model_outputs['clip'] = F.softmax(clip_logits, dim=1)
            weights.append(self.weights[idx])
            idx += 1

        if not outputs:
            raise ValueError("No model outputs available for ensemble")

        # Normalize weights
        weights = torch.stack(weights)
        weights = F.softmax(weights, dim=0)

        # Weighted average of logits
        stacked_outputs = torch.stack(outputs, dim=0)  # (num_models, batch, 2)
        weighted_logits = torch.sum(
            stacked_outputs * weights.view(-1, 1, 1),
            dim=0
        )

        model_outputs['ensemble'] = F.softmax(weighted_logits, dim=1)
        model_outputs['weights'] = weights

        return weighted_logits, model_outputs


def compute_dct_2d(image: np.ndarray) -> np.ndarray:
    """
    Compute 2D Discrete Cosine Transform.

    Args:
        image: Grayscale image (H, W) or RGB (H, W, 3)

    Returns:
        DCT coefficients (H, W) or (H, W, 3)
    """
    import scipy.fft

    if len(image.shape) == 3:
        # Process each channel separately
        dct_channels = []
        for i in range(image.shape[2]):
            dct_channel = scipy.fft.dctn(image[:, :, i], norm='ortho')
            dct_channels.append(dct_channel)
        return np.stack(dct_channels, axis=2)
    else:
        return scipy.fft.dctn(image, norm='ortho')


def compute_frequency_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute frequency spectrum using FFT.

    Args:
        image: Grayscale image (H, W)

    Returns:
        Magnitude spectrum (H, W)
    """
    # Compute FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # Compute magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = np.abs(fft_shift)
    magnitude_spectrum = np.log1p(magnitude_spectrum)

    return magnitude_spectrum


def extract_noise_pattern(image: np.ndarray) -> np.ndarray:
    """
    Extract noise pattern from image using wavelet denoising.

    Args:
        image: RGB image (H, W, 3)

    Returns:
        Noise residual (H, W, 3)
    """
    import cv2

    # Convert to float
    image_float = image.astype(np.float32) / 255.0

    # Apply bilateral filter to get denoised version
    denoised = cv2.bilateralFilter(image_float, d=9, sigmaColor=75, sigmaSpace=75)

    # Noise is the residual
    noise = image_float - denoised

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    return noise


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.

    Improves probability estimates to match actual accuracy.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling.

        Args:
            logits: Model logits (batch, num_classes)

        Returns:
            Calibrated probabilities (batch, num_classes)
        """
        return F.softmax(logits / self.temperature, dim=1)
