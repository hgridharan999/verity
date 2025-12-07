"""
Image preprocessing pipeline for ML detection models.

Handles:
- Image loading and validation
- Multi-scale preprocessing
- Frequency domain transforms
- Noise extraction
- Data augmentation for inference-time ensembling
"""

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, Dict, Any
import scipy.fft
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Comprehensive image preprocessing for AI detection models.
    """

    def __init__(
        self,
        spatial_size: Tuple[int, int] = (224, 224),
        frequency_size: Tuple[int, int] = (256, 256),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize preprocessor.

        Args:
            spatial_size: Target size for spatial models
            frequency_size: Target size for frequency analysis
            mean: Normalization mean (ImageNet default)
            std: Normalization std (ImageNet default)
        """
        self.spatial_size = spatial_size
        self.frequency_size = frequency_size
        self.mean = mean
        self.std = std

        # Spatial domain transform
        self.spatial_transform = transforms.Compose([
            transforms.Resize(spatial_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Frequency domain transform (no normalization)
        self.frequency_transform = transforms.Compose([
            transforms.Resize(frequency_size),
        ])

    def load_image(self, image_path: str) -> Tuple[np.ndarray, Image.Image]:
        """
        Load image from file.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (numpy array RGB, PIL Image)

        Raises:
            ValueError: If image cannot be loaded
        """
        try:
            # Load with PIL
            pil_image = Image.open(image_path)

            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert to numpy
            np_image = np.array(pil_image)

            return np_image, pil_image

        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {str(e)}")

    def preprocess_spatial(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for spatial artifact detection.

        Args:
            pil_image: PIL Image

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        tensor = self.spatial_transform(pil_image)
        return tensor.unsqueeze(0)  # Add batch dimension

    def preprocess_frequency(self, np_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for frequency domain analysis.

        Args:
            np_image: Numpy array (H, W, 3)

        Returns:
            Frequency spectrum tensor (1, 1, H, W)
        """
        # Convert to grayscale
        if len(np_image.shape) == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_image

        # Resize
        gray_resized = cv2.resize(gray, self.frequency_size)

        # Compute FFT
        fft = np.fft.fft2(gray_resized)
        fft_shift = np.fft.fftshift(fft)

        # Compute magnitude spectrum (log scale)
        magnitude_spectrum = np.abs(fft_shift)
        magnitude_spectrum = np.log1p(magnitude_spectrum)

        # Normalize to [0, 1]
        magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / \
                            (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)

        # Convert to tensor
        tensor = torch.from_numpy(magnitude_spectrum).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        return tensor

    def preprocess_dct(self, np_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image using DCT (alternative to FFT).

        Args:
            np_image: Numpy array (H, W, 3)

        Returns:
            DCT coefficient tensor (1, 1, H, W)
        """
        # Convert to grayscale
        if len(np_image.shape) == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_image

        # Resize
        gray_resized = cv2.resize(gray, self.frequency_size)

        # Compute DCT
        dct = scipy.fft.dctn(gray_resized, norm='ortho')

        # Log transform for better visualization
        dct_log = np.log1p(np.abs(dct))

        # Normalize to [0, 1]
        dct_norm = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-8)

        # Convert to tensor
        tensor = torch.from_numpy(dct_norm).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor

    def extract_noise(self, np_image: np.ndarray) -> torch.Tensor:
        """
        Extract noise pattern from image.

        Args:
            np_image: Numpy array (H, W, 3)

        Returns:
            Noise tensor (1, 3, H, W)
        """
        # Resize
        image_resized = cv2.resize(np_image, self.spatial_size)

        # Convert to float
        image_float = image_resized.astype(np.float32) / 255.0

        # Apply bilateral filter to get denoised version
        denoised = cv2.bilateralFilter(
            image_float,
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )

        # Noise is the residual
        noise = image_float - denoised

        # Enhance noise signal
        noise = noise * 10.0  # Amplify

        # Clip to valid range
        noise = np.clip(noise, -1.0, 1.0)

        # Normalize to [0, 1]
        noise = (noise + 1.0) / 2.0

        # Convert to tensor (H, W, C) -> (C, H, W)
        noise_tensor = torch.from_numpy(noise).float()
        noise_tensor = noise_tensor.permute(2, 0, 1)
        noise_tensor = noise_tensor.unsqueeze(0)  # Add batch dimension

        return noise_tensor

    def preprocess_all(
        self,
        image_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess image for all models.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with preprocessed tensors for each model type
        """
        # Load image
        np_image, pil_image = self.load_image(image_path)

        # Preprocess for each model type
        preprocessed = {
            'spatial': self.preprocess_spatial(pil_image),
            'frequency': self.preprocess_frequency(np_image),
            'noise': self.extract_noise(np_image),
        }

        # Add DCT as alternative to FFT
        preprocessed['dct'] = self.preprocess_dct(np_image)

        return preprocessed

    def extract_patches(
        self,
        pil_image: Image.Image,
        patch_size: int = 224,
        stride: int = 112,
        max_patches: int = 16
    ) -> torch.Tensor:
        """
        Extract overlapping patches for patch-based detection.

        Args:
            pil_image: PIL Image
            patch_size: Size of each patch
            stride: Stride between patches
            max_patches: Maximum number of patches to extract

        Returns:
            Tensor of patches (N, 3, patch_size, patch_size)
        """
        # Convert to tensor
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(pil_image)

        _, h, w = tensor.shape

        patches = []
        positions = []

        # Extract patches
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = tensor[:, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                positions.append((x, y))

                if len(patches) >= max_patches:
                    break
            if len(patches) >= max_patches:
                break

        if not patches:
            # Image too small, resize and extract center
            resized = transforms.Resize((patch_size, patch_size))(pil_image)
            patch = to_tensor(resized)
            patches.append(patch)

        # Stack patches
        patches_tensor = torch.stack(patches)

        # Normalize
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        patches_tensor = normalize(patches_tensor)

        return patches_tensor

    def apply_tta(
        self,
        pil_image: Image.Image
    ) -> torch.Tensor:
        """
        Apply Test-Time Augmentation.

        Args:
            pil_image: PIL Image

        Returns:
            Tensor with augmented versions (N, 3, H, W)
        """
        augmentations = [
            # Original
            transforms.Compose([
                transforms.Resize(self.spatial_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize(self.spatial_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]),
            # Center crop
            transforms.Compose([
                transforms.Resize(int(self.spatial_size[0] * 1.1)),
                transforms.CenterCrop(self.spatial_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]),
            # Slight rotation
            transforms.Compose([
                transforms.Resize(int(self.spatial_size[0] * 1.1)),
                transforms.RandomRotation(degrees=5),
                transforms.CenterCrop(self.spatial_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]),
        ]

        augmented_tensors = []
        for aug_transform in augmentations:
            tensor = aug_transform(pil_image)
            augmented_tensors.append(tensor)

        return torch.stack(augmented_tensors)


class VideoPreprocessor:
    """
    Video preprocessing for AI detection.

    Samples key frames and preprocesses them.
    """

    def __init__(
        self,
        num_frames: int = 8,
        frame_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize video preprocessor.

        Args:
            num_frames: Number of frames to sample
            frame_size: Target frame size
            mean: Normalization mean
            std: Normalization std
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.mean = mean
        self.std = std

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def sample_frames(
        self,
        video_path: str,
        method: str = 'uniform'
    ) -> np.ndarray:
        """
        Sample frames from video.

        Args:
            video_path: Path to video file
            method: Sampling method ('uniform', 'random', 'keyframes')

        Returns:
            Array of frames (num_frames, H, W, 3)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")

        # Determine frame indices to sample
        if method == 'uniform':
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        elif method == 'random':
            indices = np.random.choice(total_frames, self.num_frames, replace=False)
            indices.sort()
        else:  # keyframes - simplified: take from beginning, middle, end
            indices = []
            for i in range(self.num_frames):
                idx = int((i / (self.num_frames - 1)) * (total_frames - 1))
                indices.append(idx)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()

        if not frames:
            raise ValueError(f"Failed to extract frames from video: {video_path}")

        return np.array(frames)

    def preprocess_frames(
        self,
        frames: np.ndarray
    ) -> torch.Tensor:
        """
        Preprocess video frames.

        Args:
            frames: Array of frames (num_frames, H, W, 3)

        Returns:
            Preprocessed tensor (num_frames, 3, H, W)
        """
        preprocessed = []

        for frame in frames:
            tensor = self.transform(frame)
            preprocessed.append(tensor)

        return torch.stack(preprocessed)
