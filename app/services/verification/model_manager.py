"""
Model manager for loading, caching, and managing ML models.

Handles:
- Model downloading from HuggingFace Hub
- Model caching and lazy loading
- GPU/CPU device management
- Model weight initialization
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
from huggingface_hub import hf_hub_download
from threading import Lock
import hashlib
import json

from app.config import settings
from app.utils.logging import get_logger
from app.services.verification.ml_models import (
    SpatialArtifactDetector,
    FrequencyDomainCNN,
    NoisePatternCNN,
    CLIPBasedDetector,
    EnsembleDetector,
    TemperatureScaling,
)

logger = get_logger(__name__)


class ModelManager:
    """
    Singleton model manager for ML detection models.

    Provides centralized model loading, caching, and device management.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.device = self._get_device()
            self.models: Dict[str, Optional[nn.Module]] = {}
            self.model_configs: Dict[str, Dict[str, Any]] = {}
            self._load_configs()

            logger.info(
                "ModelManager initialized",
                device=str(self.device),
                cuda_available=torch.cuda.is_available()
            )

    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if settings.ml_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                "Using CUDA device",
                gpu_name=torch.cuda.get_device_name(0),
                gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9
            )
        else:
            device = torch.device("cpu")
            if settings.ml_device == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
            else:
                logger.info("Using CPU device")

        return device

    def _load_configs(self):
        """Load model configurations"""
        # These would typically be loaded from a config file
        # For now, we'll define them here

        self.model_configs = {
            "spatial": {
                "class": "SpatialArtifactDetector",
                "params": {
                    "model_name": "efficientnet_b3",
                    "pretrained": True
                },
                "weights_file": "spatial_detector_v1.pth",
                "hub_repo": None,  # None means use pretrained from timm
                "input_size": (224, 224),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "frequency": {
                "class": "FrequencyDomainCNN",
                "params": {},
                "weights_file": "frequency_detector_v1.pth",
                "hub_repo": None,
                "input_size": (256, 256),
            },
            "noise": {
                "class": "NoisePatternCNN",
                "params": {},
                "weights_file": "noise_detector_v1.pth",
                "hub_repo": None,
                "input_size": (224, 224),
            },
            "clip": {
                "class": "CLIPBasedDetector",
                "params": {
                    "clip_model_name": "openai/clip-vit-base-patch32"
                },
                "weights_file": "clip_detector_v1.pth",
                "hub_repo": None,
                "input_size": (224, 224),
            },
            "ensemble": {
                "class": "EnsembleDetector",
                "params": {
                    "use_learned_weights": True
                },
                "weights_file": "ensemble_weights_v1.pth",
                "hub_repo": None,
            },
            "temperature": {
                "class": "TemperatureScaling",
                "params": {},
                "weights_file": "temperature_scaling_v1.pth",
                "hub_repo": None,
            }
        }

    def get_model(self, model_name: str) -> nn.Module:
        """
        Get a model by name, loading it if necessary.

        Args:
            model_name: Name of the model (spatial, frequency, noise, clip, ensemble)

        Returns:
            Loaded model on the appropriate device

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")

        # Return cached model if available
        if model_name in self.models and self.models[model_name] is not None:
            return self.models[model_name]

        # Load the model
        logger.info(f"Loading model: {model_name}")
        model = self._load_model(model_name)
        self.models[model_name] = model

        return model

    def _load_model(self, model_name: str) -> nn.Module:
        """Load a specific model"""
        config = self.model_configs[model_name]
        model_class_name = config["class"]
        model_params = config["params"]
        weights_file = config.get("weights_file")
        hub_repo = config.get("hub_repo")

        # Create model instance
        if model_class_name == "SpatialArtifactDetector":
            model = SpatialArtifactDetector(**model_params)
        elif model_class_name == "FrequencyDomainCNN":
            model = FrequencyDomainCNN(**model_params)
        elif model_class_name == "NoisePatternCNN":
            model = NoisePatternCNN(**model_params)
        elif model_class_name == "CLIPBasedDetector":
            model = CLIPBasedDetector(**model_params)
        elif model_class_name == "EnsembleDetector":
            # For ensemble, we need to load sub-models first
            spatial_model = self.get_model("spatial") if model_name == "ensemble" else None
            frequency_model = self.get_model("frequency") if model_name == "ensemble" else None
            noise_model = self.get_model("noise") if model_name == "ensemble" else None
            clip_model = None  # Optional for now

            model = EnsembleDetector(
                spatial_model=spatial_model,
                frequency_model=frequency_model,
                noise_model=noise_model,
                clip_model=clip_model,
                **model_params
            )
        elif model_class_name == "TemperatureScaling":
            model = TemperatureScaling(**model_params)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")

        # Load weights if available
        if weights_file is not None:
            weights_path = self._get_weights_path(weights_file, hub_repo)

            if weights_path and os.path.exists(weights_path):
                logger.info(f"Loading weights from {weights_path}")
                try:
                    state_dict = torch.load(
                        weights_path,
                        map_location=self.device,
                        weights_only=True
                    )
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded weights for {model_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load weights for {model_name}: {str(e)}. "
                        f"Using randomly initialized weights."
                    )
            else:
                logger.warning(
                    f"No weights file found for {model_name} at {weights_path}. "
                    f"Using pretrained or randomly initialized weights."
                )

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        return model

    def _get_weights_path(
        self,
        weights_file: str,
        hub_repo: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the path to model weights, downloading if necessary.

        Args:
            weights_file: Name of the weights file
            hub_repo: Optional HuggingFace Hub repository (format: "username/repo")

        Returns:
            Path to weights file, or None if not found
        """
        # First check local models directory
        local_path = Path(settings.ml_model_dir) / weights_file

        if local_path.exists():
            return str(local_path)

        # If hub_repo is specified, try to download
        if hub_repo is not None:
            try:
                logger.info(f"Downloading {weights_file} from {hub_repo}")

                downloaded_path = hf_hub_download(
                    repo_id=hub_repo,
                    filename=weights_file,
                    cache_dir=settings.huggingface_cache_dir,
                    resume_download=True
                )

                # Copy to local models directory for faster access next time
                local_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(downloaded_path, local_path)

                logger.info(f"Downloaded and cached {weights_file}")
                return str(local_path)

            except Exception as e:
                logger.error(f"Failed to download {weights_file} from {hub_repo}: {str(e)}")
                return None

        return None

    def preload_models(self, model_names: Optional[list] = None):
        """
        Preload models to avoid lazy loading delays.

        Args:
            model_names: List of model names to preload. If None, preload all.
        """
        if model_names is None:
            model_names = ["spatial", "frequency", "noise"]

        logger.info(f"Preloading models: {model_names}")

        for model_name in model_names:
            try:
                self.get_model(model_name)
            except Exception as e:
                logger.error(f"Failed to preload {model_name}: {str(e)}")

        logger.info("Model preloading complete")

    def clear_cache(self):
        """Clear all cached models to free memory"""
        logger.info("Clearing model cache")

        for model_name in list(self.models.keys()):
            if self.models[model_name] is not None:
                del self.models[model_name]
                self.models[model_name] = None

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model cache cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model status and memory usage
        """
        info = {
            "device": str(self.device),
            "models_loaded": [],
            "models_available": list(self.model_configs.keys())
        }

        for model_name, model in self.models.items():
            if model is not None:
                info["models_loaded"].append(model_name)

        if self.device.type == "cuda":
            info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
            info["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1e6

        return info

    def save_model_weights(
        self,
        model: nn.Module,
        model_name: str,
        version: str = "v1"
    ):
        """
        Save model weights to disk.

        Args:
            model: The model to save
            model_name: Name of the model
            version: Version string
        """
        Path(settings.ml_model_dir).mkdir(parents=True, exist_ok=True)

        weights_file = f"{model_name}_{version}.pth"
        weights_path = Path(settings.ml_model_dir) / weights_file

        logger.info(f"Saving {model_name} weights to {weights_path}")

        torch.save(model.state_dict(), weights_path)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "device": str(self.device),
            "config": self.model_configs.get(model_name, {})
        }

        metadata_path = weights_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model weights saved successfully")


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager

    if _model_manager is None:
        _model_manager = ModelManager()

    return _model_manager
