#!/usr/bin/env python3
"""
Model training script for AI detection models.

Usage:
    python scripts/train_models.py --model spatial --data_dir /path/to/data
    python scripts/train_models.py --model frequency --data_dir /path/to/data
    python scripts/train_models.py --model noise --data_dir /path/to/data
    python scripts/train_models.py --model all --data_dir /path/to/data
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.verification.ml_models import (
    SpatialArtifactDetector,
    FrequencyDomainCNN,
    NoisePatternCNN,
)
from app.services.verification.preprocessing import ImagePreprocessor
from app.services.verification.model_manager import get_model_manager
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AIDetectionDataset(Dataset):
    """
    Dataset for AI-generated image detection.

    Expected directory structure:
    data_dir/
        authentic/
            img1.jpg
            img2.jpg
            ...
        ai_generated/
            img1.jpg
            img2.jpg
            ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        preprocessing_mode: str = 'spatial'
    ):
        """
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            transform: Optional transform
            preprocessing_mode: 'spatial', 'frequency', or 'noise'
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.preprocessing_mode = preprocessing_mode

        # Load file paths and labels
        self.samples = []

        # Authentic images (label 0)
        authentic_dir = self.data_dir / 'authentic'
        if authentic_dir.exists():
            for img_path in authentic_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 0))
            for img_path in authentic_dir.glob('*.png'):
                self.samples.append((str(img_path), 0))

        # AI-generated images (label 1)
        ai_dir = self.data_dir / 'ai_generated'
        if ai_dir.exists():
            for img_path in ai_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 1))
            for img_path in ai_dir.glob('*.png'):
                self.samples.append((str(img_path), 1))

        logger.info(
            f"Loaded {len(self.samples)} samples for {split} split",
            authentic=sum(1 for _, label in self.samples if label == 0),
            ai_generated=sum(1 for _, label in self.samples if label == 1)
        )

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        try:
            if self.preprocessing_mode == 'spatial':
                _, pil_image = self.preprocessor.load_image(img_path)
                if self.transform:
                    image = self.transform(pil_image)
                else:
                    image = self.preprocessor.preprocess_spatial(pil_image).squeeze(0)

            elif self.preprocessing_mode == 'frequency':
                np_image, _ = self.preprocessor.load_image(img_path)
                image = self.preprocessor.preprocess_frequency(np_image).squeeze(0)

            elif self.preprocessing_mode == 'noise':
                np_image, _ = self.preprocessor.load_image(img_path)
                image = self.preprocessor.extract_noise(np_image).squeeze(0)

            else:
                raise ValueError(f"Unknown preprocessing mode: {self.preprocessing_mode}")

            return image, label

        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return random tensor and label as fallback
            if self.preprocessing_mode == 'frequency':
                return torch.zeros(1, 256, 256), label
            else:
                return torch.zeros(3, 224, 224), label


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device('cpu'),
    save_dir: str = './models',
    model_name: str = 'model'
) -> Dict[str, List[float]]:
    """
    Train a detection model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        model_name: Name for saved model

    Returns:
        Dictionary with training history
    """
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * val_correct / val_total:.2f}%'
                })

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Learning rate scheduler step
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}",
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=scheduler.get_last_lr()[0]
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save checkpoint
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_path = Path(save_dir) / f'{model_name}_best.pth'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)

            logger.info(f"Saved best model to {checkpoint_path} (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    final_path = Path(save_dir) / f'{model_name}_v1.pth'
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")

    # Save training history
    history_path = Path(save_dir) / f'{model_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return history


def train_spatial_model(data_dir: str, save_dir: str, device: torch.device):
    """Train spatial artifact detector"""
    logger.info("Training Spatial Artifact Detector")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = AIDetectionDataset(
        data_dir, split='train',
        transform=train_transform,
        preprocessing_mode='spatial'
    )
    val_dataset = AIDetectionDataset(
        data_dir, split='val',
        transform=val_transform,
        preprocessing_mode='spatial'
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = SpatialArtifactDetector(model_name='efficientnet_b3', pretrained=True)

    # Train
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=50, learning_rate=1e-4,
        device=device, save_dir=save_dir,
        model_name='spatial_detector'
    )

    return history


def train_frequency_model(data_dir: str, save_dir: str, device: torch.device):
    """Train frequency domain analyzer"""
    logger.info("Training Frequency Domain Analyzer")

    # Datasets (no augmentation for frequency domain)
    train_dataset = AIDetectionDataset(
        data_dir, split='train',
        preprocessing_mode='frequency'
    )
    val_dataset = AIDetectionDataset(
        data_dir, split='val',
        preprocessing_mode='frequency'
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = FrequencyDomainCNN()

    # Train
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=50, learning_rate=1e-3,
        device=device, save_dir=save_dir,
        model_name='frequency_detector'
    )

    return history


def train_noise_model(data_dir: str, save_dir: str, device: torch.device):
    """Train noise pattern detector"""
    logger.info("Training Noise Pattern Detector")

    # Datasets
    train_dataset = AIDetectionDataset(
        data_dir, split='train',
        preprocessing_mode='noise'
    )
    val_dataset = AIDetectionDataset(
        data_dir, split='val',
        preprocessing_mode='noise'
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = NoisePatternCNN()

    # Train
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=50, learning_rate=1e-3,
        device=device, save_dir=save_dir,
        model_name='noise_detector'
    )

    return history


def main():
    parser = argparse.ArgumentParser(description='Train AI detection models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['spatial', 'frequency', 'noise', 'all'],
                       help='Model to train')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Train models
    if args.model == 'spatial' or args.model == 'all':
        train_spatial_model(args.data_dir, args.save_dir, device)

    if args.model == 'frequency' or args.model == 'all':
        train_frequency_model(args.data_dir, args.save_dir, device)

    if args.model == 'noise' or args.model == 'all':
        train_noise_model(args.data_dir, args.save_dir, device)

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
