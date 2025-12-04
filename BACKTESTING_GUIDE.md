# Backtesting Guide for Verity

## Overview

The current ML detection system uses **placeholder heuristics** instead of trained models. This guide explains how to backtest the system and prepare for real ML model integration.

## Current ML Detection Issues

### What's Actually Running (Not Real ML):
1. **Spatial Detection**: Basic Laplacian variance + edge density checks
2. **Frequency Analysis**: Simple FFT ratio calculations
3. **Problem Areas**: Placeholder code with TODOs

### Why It Fails on AI-Generated Images:
- No trained neural networks
- Simple heuristics can't detect modern AI artifacts
- Lacks semantic understanding
- No learned features from AI generators

---

## Setting Up Backtesting

### 1. Prepare Your Test Dataset

Create a directory structure:
```
test_datasets/
└── ai_detection_v1/
    ├── manifest.json
    ├── authentic/
    │   ├── camera_photo_1.jpg
    │   ├── camera_photo_2.jpg
    │   └── ...
    └── ai_generated/
        ├── midjourney_1.png
        ├── stable_diffusion_1.png
        ├── dalle_1.png
        └── ...
```

### 2. Create Manifest File

Example `manifest.json`:
```json
{
  "dataset_name": "AI Detection Benchmark v1",
  "version": "1.0.0",
  "created": "2025-12-03",
  "test_cases": [
    {
      "file_path": "authentic/camera_photo_1.jpg",
      "true_label": "authentic",
      "metadata": {
        "source": "Canon EOS R5",
        "category": "nature",
        "has_c2pa": false
      }
    },
    {
      "file_path": "ai_generated/midjourney_1.png",
      "true_label": "ai_generated",
      "metadata": {
        "generator": "Midjourney",
        "version": "v6",
        "prompt": "...",
        "category": "portrait"
      }
    }
  ]
}
```

### 3. Run Backtest

```bash
# From project root
python scripts/backtest.py
```

Or programmatically:
```python
from scripts.backtest import BacktestRunner

runner = BacktestRunner(dataset_dir="./test_datasets/ai_detection_v1")
test_cases = runner.load_dataset("manifest.json")
results = await runner.run_backtest(test_cases)
metrics = runner.calculate_metrics(results)
runner.save_results(results, metrics, "baseline_run")
```

---

## Key Metrics to Track

### Binary Classification Metrics:
- **Accuracy**: Overall correctness
- **Precision**: % of AI predictions that were correct (low FP)
- **Recall**: % of actual AI images detected (low FN)
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate (FPR)**: Authentic images marked as AI
- **False Negative Rate (FNR)**: AI images marked as authentic

### Business Metrics:
- **FNR is critical** - Missing AI content is worse than false alarms
- **Per-vertical performance** - Different industries have different thresholds
- **Processing time** - Must stay under 3 seconds for images

### Expected Current Performance:
```
Current Heuristics (Baseline):
├── Accuracy:  ~50-60% (essentially random)
├── Precision: ~40-50%
├── Recall:    ~30-40%
└── F1 Score:  ~35-45%
```

### Target Performance with Real Models:
```
With Trained Models:
├── Accuracy:  >85%
├── Precision: >80%
├── Recall:    >90% (prioritize catching AI)
└── F1 Score:  >85%
```

---

## Recommended Test Datasets

### Public Datasets:

1. **CIFAKE** (CIFAR-10 Real vs AI)
   - 100K real images, 100K AI-generated
   - Good for baseline testing
   - [Kaggle Link](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

2. **DiffusionDB**
   - Large-scale text-to-image dataset
   - Various prompts and generators
   - [HuggingFace](https://huggingface.co/datasets/poloclub/diffusiondb)

3. **Real vs AI Art**
   - Artistic images (paintings, digital art)
   - Mix of styles
   - Harder test case

4. **ForenSynths**
   - Academic dataset for synthetic image detection
   - Multiple GAN architectures

### Creating Your Own Dataset:

1. **Collect Authentic Images**:
   - Camera photos with EXIF data
   - Screenshots with metadata
   - Various sources (phones, DSLRs, scans)

2. **Generate AI Images**:
   - Midjourney (various versions)
   - Stable Diffusion (1.5, 2.1, XL)
   - DALL-E 3
   - Adobe Firefly
   - Various prompts and styles

3. **Edge Cases**:
   - Heavily edited authentic photos
   - AI-generated + post-processed
   - Low-quality images
   - Specific domains (medical, legal, etc.)

---

## Integrating Real ML Models

### Step 1: Choose Models

Recommended ensemble:

```python
# Add to requirements.txt
timm==0.9.12           # For model architectures
transformers==4.37.0    # For CLIP/ViT
torch==2.1.2            # PyTorch
```

### Step 2: Download Pretrained Weights

```bash
# Example: Download UniversalFakeDetect
mkdir -p models/universal_fake_detect
cd models/universal_fake_detect
wget https://github.com/.../universal_fake_detect.pth
```

### Step 3: Update ml_detector.py

Replace heuristics with:

```python
import torch
import timm
from transformers import CLIPProcessor, CLIPModel

class RealMLDetector:
    def __init__(self):
        # Load models
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=2)
        self.resnet.load_state_dict(torch.load('models/ai_detector_resnet.pth'))

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Ensemble weights
        self.weights = {'resnet': 0.5, 'clip': 0.3, 'freq': 0.2}

    async def detect(self, image_path):
        # Run each model
        resnet_score = await self._run_resnet(image_path)
        clip_score = await self._run_clip(image_path)
        freq_score = await self._frequency_analysis(image_path)

        # Weighted ensemble
        final_score = (
            resnet_score * self.weights['resnet'] +
            clip_score * self.weights['clip'] +
            freq_score * self.weights['freq']
        )

        return final_score
```

### Step 4: Retrain Models on Your Data

```python
# Fine-tune on your domain-specific data
from torch.utils.data import DataLoader
from your_dataset import VerityDataset

dataset = VerityDataset(manifest_path="train_manifest.json")
dataloader = DataLoader(dataset, batch_size=32)

# Fine-tune
for epoch in range(10):
    for batch in dataloader:
        images, labels = batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Continuous Backtesting Pipeline

### Automated Testing:

```bash
# scripts/nightly_backtest.sh
#!/bin/bash

# Run backtests on multiple datasets
python scripts/backtest.py --dataset test_datasets/baseline_v1
python scripts/backtest.py --dataset test_datasets/hard_cases_v1
python scripts/backtest.py --dataset test_datasets/vertical_insurance
python scripts/backtest.py --dataset test_datasets/vertical_legal

# Compare to baseline
python scripts/compare_results.py --baseline baseline_run --current latest_run

# Alert if performance drops
python scripts/alert_if_degraded.py --threshold 0.05
```

### A/B Testing New Models:

```python
# Test model variants
variants = [
    {"name": "baseline", "model": "heuristics"},
    {"name": "resnet50", "model": "resnet50_v1"},
    {"name": "ensemble", "model": "ensemble_v2"},
]

for variant in variants:
    results = await runner.run_backtest(test_cases, model=variant["model"])
    metrics = runner.calculate_metrics(results)
    print(f"{variant['name']}: F1={metrics['overall']['f1_score']:.2%}")
```

---

## Monitoring in Production

### Log All Predictions:

```python
# In production, log every verification
await log_verification(
    verification_id=ver_id,
    prediction=result.risk_category,
    trust_score=result.trust_score,
    processing_time=result.processing_time_ms,
    model_version="ensemble_v2.1"
)
```

### Periodic Revalidation:

1. **Sample verified images** (human review)
2. **Run backtest monthly** on production logs
3. **Retrain quarterly** with new data
4. **Track model drift** - performance degradation over time

### Red Flags:

- FNR increasing (missing more AI images)
- Processing time creeping up
- Confidence scores shifting
- New AI generators not being caught

---

## Next Steps

1. **Download a test dataset** (start with CIFAKE)
2. **Run baseline backtest** with current heuristics
3. **Document current performance** (likely 50-60% accuracy)
4. **Research and select** appropriate ML models
5. **Integrate and retrain** on your domain
6. **Re-run backtest** - should see 85%+ accuracy
7. **Set up continuous testing** pipeline

## Questions to Answer:

1. **What's your acceptable FNR?** (Missing AI images)
2. **What's your acceptable FPR?** (False alarms on real photos)
3. **Domain-specific needs?** (Medical, legal, insurance)
4. **Processing time budget?** (Real-time vs batch)
5. **Model size constraints?** (Edge deployment, cloud)

---

## Resources

- **Papers**:
  - "CNNSpot: Exposing Deep Generative Models" (2020)
  - "DIRE: Diffusion-Generated Image Detection" (2023)
  - "UniversalFakeDetect" (2023)

- **Datasets**:
  - CIFAKE (Kaggle)
  - DiffusionDB (HuggingFace)
  - ForenSynths (Academic)

- **Tools**:
  - Weights & Biases (experiment tracking)
  - MLflow (model versioning)
  - TensorBoard (training visualization)

