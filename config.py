"""
Configuration file for oilseed plant classification model.
Detects only: soybean, sunflower, coconut, peanut.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = BASE_DIR  # Current directory with class folders
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "train").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "val").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "test").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# OILSEED CLASSES ONLY (4 classes)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    "Soja": "soybean",
    "Tournesol": "sunflower",
    "Cocotier": "coconut",
    "Arachide": "peanut",
}

# Class names in order (for model output) — used by dataset and inference
CLASS_NAMES_ORDERED = [
    "soybean",
    "sunflower",
    "coconut",
    "peanut",
]

# Dataset split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model hyperparameters
NUM_CLASSES = 4
IMG_SIZE = 224  # Standard for MobileNet (ImageNet preprocessing)
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

# Inference: confidence below this → message "not oilseed" (French)
CONFIDENCE_THRESHOLD = 0.6
MESSAGE_NOT_OILSEED_FR = "Ce n'est pas une plante oléagineuse."

# Device configuration
DEVICE = "cuda"

# Model architecture: MobileNet for mobile-friendly oilseed detection
MODEL_NAME = "mobilenet_v2"  # Options: "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"

# Saved model filename (oilseed MobileNet)
OILSEED_MODEL_FILENAME = "oilseed_mobilenet.pth"
