"""
Utility functions for oilseed plant classification.
"""
import torch
import numpy as np
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from dataset import OilseedPlantDataset, get_transforms

def get_class_weights(data_dir):
    """
    Calculate class weights for handling imbalanced datasets.
    
    Args:
        data_dir: Path to training data directory
    
    Returns:
        Tensor of class weights
    """
    dataset = OilseedPlantDataset(data_dir, transform=get_transforms('train'))
    
    # Count samples per class
    label_counts = Counter(dataset.labels)
    total_samples = len(dataset.labels)
    
    # Calculate weights (inverse frequency)
    weights = []
    for i in range(config.NUM_CLASSES):
        count = label_counts.get(i, 1)  # Avoid division by zero
        weight = total_samples / (config.NUM_CLASSES * count)
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum() * config.NUM_CLASSES
    
    print("Class weights:")
    for i, class_name in enumerate(config.CLASS_NAMES_ORDERED):
        print(f"  {class_name:15s}: {weights[i]:.4f} (samples: {label_counts.get(i, 0)})")
    
    return torch.FloatTensor(weights)

def visualize_samples(data_dir, num_samples=8, save_path=None):
    """
    Visualize sample images from the dataset.
    
    Args:
        data_dir: Path to dataset directory
        num_samples: Number of samples to visualize per class
        save_path: Path to save the visualization (optional)
    """
    dataset = OilseedPlantDataset(data_dir, transform=get_transforms('val'))
    
    # Get samples from each class
    samples_per_class = num_samples // config.NUM_CLASSES
    if samples_per_class == 0:
        samples_per_class = 1
    
    fig, axes = plt.subplots(config.NUM_CLASSES, samples_per_class, 
                            figsize=(samples_per_class * 3, config.NUM_CLASSES * 3))
    
    if config.NUM_CLASSES == 1:
        axes = axes.reshape(1, -1)
    if samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    for class_idx in range(config.NUM_CLASSES):
        # Get indices for this class
        class_indices = [i for i, label in enumerate(dataset.labels) if label == class_idx]
        
        if not class_indices:
            continue
        
        # Sample random indices
        selected_indices = np.random.choice(
            class_indices, 
            size=min(samples_per_class, len(class_indices)), 
            replace=False
        )
        
        for col, idx in enumerate(selected_indices):
            image, label = dataset[idx]
            
            # Convert tensor to numpy for visualization
            image_np = image.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            
            ax = axes[class_idx, col] if samples_per_class > 1 else axes[class_idx]
            ax.imshow(image_np)
            ax.set_title(f"{config.CLASS_NAMES_ORDERED[label]}")
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def print_dataset_statistics():
    """Print statistics about the dataset."""
    print("="*60)
    print("Dataset Statistics")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_dir = config.DATA_DIR / split
        if not split_dir.exists():
            print(f"\n{split.upper()} set: Not found")
            continue
        
        print(f"\n{split.upper()} set:")
        total = 0
        for class_name in config.CLASS_NAMES_ORDERED:
            class_dir = split_dir / class_name
            count = len(list(class_dir.glob('*'))) if class_dir.exists() else 0
            total += count
            print(f"  {class_name:15s}: {count:5d} images")
        print(f"  {'Total':15s}: {total:5d} images")

if __name__ == "__main__":
    # Print dataset statistics
    print_dataset_statistics()
    
    # Visualize samples from training set
    if (config.DATA_DIR / 'train').exists():
        print("\nGenerating sample visualizations...")
        visualize_samples(
            config.DATA_DIR / 'train',
            num_samples=14,
            save_path=config.RESULTS_DIR / 'sample_images.png'
        )



