"""
Dataset for oilseed plant classification.
Handles only 4 classes: soybean, sunflower, coconut, peanut.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import config

# Only these 4 oilseed classes are loaded (defined in config.CLASS_NAMES_ORDERED)
OILSEED_CLASSES = ["soybean", "sunflower", "coconut", "peanut"]


class OilseedPlantDataset(Dataset):
    """
    Dataset for oilseed plant classification (4 classes only).
    Expects data_dir to contain one subfolder per class in CLASS_NAMES_ORDERED.

    Args:
        data_dir: Path to dataset directory (train/val/test)
        transform: Optional transform to be applied on images
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        # Load only the 4 oilseed classes (soybean, sunflower, coconut, peanut)
        for class_idx, class_name in enumerate(config.CLASS_NAMES_ORDERED):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.webp'}
            for ext in image_extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(mode="train"):
    """
    Get data transforms for training or validation.
    Optimized for MobileNet: ImageNet normalization, 224x224 input.

    Args:
        mode: "train" (augmentation) or "val"/"test" (no augmentation)

    Returns:
        torchvision.transforms.Compose
    """
    # ImageNet stats — required for MobileNet/transfer learning
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

def get_data_loaders(batch_size=None, num_workers=4):
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        batch_size: Batch size (defaults to config.BATCH_SIZE)
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Create datasets
    train_dataset = OilseedPlantDataset(
        config.DATA_DIR / 'train',
        transform=get_transforms('train')
    )
    
    val_dataset = OilseedPlantDataset(
        config.DATA_DIR / 'val',
        transform=get_transforms('val')
    )
    
    test_dataset = OilseedPlantDataset(
        config.DATA_DIR / 'test',
        transform=get_transforms('test')
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


