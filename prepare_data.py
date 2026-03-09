"""
Data preparation: split raw images into train/val/test.
Processes only the 4 oilseed classes (soybean, sunflower, coconut, peanut)
defined in config.CLASS_NAMES. Other classes are ignored.
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

def get_image_files(class_dir):
    """Get all image files from a class directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(class_dir.glob(f'*{ext}')))
    return sorted(image_files)

def split_dataset():
    """Split dataset into train/val/test sets."""
    print("Preparing dataset splits...")
    
    # Create dataset directory structure
    for split in ['train', 'val', 'test']:
        split_dir = config.DATA_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_name in config.CLASS_NAMES_ORDERED:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for french_name, english_name in config.CLASS_NAMES.items():
        class_dir = config.RAW_DATA_DIR / french_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist. Skipping...")
            continue
        
        print(f"\nProcessing {french_name} -> {english_name}...")
        image_files = get_image_files(class_dir)
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        print(f"Found {len(image_files)} images")
        
        # First split: train (70%) and temp (30%)
        train_files, temp_files = train_test_split(
            image_files,
            test_size=(config.VAL_RATIO + config.TEST_RATIO),
            random_state=42,
            shuffle=True
        )
        
        # Second split: val (15%) and test (15%) from temp
        val_files, test_files = train_test_split(
            temp_files,
            test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
            random_state=42,
            shuffle=True
        )
        
        # Copy files to respective directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_dir = config.DATA_DIR / split_name / english_name
            print(f"  Copying {len(files)} images to {split_name}/...")
            for img_file in tqdm(files, desc=f"  {split_name}", leave=False):
                shutil.copy2(img_file, split_dir / img_file.name)
    
    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print("="*50)
    
    # Print statistics
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} set:")
        split_dir = config.DATA_DIR / split
        for class_name in config.CLASS_NAMES_ORDERED:
            class_dir = split_dir / class_name
            count = len(list(class_dir.glob('*'))) if class_dir.exists() else 0
            print(f"  {class_name}: {count} images")

if __name__ == "__main__":
    split_dataset()



