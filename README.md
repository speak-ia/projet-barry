# Oilseed Plant Classification with PyTorch

A complete deep learning pipeline for classifying 7 types of oilseed plants from images using transfer learning with PyTorch.

## 🌱 Classes

1. **Sunflower** (Tournesol)
2. **Peanut/Groundnut** (Arachide)
3. **Coconut Tree** (Cocotier)
4. **Soybean** (Soja)
5. **Sesame** (Sésame)
6. **Cotton** (Coton)
7. **Oil Palm** (Palmier à huile)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended) or CPU

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Split your raw images into train/val/test sets:

```bash
python prepare_data.py
```

This will create the following structure:
```
dataset/
├── train/
│   ├── sunflower/
│   ├── peanut/
│   ├── coconut/
│   ├── soybean/
│   ├── sesame/
│   ├── cotton/
│   └── oil_palm/
├── val/
│   └── [same structure]
└── test/
    └── [same structure]
```

### 3. Train the Model

```bash
python train.py
```

The training script will:
- Load and augment the data
- Train a ResNet50 model (or EfficientNet if configured)
- Save the best model based on validation accuracy
- Generate training curves
- Implement early stopping

### 4. Evaluate the Model

```bash
python evaluate.py
```

This will:
- Load the best model
- Evaluate on the test set
- Generate confusion matrix
- Print classification report with per-class metrics

### 5. Run Inference on New Images

```bash
python inference.py path/to/image.jpg
```

For top-3 predictions:
```bash
python inference.py path/to/image.jpg --top_k 3
```

## 📁 Project Structure

```
BARRY/
├── config.py              # Configuration file
├── prepare_data.py        # Dataset preparation script
├── dataset.py             # Dataset and DataLoader classes
├── model.py               # Model architecture
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── models/                # Saved model checkpoints
├── results/               # Training curves, confusion matrices, etc.
└── dataset/               # Processed dataset (created by prepare_data.py)
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Model Architecture**: Change `MODEL_NAME` to `'resnet50'`, `'efficientnet_b0'`, or `'efficientnet_b3'`
- **Hyperparameters**: Batch size, learning rate, number of epochs
- **Image Size**: Default is 224x224 (standard for ResNet/EfficientNet)
- **Data Split**: Train/val/test ratios (default: 70/15/15)

## 🎯 Model Architecture

The pipeline uses **transfer learning** with pretrained models:

- **ResNet50**: Deep residual network, excellent for general image classification
- **EfficientNet-B0/B3**: Efficient architecture with better accuracy-to-parameters ratio

The final classification layer is replaced to output 7 classes. The backbone is fine-tuned on your agricultural dataset.

## 📊 Data Augmentation

Training augmentations include:
- Random horizontal/vertical flips
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations
- Random crop with resize

These augmentations help the model generalize to different:
- Lighting conditions
- Plant orientations
- Camera angles
- Field conditions

## 📈 Training Features

- **Early Stopping**: Prevents overfitting by stopping when validation loss doesn't improve
- **Learning Rate Scheduling**: Reduces LR when validation loss plateaus
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Training Curves**: Visualizes loss and accuracy over epochs
- **GPU Support**: Automatically uses CUDA if available

## 🔍 Evaluation Metrics

The evaluation script provides:
- Overall test accuracy
- Per-class accuracy
- Confusion matrix (visualized)
- Classification report (precision, recall, F1-score)

## 💡 Best Practices

### Dataset Size Recommendations

- **Minimum**: 100-200 images per class for transfer learning
- **Recommended**: 500+ images per class for robust performance
- **Ideal**: 1000+ images per class with diverse conditions

### Image Resolution

- **Minimum**: 224x224 pixels (standard input size)
- **Recommended**: 256x256 or higher (will be resized to 224x224)
- **Aspect Ratio**: Square images work best, but the model handles rectangular images

### Handling Class Imbalance

If your dataset has imbalanced classes:
1. Use **weighted loss** (modify `train.py` to add class weights)
2. Use **oversampling** for minority classes
3. Use **focal loss** instead of CrossEntropyLoss

### Common Pitfalls in Agricultural Image Classification

1. **Seasonal Variation**: Include images from different growth stages
2. **Lighting Conditions**: Augment with various brightness/contrast
3. **Background Clutter**: Consider background removal or object detection
4. **Similar Species**: Some plants look similar; ensure sufficient training data
5. **Field vs. Lab Images**: Train on images similar to deployment conditions

## 🔄 Extending to Object Detection

To extend this to **object detection** (detecting plants in images with bounding boxes):

1. **YOLO (You Only Look Once)**:
   - Use YOLOv5 or YOLOv8 from Ultralytics
   - Convert classification dataset to YOLO format (bounding box annotations)
   - Train on object detection task

2. **Faster R-CNN**:
   - Use torchvision's Faster R-CNN
   - Requires bounding box annotations (COCO or Pascal VOC format)
   - More accurate but slower than YOLO

3. **Dataset Requirements**:
   - Need bounding box annotations for each plant
   - Tools: LabelImg, CVAT, or Roboflow

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `IMG_SIZE` (e.g., 224 → 192)
- Use gradient accumulation

### Poor Validation Accuracy
- Check for data leakage (same images in train/val)
- Increase dataset size
- Try different model architectures
- Adjust learning rate

### Training is Slow
- Use GPU (CUDA)
- Increase `num_workers` in DataLoader
- Use mixed precision training (FP16)

## 📝 Example Usage

### Complete Workflow

```bash
# 1. Prepare data
python prepare_data.py

# 2. Train model
python train.py

# 3. Evaluate
python evaluate.py

# 4. Predict on new image
python inference.py test_image.jpg
```

### Custom Model Path

```bash
python inference.py image.jpg --model models/custom_model.pth
```

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements!

---

**Note**: This pipeline is designed for single-label, multiclass classification. For multi-label classification (one image can have multiple classes), modify the loss function and output layer accordingly.

