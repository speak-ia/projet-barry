"""
Evaluation script for oilseed plant classification model.
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import config
from model import get_model
from dataset import get_data_loaders

def evaluate_model(model_path=None, test_loader=None):
    """
    Evaluate the trained model on test set.
    
    Args:
        model_path: Path to saved model checkpoint
        test_loader: Test data loader (if None, will create one)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model (oilseed_mobilenet.pth)
    if model_path is None:
        model_path = config.MODEL_DIR / config.OILSEED_MODEL_FILENAME
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = get_model(
        model_name=checkpoint.get('model_name', config.MODEL_NAME),
        pretrained=False,
        num_classes=checkpoint.get('num_classes', config.NUM_CLASSES)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Get test loader
    if test_loader is None:
        _, _, test_loader = get_data_loaders()
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Classification report
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=config.CLASS_NAMES_ORDERED,
        output_dict=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Test Accuracy: {overall_acc*100:.2f}%")
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(config.CLASS_NAMES_ORDERED):
        print(f"  {class_name:15s}: {per_class_acc[i]*100:6.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=config.CLASS_NAMES_ORDERED
    ))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, config.CLASS_NAMES_ORDERED)
    
    # Save results
    results = {
        'overall_accuracy': float(overall_acc),
        'per_class_accuracy': {name: float(acc) for name, acc in zip(config.CLASS_NAMES_ORDERED, per_class_acc)},
        'classification_report': class_report,
        'confusion_matrix': cm.tolist()
    }
    
    results_path = config.RESULTS_DIR / "evaluation_results_oilseed.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {results_path}")
    
    return results

def plot_confusion_matrix(cm, class_names):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = config.RESULTS_DIR / "confusion_matrix_oilseed.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {cm_path}")
    plt.close()

if __name__ == "__main__":
    evaluate_model()



