"""
Convert PyTorch model to ONNX format for mobile deployment.
ONNX models can be used with onnxruntime in Flutter.
"""
import torch
import torch.onnx
from pathlib import Path
import argparse
import sys

import config
from model import get_model

def convert_to_onnx(model_path=None, output_path=None, opset_version=11):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model checkpoint
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (default: 11 for compatibility)
    """
    device = torch.device('cpu')  # ONNX export should be on CPU
    
    # Load model (oilseed_mobilenet.pth)
    if model_path is None:
        model_path = config.MODEL_DIR / config.OILSEED_MODEL_FILENAME
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}")
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
    
    # Set output path
    if output_path is None:
        output_path = config.MODEL_DIR / "oilseed_mobilenet.onnx"
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
    
    # Export to ONNX
    print(f"Converting model to ONNX format...")
    print(f"Output path: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model successfully converted to ONNX!")
    print(f"  File saved at: {output_path}")
    print(f"\nTo use in Flutter:")
    print(f"  1. Add onnxruntime package to your Flutter project")
    print(f"  2. Load the ONNX model: {output_path}")
    print(f"  3. Preprocess images with the same transforms (resize to {config.IMG_SIZE}x{config.IMG_SIZE}, normalize)")
    
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model', type=str, default=None, help='Path to PyTorch model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Path to save ONNX model')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    
    args = parser.parse_args()
    
    try:
        convert_to_onnx(args.model, args.output, args.opset)
    except Exception as e:
        print(f"Error converting model: {e}")
        sys.exit(1)

