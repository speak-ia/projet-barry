"""
Inference script for oilseed plant classification.
Retourne la classe prédite si confiance >= seuil, sinon un message en français.
"""
import torch
from PIL import Image
from pathlib import Path
import argparse
import sys

import config
from model import get_model
from dataset import get_transforms

# Message de sortie en français
NOT_OILSEED_MESSAGE = config.MESSAGE_NOT_OILSEED_FR


def predict_image(
    image_path,
    model_path=None,
    top_k=3,
    confidence_threshold=None,
):
    """
    Predict the class of a single image (oilseed or not).

    Args:
        image_path: Path to the image file (or PIL Image)
        model_path: Path to saved model (default: models/oilseed_mobilenet.pth)
        top_k: Number of top predictions to return when valid
        confidence_threshold: Min confidence to accept as oilseed (default: config.CONFIDENCE_THRESHOLD)

    Returns:
        dict with:
          - is_oilseed: bool
          - plant: str (class name or "not_oilseed")
          - confidence: float (0-1) or None if not_oilseed
          - message: str (NOT_OILSEED_MESSAGE if below threshold)
          - predictions: list of {class, confidence} (optional, for top_k)
    """
    threshold = confidence_threshold if confidence_threshold is not None else config.CONFIDENCE_THRESHOLD
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = config.MODEL_DIR / config.OILSEED_MODEL_FILENAME

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model = get_model(
        model_name=checkpoint.get("model_name", config.MODEL_NAME),
        pretrained=False,
        num_classes=checkpoint.get("num_classes", config.NUM_CLASSES),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load image (path or PIL)
    if isinstance(image_path, (str, Path)):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    else:
        image = image_path.convert("RGB") if hasattr(image_path, "convert") else image_path

    transform = get_transforms("val")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, min(top_k, config.NUM_CLASSES))

    best_idx = top_indices[0][0].item()
    best_conf = top_probs[0][0].item()

    if best_conf < threshold:
        return {
            "is_oilseed": False,
            "plant": "not_oilseed",
            "confidence": None,
            "message": NOT_OILSEED_MESSAGE,
            "predictions": [],
        }

    predictions = []
    for i in range(top_indices.shape[1]):
        idx = top_indices[0][i].item()
        conf = top_probs[0][i].item()
        predictions.append({
            "class": config.CLASS_NAMES_ORDERED[idx],
            "confidence": conf,
        })

    return {
        "is_oilseed": True,
        "plant": config.CLASS_NAMES_ORDERED[best_idx],
        "confidence": best_conf,
        "message": None,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict oilseed plant class from image")
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to show")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )

    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)

    try:
        result = predict_image(
            args.image_path,
            model_path=args.model,
            top_k=args.top_k,
            confidence_threshold=args.threshold,
        )

        print("\n" + "=" * 60)
        print("RÉSULTATS DE PRÉDICTION")
        print("=" * 60)
        print(f"\nImage : {args.image_path}")

        if result["is_oilseed"]:
            print(f"\nPlante : {result['plant']}")
            print(f"Confiance : {result['confidence']*100:.2f} %")
            if result["predictions"]:
                print("\nTop prédictions :")
                for i, p in enumerate(result["predictions"], 1):
                    print(f"  {i}. {p['class']:15s} : {p['confidence']*100:6.2f} %")
        else:
            print(f"\n{result['message']}")

        print("=" * 60)
        return result

    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
