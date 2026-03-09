"""
Flask API for oilseed plant classification.
POST /predict accepts multipart/form-data (image file) for Flutter integration.
Déployable sur Render : peut démarrer sans modèle (en attendant la fin de l'entraînement).
"""
import logging
import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from pathlib import Path
import sys
from datetime import datetime

import config
from model import get_model
from dataset import get_transforms
from inference import predict_image

# Message de sortie en français pour les réponses API
MESSAGE_NOT_OILSEED = config.MESSAGE_NOT_OILSEED_FR

# ---------------------------------------------------------------------------
# Logging: log predictions for monitoring
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model (loaded once)
model = None
device = None
transform = None


def load_model(model_path=None, required=False):
    """
    Charge le modèle oilseed (oilseed_mobilenet.pth).
    Si required=False et fichier absent, ne lève pas d'exception (pour déploiement Render).
    Returns:
        True si le modèle est chargé, False sinon.
    """
    global model, device, transform

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = config.MODEL_DIR / config.OILSEED_MODEL_FILENAME
        # Render Secret Files : fichier uploadé disponible dans /etc/secrets/<filename>
        if not Path(model_path).exists():
            secret_path = Path("/etc/secrets") / config.OILSEED_MODEL_FILENAME
            if secret_path.exists():
                model_path = secret_path
        # Fallback : chemin relatif au répertoire de travail (Render peut lancer depuis ailleurs)
        if not Path(model_path).exists():
            cwd_path = Path(os.getcwd()) / "models" / config.OILSEED_MODEL_FILENAME
            if cwd_path.exists():
                model_path = cwd_path

    path_resolved = Path(model_path).resolve()
    exists = path_resolved.exists()
    logger.info("Model path: %s (exists: %s)", path_resolved, exists)
    if not exists and config.MODEL_DIR.exists():
        try:
            logger.info("Contents of models dir: %s", list(config.MODEL_DIR.iterdir()))
        except Exception as e:
            logger.warning("Could not list models dir: %s", e)

    if not Path(model_path).exists():
        if required:
            raise FileNotFoundError(f"Model not found at {model_path}")
        logger.warning("Model not found at %s — API will start without model. /predict will return 503.", model_path)
        model = None
        transform = None
        return False

    logger.info("Loading model from %s", model_path)
    checkpoint = torch.load(model_path, map_location=device)

    model = get_model(
        model_name=checkpoint.get("model_name", config.MODEL_NAME),
        pretrained=False,
        num_classes=checkpoint.get("num_classes", config.NUM_CLASSES),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = get_transforms("val")
    logger.info("Model loaded on %s", device)
    return True


@app.route("/", methods=["GET"])
def index():
    """Page d'accueil : infos et liens vers les endpoints."""
    base = request.url_root.rstrip("/")
    return jsonify({
        "service": "API Oilseed — détection plantes oléagineuses",
        "endpoints": {
            "health": f"{base}/health",
            "classes": f"{base}/classes",
            "predict": f"{base}/predict (POST, multipart/form-data: image)",
        },
        "model_loaded": model is not None,
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict oilseed plant from image.

    Input:
      - multipart/form-data with field "image" (file) — for Flutter
      - optional query param: threshold (float, default 0.6)

    Output JSON:
      - Si plante oléagineuse : {"plant": "sunflower", "confidence": 0.92}
      - Sinon : {"plant": "not_oilseed", "message": "Ce n'est pas une plante oléagineuse."}
    """
    if model is None:
        return jsonify({
            "error": "Modèle non chargé",
            "message": "Le modèle n'est pas encore disponible. Déployez models/oilseed_mobilenet.pth ou attendez la fin de l'entraînement.",
        }), 503

    # Confidence threshold (bonus: configurable via query param)
    try:
        threshold = float(request.args.get("threshold", config.CONFIDENCE_THRESHOLD))
    except (TypeError, ValueError):
        threshold = config.CONFIDENCE_THRESHOLD

    if "image" not in request.files and not request.is_json:
        return jsonify({"error": "No image provided. Use multipart/form-data with 'image' file."}), 400

    try:
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        elif request.is_json:
            data = request.get_json()
            if "image" in data:
                import base64
                image_data = data["image"]
                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
            elif "image_path" in data:
                image = Image.open(data["image_path"]).convert("RGB")
            else:
                return jsonify({"error": "No image data in JSON"}), 400
        else:
            return jsonify({"error": "Invalid request: send image as multipart file or base64 in JSON"}), 400

        result = predict_image(
            image,
            model_path=config.MODEL_DIR / config.OILSEED_MODEL_FILENAME,
            confidence_threshold=threshold,
        )

        # Log prediction (bonus)
        logger.info(
            "prediction | plant=%s | confidence=%s | is_oilseed=%s",
            result["plant"],
            result.get("confidence"),
            result["is_oilseed"],
        )

        if result["is_oilseed"]:
            return jsonify({
                "plant": result["plant"],
                "confidence": round(result["confidence"], 4),
            })
        else:
            return jsonify({
                "plant": "not_oilseed",
                "message": MESSAGE_NOT_OILSEED,
            })

    except Exception as e:
        logger.exception("Predict error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/classes", methods=["GET"])
def get_classes():
    """Return the 4 oilseed class names."""
    return jsonify({
        "classes": config.CLASS_NAMES_ORDERED,
        "num_classes": len(config.CLASS_NAMES_ORDERED),
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Oilseed classification API")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="Port (default: 5000 or env PORT for Render)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--require-model", action="store_true", help="Exit if model file not found")

    args = parser.parse_args()

    port = args.port or int(os.environ.get("PORT", 5000))

    try:
        load_model(args.model, required=args.require_model)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    logger.info("API starting on %s:%s | Model loaded: %s", args.host, port, model is not None)
    app.run(host=args.host, port=port, debug=args.debug)
