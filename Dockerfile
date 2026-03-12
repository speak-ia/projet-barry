# API Oilseed — déploiement Railway / Cloud Run (build léger pour éviter timeout)
# PyTorch CPU uniquement = build ~5–10 min au lieu de 25+ min.
FROM python:3.11-slim

WORKDIR /app

# PyTorch CPU uniquement (beaucoup plus rapide que torch+CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dépendances API (sans matplotlib, sklearn, onnx, etc.)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

# Le modèle doit être dans models/oilseed_mobilenet.pth (inclus dans le build ou monté)
ENV PORT=8080
EXPOSE 8080

# Cloud Run et Railway fournissent PORT en variable d'environnement.
# CMD en forme "shell" pour que ${PORT} soit bien développé au démarrage.
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 120 api_server:app"]
