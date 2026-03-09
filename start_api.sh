#!/bin/bash
# Script pour démarrer l'API d'inference

echo "=========================================="
echo "Démarrage de l'API d'inference"
echo "=========================================="

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "Erreur: Python 3 n'est pas installé"
    exit 1
fi

# Vérifier si les dépendances sont installées
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installation des dépendances..."
    pip install -r requirements.txt
fi

# Vérifier si le modèle existe
MODEL_PATH="models/best_model_resnet50.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Attention: Le modèle $MODEL_PATH n'existe pas"
    echo "Assurez-vous d'avoir entraîné le modèle d'abord"
    exit 1
fi

# Démarrer le serveur
echo ""
echo "Démarrage du serveur sur http://0.0.0.0:5000"
echo "Pour tester: curl http://localhost:5000/health"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""

python3 api_server.py --host 0.0.0.0 --port 5000

