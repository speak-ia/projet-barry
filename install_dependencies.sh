#!/bin/bash
# Script pour installer les dépendances avec pip dans l'environnement conda

echo "Activation de l'environnement conda..."
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || conda activate base

echo "Installation de PyTorch et torchvision..."
pip install torch torchvision

echo "Installation des autres dépendances..."
pip install numpy Pillow matplotlib seaborn scikit-learn tqdm pandas

echo "Vérification de l'installation..."
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Installation terminée ! Vous pouvez maintenant lancer: python train.py"



