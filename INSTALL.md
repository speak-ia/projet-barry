# Guide d'Installation

## Option 1: Installation avec Conda (Recommandé)

Si vous utilisez Anaconda/Miniconda, installez PyTorch via conda :

```bash
# Installer PyTorch avec CUDA (si vous avez un GPU NVIDIA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# OU installer PyTorch pour CPU seulement
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Ensuite, installez les autres dépendances :
```bash
conda install numpy pillow matplotlib seaborn scikit-learn tqdm pandas
```

## Option 2: Installation avec pip (dans un environnement virtuel)

Créez un nouvel environnement virtuel pour éviter les conflits :

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur macOS/Linux:
source venv/bin/activate
# Sur Windows:
# venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Option 3: Installation manuelle

Si vous avez des problèmes de permissions, installez package par package :

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy
pip install Pillow
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tqdm
pip install pandas
```

## Vérifier l'installation

Après l'installation, vérifiez que tout fonctionne :

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Si vous voyez la version de PyTorch, l'installation est réussie !

## Note sur CUDA

- Si vous avez un GPU NVIDIA, installez la version CUDA de PyTorch pour un entraînement plus rapide
- Si vous n'avez pas de GPU, installez la version CPU (plus simple)
- Le code fonctionnera dans les deux cas, mais l'entraînement sera plus lent sur CPU

## Erreur SSL (CERTIFICATE_VERIFY_FAILED) sur macOS

Si, au lancement de `python train.py`, vous voyez une erreur du type **SSL: CERTIFICATE_VERIFY_FAILED** lors du téléchargement des poids MobileNet, le projet applique automatiquement un contournement dans `model.py`. Relancez simplement :

```bash
python train.py
```

**Pour corriger définitivement les certificats SSL** (Python installé depuis python.org) :

- Exécutez une fois : **Applications → Python 3.x → Install Certificates.command**
- Ou dans le terminal :  
  `pip install certifi` puis  
  `export SSL_CERT_FILE=$(python -m certifi)` (à ajouter dans votre `~/.zshrc` si besoin)

---

## Lancer l'entraînement (4 classes oléagineuses)

Le modèle détecte uniquement **4 plantes oléagineuses** : Soja, Tournesol, Cocotier, Arachide.

### 1. Préparer les images

Placez vos images brutes dans des dossiers nommés **exactement** comme dans `config.py` (noms français) :

- **Soja** → dossier `Soja/`
- **Tournesol** → dossier `Tournesol/`
- **Cocotier** → dossier `Cocotier/`
- **Arachide** → dossier `Arachide/`

Ces dossiers doivent être à la **racine du projet** (à côté de `train.py`), ou alors modifiez `RAW_DATA_DIR` dans `config.py` pour pointer vers le dossier qui les contient.

Exemple de structure :

```
BARRY/
├── Soja/           ← images de soja
│   ├── img1.jpg
│   └── ...
├── Tournesol/
├── Cocotier/
├── Arachide/
├── train.py
├── config.py
└── ...
```

### 2. Créer les jeux train / val / test

Depuis la racine du projet :

```bash
python prepare_data.py
```

Cela crée les dossiers `dataset/train`, `dataset/val`, `dataset/test` et répartit les images (70 % train, 15 % val, 15 % test) pour chaque classe.

### 3. Lancer l'entraînement

```bash
python train.py
```

- Le modèle utilisé est **MobileNetV2** (configurable dans `config.py` : `MODEL_NAME`).
- Le modèle entraîné est enregistré dans : **`models/oilseed_mobilenet.pth`**.
- Les courbes (loss, accuracy) et l’historique sont sauvegardés dans **`results/`**.

### 4. (Optionnel) Évaluer le modèle

```bash
python evaluate.py
```

Utilise par défaut `models/oilseed_mobilenet.pth` et écrit les métriques dans `results/`.

### 5. Prédiction sur une image

```bash
python inference.py chemin/vers/image.jpg
```

Les messages de prédiction sont en **français** (ex. « Ce n'est pas une plante oléagineuse » si la confiance est sous le seuil).



