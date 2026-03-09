# Guide d'Intégration Flutter

Ce guide explique comment intégrer le modèle de classification de plantes oléagineuses dans votre application Flutter.

## Options d'Intégration

Il existe deux principales approches pour intégrer le modèle dans Flutter :

### Option 1: API REST

Cette approche utilise un serveur Flask qui expose une API REST. L'application Flutter envoie des images au serveur et reçoit les prédictions.

**Avantages:**
- ✅ Simple à mettre en place
- ✅ Pas besoin de modifier le modèle
- ✅ Facile à mettre à jour (juste mettre à jour le serveur)
- ✅ Fonctionne sur iOS et Android

**Inconvénients:**
- ❌ Nécessite une connexion Internet
- ❌ Nécessite un serveur en cours d'exécution

### Option 2: ONNX Runtime ⭐ (Recommandé pour déploiement mobile)

Cette approche convertit le modèle PyTorch en format ONNX et l'utilise directement dans l'application Flutter avec onnxruntime.

**Avantages:**
- ✅ Fonctionne hors ligne (aucune connexion Internet requise)
- ✅ Plus rapide (pas de latence réseau)
- ✅ Pas besoin de serveur
- ✅ Expérience utilisateur optimale
- ✅ Idéal pour les applications mobiles

**Inconvénients:**
- ❌ Plus complexe à mettre en place initialement
- ❌ Taille de l'application plus grande (~50-100 MB)
- ❌ Nécessite des packages natifs

---

## Option 2: Utilisation d'ONNX Runtime ⭐ (Recommandé pour mobile)

Cette méthode permet d'exécuter le modèle directement sur l'appareil mobile, sans nécessiter de connexion Internet ni de serveur.

### Étape 1: Convertir le modèle en ONNX

Avant de pouvoir utiliser le modèle dans Flutter, vous devez convertir le modèle PyTorch (.pth) en format ONNX (.onnx).

**Conversion du modèle :**

```bash
# Depuis le répertoire du projet
python convert_to_onnx.py --model models/best_model_resnet50.pth
```

Cela créera un fichier `models/model_resnet50.onnx`.

**Options de conversion :**

```bash
# Spécifier un modèle différent
python convert_to_onnx.py --model models/final_model_resnet50.pth

# Spécifier le chemin de sortie
python convert_to_onnx.py --model models/best_model_resnet50.pth --output models/my_model.onnx

# Changer la version Opset (par défaut: 11, compatible avec la plupart des runtimes)
python convert_to_onnx.py --model models/best_model_resnet50.pth --opset 13
```

**Vérification :**

Après la conversion, vérifiez que le fichier ONNX a été créé :

```bash
ls -lh models/model_resnet50.onnx
```

Le fichier devrait faire environ 50-100 MB (selon le modèle).

**Note:** Le modèle ONNX sera optimisé pour l'inférence mobile et fonctionnera sur iOS et Android. Il utilise des axes dynamiques pour le batch_size, ce qui permet une flexibilité lors de l'inférence.

### Étape 2: Configurer le projet Flutter

#### 2.1 Ajouter les dépendances

Dans votre `pubspec.yaml` :

```yaml
dependencies:
  flutter:
    sdk: flutter
  onnxruntime: ^1.15.0  # Runtime ONNX pour Flutter
  image_picker: ^1.0.4  # Pour sélectionner des images
  image: ^4.0.0         # Pour le preprocessing des images
```

Puis exécutez :
```bash
flutter pub get
```

#### 2.2 Ajouter le modèle ONNX aux assets

1. Créez le dossier `assets/models/` dans votre projet Flutter
2. Copiez `models/model_resnet50.onnx` dans `assets/models/`

Ajoutez dans `pubspec.yaml` :
```yaml
flutter:
  assets:
    - assets/models/model_resnet50.onnx
```

#### 2.3 Configurer les permissions

**Android** (`android/app/src/main/AndroidManifest.xml`) :
```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
```

**iOS** (`ios/Runner/Info.plist`) :
```xml
<key>NSPhotoLibraryUsageDescription</key>
<string>Besoin d'accéder aux photos pour classifier les plantes</string>
<key>NSCameraUsageDescription</key>
<string>Besoin d'accéder à la caméra pour prendre des photos</string>
```

### Étape 3: Implémenter le classificateur ONNX

1. Créez le dossier `lib/services/` dans votre projet Flutter
2. Copiez le fichier `flutter_onnx_example.dart` dans `lib/services/onnx_plant_classifier.dart`

Le service inclut :
- ✅ Chargement du modèle ONNX depuis les assets
- ✅ Preprocessing des images (resize à 224x224, normalisation ImageNet)
- ✅ Inférence ONNX avec gestion des erreurs
- ✅ Postprocessing (softmax, tri par confiance)
- ✅ Mapping des classes avec noms français et anglais
- ✅ Support pour fichiers image et octets bruts
- ✅ Gestion propre des ressources (dispose)

**Structure du code :**
- `ONNXPlantClassifier` : Classe principale pour l'inférence
- `Prediction` : Classe pour représenter une prédiction
- `PredictionResult` : Classe pour représenter tous les résultats

### Étape 4: Utiliser le classificateur dans votre application

**Exemple simple :**

```dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'services/onnx_plant_classifier.dart';

class MyWidget extends StatefulWidget {
  @override
  _MyWidgetState createState() => _MyWidgetState();
}

class _MyWidgetState extends State<MyWidget> {
  final _classifier = ONNXPlantClassifier();
  bool _modelLoaded = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await _classifier.loadModel();
    setState(() => _modelLoaded = true);
  }

  Future<void> _predict() async {
    final image = await ImagePicker().pickImage(source: ImageSource.camera);
    if (image != null) {
      final result = await _classifier.predictFromFile(image.path);
      
      print('Top Prediction: ${result.topPrediction.classFr}');
      print('Confidence: ${result.topPrediction.confidence}%');
      
      // Afficher toutes les prédictions
      for (var pred in result.predictions) {
        print('${pred.classFr}: ${pred.confidence.toStringAsFixed(2)}%');
      }
    }
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: _modelLoaded
          ? ElevatedButton(
              onPressed: _predict,
              child: Text('Prendre une photo'),
            )
          : CircularProgressIndicator(),
      ),
    );
  }
}
```

**Exemple complet avec UI :** Voir le code commenté à la fin du fichier `flutter_onnx_example.dart` pour un exemple complet avec interface utilisateur.

### Étape 5: Tester l'application

1. **Vérifier que le modèle est bien dans les assets :**
   ```bash
   # Dans votre projet Flutter
   ls assets/models/model_resnet50.onnx
   ```

2. **Vérifier que pubspec.yaml inclut le modèle :**
   ```yaml
   flutter:
     assets:
       - assets/models/model_resnet50.onnx
   ```

3. **Lancer l'application :**
   ```bash
   flutter run
   ```

4. **Tester avec une image :**
   - Prendre une photo ou sélectionner une image depuis la galerie
   - Vérifier que la prédiction s'affiche correctement

### Étape 6: Optimisations (optionnel)

Pour réduire la taille de l'application et améliorer les performances :

1. **Quantifier le modèle ONNX** (réduction de ~75% de la taille) :
   ```python
   # Utiliser onnxruntime-tools pour quantifier
   # Voir la documentation ONNX pour plus de détails
   ```

2. **Utiliser un modèle plus petit** :
   - Entraîner avec EfficientNet-B0 au lieu de ResNet50
   - Convertir le nouveau modèle en ONNX

3. **Optimiser le preprocessing** :
   - Utiliser des opérations natives pour le resize et la normalisation
   - Mettre en cache les images préprocessées si nécessaire

### Dépannage

**Erreur "Model not found" :**
- Vérifiez que le fichier `model_resnet50.onnx` est bien dans `assets/models/`
- Vérifiez que `pubspec.yaml` inclut le chemin vers le modèle
- Exécutez `flutter clean` puis `flutter pub get`

**Erreur "ONNX Runtime not available" :**
- Vérifiez que `onnxruntime` est bien installé : `flutter pub get`
- Pour iOS, vous devrez peut-être exécuter `pod install` dans le dossier `ios/`

**Prédictions incorrectes :**
- Vérifiez que le preprocessing correspond exactement à celui utilisé lors de l'entraînement
- Vérifiez que l'ordre des classes correspond à `CLASS_NAMES_ORDERED`
- Testez avec des images similaires à celles du dataset d'entraînement

---

## Option 1: Utilisation de l'API REST

### Étape 1: Démarrer le serveur API

```bash
# Installer les dépendances
pip install -r requirements.txt

# Démarrer le serveur
python api_server.py --host 0.0.0.0 --port 5000
```

Le serveur sera accessible sur `http://localhost:5000` (ou l'IP de votre machine).

### Étape 2: Configurer Flutter

#### 2.1 Ajouter les dépendances

Dans votre `pubspec.yaml` :

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.1.0
  image_picker: ^1.0.4
```

Puis exécutez :
```bash
flutter pub get
```

#### 2.2 Configurer les permissions

**Android** (`android/app/src/main/AndroidManifest.xml`) :
```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.CAMERA"/>
```

**iOS** (`ios/Runner/Info.plist`) :
```xml
<key>NSPhotoLibraryUsageDescription</key>
<string>Need to access photos to classify plants</string>
<key>NSCameraUsageDescription</key>
<string>Need to access camera to take photos</string>
```

#### 2.3 Utiliser le service

Copiez le fichier `flutter_integration_example.dart` dans votre projet Flutter et utilisez-le comme suit :

```dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'plant_classification_service.dart'; // Votre service

// Dans votre widget
final result = await PlantClassificationService.predictFromXFile(imageFile);
print('Prediction: ${result['top_prediction']['class_fr']}');
print('Confidence: ${result['top_prediction']['confidence']}%');
```

#### 2.4 Configuration de l'URL

**Pour Android Emulator:**
```dart
static const String baseUrl = 'http://10.0.2.2:5000';
```

**Pour iOS Simulator:**
```dart
static const String baseUrl = 'http://localhost:5000';
```

**Pour appareil physique:**
```dart
// Utilisez l'IP locale de votre machine (ex: 192.168.1.100)
static const String baseUrl = 'http://192.168.1.100:5000';
```

**Pour production:**
```dart
static const String baseUrl = 'https://votre-serveur.com';
```

### Étape 3: Tester l'API

Vous pouvez tester l'API avec curl :

```bash
# Health check
curl http://localhost:5000/health

# Prediction (avec base64)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_here"}'
```

---


---

## Recommandation

Pour le **déploiement mobile**, **utilisez l'Option 2 (ONNX Runtime)**. C'est la solution optimale pour les applications mobiles car elle fonctionne hors ligne et offre une meilleure expérience utilisateur.

L'Option 1 (API REST) est recommandée uniquement pour :
- Le développement et les tests rapides
- Les applications web
- Les cas où vous avez besoin de mettre à jour le modèle fréquemment sans redéployer l'application

---

## Déploiement en Production

### Pour l'API REST:

1. **Déployer sur un serveur cloud** (AWS, Google Cloud, Heroku, etc.)
2. **Utiliser HTTPS** pour la sécurité
3. **Ajouter l'authentification** si nécessaire
4. **Optimiser le modèle** (quantization, pruning) pour réduire la latence

### Exemple avec Heroku:

```bash
# Créer un Procfile
echo "web: python api_server.py --host 0.0.0.0 --port \$PORT" > Procfile

# Déployer
git init
git add .
git commit -m "Initial commit"
heroku create votre-app
git push heroku main
```

---

## Support

Pour toute question ou problème, consultez:
- La documentation Flask: https://flask.palletsprojects.com/
- La documentation Flutter HTTP: https://pub.dev/packages/http
- La documentation ONNX Runtime: https://onnxruntime.ai/

