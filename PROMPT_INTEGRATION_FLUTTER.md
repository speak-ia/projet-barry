# Prompt pour intégrer le modèle ONNX dans Flutter

Copiez ce prompt et donnez-le à Cursor pour intégrer le modèle de classification de plantes oléagineuses dans votre application Flutter.

---

## 📋 PROMPT À COPIER-COLLER DANS CURSOR

```
Intègre un modèle ONNX de classification de plantes oléagineuses dans mon app Flutter.

Le modèle est dans assets/models/model_resnet50.onnx.

Fais ceci :

1. Ajoute dans pubspec.yaml :
   - onnxruntime: ^1.15.0
   - image_picker: ^1.0.4
   - image: ^4.0.0
   - Vérifie que assets/models/model_resnet50.onnx est dans la section flutter.assets

2. Crée lib/services/onnx_plant_classifier.dart avec :
   - Classe ONNXPlantClassifier qui charge le modèle depuis assets/models/model_resnet50.onnx
   - loadModel() : charge le modèle ONNX avec OrtSession.fromBuffer()
   - predictFromFile(String path) : prédit depuis un fichier image
   - Preprocessing : resize 224x224, normalisation ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   - Convertir en tensor [1,3,224,224] avec Float32List
   - Exécuter inference avec session.run()
   - Appliquer softmax sur les logits
   - Mapping classes français :
     sunflower->Tournesol, peanut->Arachide, coconut->Cocotier, soybean->Soja, 
     sesame->Sésame, cotton->Coton, oil_palm->Palmier
   - Ordre: ['sunflower','peanut','coconut','soybean','sesame','cotton','oil_palm']
   - Retourner PredictionResult avec liste triée par confiance et topPrediction
   - dispose() pour libérer OrtSession

3. Crée un écran (ex: lib/screens/plant_classifier_screen.dart) :
   - Charge le modèle dans initState()
   - Boutons pour caméra/galerie avec ImagePicker
   - Affiche CircularProgressIndicator pendant chargement
   - Affiche la prédiction principale (nom français + pourcentage)
   - Liste de toutes les prédictions triées
   - Gestion erreurs avec try/catch

4. Permissions :
   - Android/AndroidManifest.xml : INTERNET, CAMERA, READ_EXTERNAL_STORAGE
   - iOS/Info.plist : NSPhotoLibraryUsageDescription, NSCameraUsageDescription

Le modèle attend [1,3,224,224] en input et retourne [1,7] logits. Utilise onnxruntime package.
```

---

## 📝 Version détaillée (si besoin)

Si vous avez besoin de plus de détails, voici une version plus complète :

```
Je veux intégrer un modèle ONNX de classification de plantes oléagineuses dans mon application Flutter. 

Le modèle est déjà dans assets/models/model_resnet50.onnx.

J'ai besoin de :

1. **Ajouter les dépendances nécessaires dans pubspec.yaml** :
   - onnxruntime: ^1.15.0
   - image_picker: ^1.0.4
   - image: ^4.0.0

2. **Créer un service de classification** dans lib/services/onnx_plant_classifier.dart avec :
   - Une classe ONNXPlantClassifier qui charge le modèle depuis les assets
   - Méthode loadModel() pour charger le modèle ONNX
   - Méthode predictFromFile(String imagePath) pour faire des prédictions
   - Preprocessing des images : resize à 224x224, normalisation ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - Postprocessing : softmax pour convertir les logits en probabilités
   - Mapping des classes avec noms français :
     - 'sunflower' -> 'Tournesol'
     - 'peanut' -> 'Arachide'
     - 'coconut' -> 'Cocotier'
     - 'soybean' -> 'Soja'
     - 'sesame' -> 'Sésame'
     - 'cotton' -> 'Coton'
     - 'oil_palm' -> 'Palmier'
   - Ordre des classes : ['sunflower', 'peanut', 'coconut', 'soybean', 'sesame', 'cotton', 'oil_palm']
   - Retourner un objet avec les prédictions triées par confiance décroissante
   - Gestion des erreurs et dispose() pour libérer les ressources

3. **Créer un écran d'exemple** qui :
   - Charge le modèle au démarrage
   - Permet de prendre une photo ou sélectionner une image depuis la galerie
   - Affiche la prédiction principale avec le nom français et le pourcentage de confiance
   - Affiche toutes les prédictions triées par confiance
   - Gère les états de chargement et les erreurs

4. **Configurer les permissions** :
   - Android : ajouter INTERNET, CAMERA, READ_EXTERNAL_STORAGE dans AndroidManifest.xml
   - iOS : ajouter NSPhotoLibraryUsageDescription et NSCameraUsageDescription dans Info.plist

5. **Vérifier que pubspec.yaml inclut le modèle dans les assets** :
   ```yaml
   flutter:
     assets:
       - assets/models/model_resnet50.onnx
   ```

Le modèle attend un input de shape [1, 3, 224, 224] (batch_size=1, channels=3, height=224, width=224) et retourne des logits de shape [1, 7] (7 classes).

Utilise le package onnxruntime pour charger et exécuter le modèle ONNX.
```

---

## 🔧 Informations techniques supplémentaires

### Format du modèle
- **Input** : Tensor de shape `[1, 3, 224, 224]` (RGB, normalisé avec ImageNet stats)
- **Output** : Tensor de shape `[1, 7]` (logits pour 7 classes)
- **Preprocessing requis** :
  - Resize à 224x224
  - Normalisation : `(pixel / 255.0 - mean) / std`
  - Mean = [0.485, 0.456, 0.406]
  - Std = [0.229, 0.224, 0.225]
  - Format : [batch, channels, height, width] = [1, 3, 224, 224]

### Classes (ordre d'index)
0. sunflower → Tournesol
1. peanut → Arachide
2. coconut → Cocotier
3. soybean → Soja
4. sesame → Sésame
5. cotton → Coton
6. oil_palm → Palmier

### Structure de retour attendue
```dart
class Prediction {
  final String classEn;
  final String classFr;
  final double confidence; // 0-100
}

class PredictionResult {
  final List<Prediction> predictions; // Triées par confiance décroissante
  final Prediction topPrediction;
}
```

---

## 📚 Exemple de code de référence

Vous pouvez consulter le fichier `flutter_onnx_example.dart` dans ce projet pour voir une implémentation complète de référence.
