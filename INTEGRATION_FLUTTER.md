# Intégration Flutter — API Oilseed (Render)

Guide pour connecter votre application Flutter à l’API de détection de plantes oléagineuses déployée sur Render.

---

## 1. URL et endpoints

### URL de base (production)

```
https://projet-barry.onrender.com
```

| Méthode | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Infos API et liens des endpoints |
| GET | `/health` | Santé du service + `model_loaded` (true/false) |
| GET | `/classes` | Liste des 4 classes (soybean, sunflower, coconut, peanut) |
| POST | `/predict` | Prédiction à partir d’une image (multipart) |

### Réponses de `/predict`

**Plante oléagineuse reconnue (200) :**
```json
{
  "plant": "sunflower",
  "confidence": 0.92
}
```

**Pas une plante oléagineuse (200) :**
```json
{
  "plant": "not_oilseed",
  "message": "Ce n'est pas une plante oléagineuse."
}
```

**Modèle non chargé (503) :**
```json
{
  "error": "Modèle non chargé",
  "message": "Le modèle n'est pas encore disponible..."
}
```

**Paramètre optionnel :** `?threshold=0.6` (seuil de confiance, défaut 0.6).

### Timeout et cold start (Render free)

Sur **Render free**, l’instance s’éteint après inactivité. Au premier appel après un moment :
- le réveil peut prendre **~1 min** ;
- le chargement du modèle **~1–2 min**.

Donc le **timeout côté Flutter doit être d’au moins 4–5 minutes** (ex. `Duration(minutes: 5)`), sinon tu obtiendras « Délai dépassé » même quand l’API fonctionne.

**Astuce :** au démarrage de l’app, appeler une fois `GET /health` en arrière-plan (sans bloquer l’UI). Cela réveille l’instance ; quand l’utilisateur prend une photo, la prédiction sera plus rapide.

### Erreur « Failed to fetch » (Flutter Web)

Si tu vois **`ClientException: Failed to fetch`** en appelant `/predict` depuis l’app web (ex. planthuile.web.app), les causes possibles sont :

1. **CORS** — L’API doit renvoyer les en-têtes CORS pour ton origine. Le backend a été configuré pour accepter `*` (toutes origines). Après déploiement, réessaie.
2. **Instance Render éteinte** — La requête part avant que l’instance soit réveillée, ou l’instance coupe la connexion. Utilise le « réveil » (`GET /health` au démarrage) et un timeout client d’au moins 5 minutes.
3. **Réseau / blocage** — Vérifier que l’URL `https://projet-barry.onrender.com` s’ouvre dans un navigateur sur le même appareil.

---

## 2. Dépendances Flutter

Dans `pubspec.yaml` :

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.0
  image_picker: ^1.0.7
```

Puis :

```bash
flutter pub get
```

---

## 3. Permissions

**Android** — `android/app/src/main/AndroidManifest.xml` :

```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES"/>
```

**iOS** — `ios/Runner/Info.plist` :

```xml
<key>NSPhotoLibraryUsageDescription</key>
<string>Accès aux photos pour identifier les plantes oléagineuses</string>
<key>NSCameraUsageDescription</key>
<string>Accès à la caméra pour prendre des photos de plantes</string>
<key>NSAppTransportSecurity</key>
<dict>
  <key>NSAllowsArbitraryLoads</key>
  <false/>
  <key>NSAllowsLocalNetworking</key>
  <true/>
</dict>
```

(En production avec HTTPS Render, vous n’avez pas besoin d’autoriser le HTTP arbitraire.)

---

## 4. Service API

Créez `lib/services/oilseed_api_service.dart` :

**Flutter Web (Firebase Hosting) :** n’utilisez pas `predict(imagePath: ...)` (ça provoque « MultipartFile is only supported where dart:io is available »). Utilisez **`predictFromXFile(xFile)`** avec le `XFile` retourné par `ImagePicker`, qui envoie l’image en bytes et fonctionne sur web et mobile.

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class OilseedApiService {
  static const String baseUrl = 'https://projet-barry.onrender.com';

  /// Vérifier que l’API est joignable et que le modèle est chargé
  static Future<Map<String, dynamic>> checkHealth() async {
    final response = await http.get(Uri.parse('$baseUrl/health'));
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  /// Récupérer la liste des 4 classes
  static Future<List<String>> getClasses() async {
    final response = await http.get(Uri.parse('$baseUrl/classes'));
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    final list = data['classes'] as List<dynamic>?;
    return list?.map((e) => e.toString()).toList() ?? [];
  }

  /// Envoyer une image et obtenir la prédiction.
  /// Utilise [predictFromXFile] pour être compatible Web (Firebase) et mobile.
  /// [imagePath] : chemin du fichier (mobile/desktop uniquement — pas supporté sur le web).
  static Future<OilseedPrediction> predict({
    required String imagePath,
    double? threshold,
  }) async {
    final uri = threshold != null
        ? Uri.parse('$baseUrl/predict?threshold=$threshold')
        : Uri.parse('$baseUrl/predict');

    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('image', imagePath));

    final streamed = await request.send().timeout(
      const Duration(minutes: 5),
      onTimeout: () => throw Exception('Délai dépassé (5 min)'),
    );
    final response = await http.Response.fromStream(streamed);
    final data = jsonDecode(response.body) as Map<String, dynamic>;

    if (response.statusCode == 503) {
      return OilseedPrediction(
        isOilseed: false,
        plant: null,
        confidence: null,
        message: data['message'] as String? ?? 'Modèle non disponible',
        error: true,
      );
    }

    if (response.statusCode != 200) {
      return OilseedPrediction(
        isOilseed: false,
        plant: null,
        confidence: null,
        message: data['error'] as String? ?? 'Erreur ${response.statusCode}',
        error: true,
      );
    }

    final plant = data['plant'] as String?;
    final isOilseed = plant != null && plant != 'not_oilseed';

    return OilseedPrediction(
      isOilseed: isOilseed,
      plant: isOilseed ? plant : null,
      confidence: data['confidence'] != null
          ? (data['confidence'] as num).toDouble()
          : null,
      message: data['message'] as String?,
      error: false,
    );
  }

  /// Prédiction à partir d’un XFile (ImagePicker). Compatible Web et mobile.
  /// À utiliser sur Flutter Web (Firebase) pour éviter « MultipartFile is only supported where dart:io is available ».
  static Future<OilseedPrediction> predictFromXFile(
    dynamic xFile, {
    double? threshold,
  }) async {
    final bytes = await xFile.readAsBytes();
    final name = xFile.name ?? 'image.jpg';
    final uri = threshold != null
        ? Uri.parse('$baseUrl/predict?threshold=$threshold')
        : Uri.parse('$baseUrl/predict');

    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes(
      'image',
      bytes,
      filename: name,
    ));

    final streamed = await request.send().timeout(
      const Duration(minutes: 5),
      onTimeout: () => throw Exception('Délai dépassé (5 min)'),
    );
    final response = await http.Response.fromStream(streamed);
    final data = jsonDecode(response.body) as Map<String, dynamic>;

    if (response.statusCode == 503) {
      return OilseedPrediction(
        isOilseed: false,
        plant: null,
        confidence: null,
        message: data['message'] as String? ?? 'Modèle non disponible',
        error: true,
      );
    }
    if (response.statusCode != 200) {
      return OilseedPrediction(
        isOilseed: false,
        plant: null,
        confidence: null,
        message: data['error'] as String? ?? 'Erreur ${response.statusCode}',
        error: true,
      );
    }

    final plant = data['plant'] as String?;
    final isOilseed = plant != null && plant != 'not_oilseed';
    return OilseedPrediction(
      isOilseed: isOilseed,
      plant: isOilseed ? plant : null,
      confidence: data['confidence'] != null
          ? (data['confidence'] as num).toDouble()
          : null,
      message: data['message'] as String?,
      error: false,
    );
  }
}

/// Résultat de prédiction
class OilseedPrediction {
  final bool isOilseed;
  final String? plant;
  final double? confidence;
  final String? message;
  final bool error;

  OilseedPrediction({
    required this.isOilseed,
    this.plant,
    this.confidence,
    this.message,
    required this.error,
  });

  /// Nom français pour l’affichage
  static const Map<String, String> plantNamesFr = {
    'soybean': 'Soja',
    'sunflower': 'Tournesol',
    'coconut': 'Cocotier',
    'peanut': 'Arachide',
  };

  String get displayLabel {
    if (error) return message ?? 'Erreur';
    if (isOilseed && plant != null) {
      return '${plantNamesFr[plant] ?? plant} (${((confidence ?? 0) * 100).toStringAsFixed(0)} %)';
    }
    return message ?? "Ce n'est pas une plante oléagineuse.";
  }
}
```

---

## 5. Données plantes (fiches à afficher après prédiction)

Après une prédiction réussie, l’app peut afficher une fiche détaillée (nom, scientifique, description, extraction d’huile, usages, bienfaits, saison, régions). Les données sont dans un JSON embarqué dans l’app.

### 5.1 Fichier JSON

Le fichier `plant_info.json` est fourni dans le dépôt : **`assets/plant_info.json`** (à la racine du projet BARRY). Copiez-le dans votre projet Flutter :

- **Emplacement dans Flutter :** `assets/plant_info.json`

Dans `pubspec.yaml` :

```yaml
flutter:
  assets:
    - assets/plant_info.json
```

Les clés du JSON sont en **français** (soja, tournesol, cocotier, arachide). L’API retourne les noms en **anglais** (soybean, sunflower, coconut, peanut). Il faut donc faire la correspondance :

| API (`plant`) | Clé JSON |
|---------------|----------|
| soybean       | soja     |
| sunflower     | tournesol|
| coconut       | cocotier |
| peanut        | arachide |

### 5.2 Modèle et chargement

Créez `lib/models/plant_info.dart` :

```dart
import 'package:flutter/services.dart';
import 'dart:convert';

/// Correspondance nom API (anglais) → clé JSON (français)
const Map<String, String> apiToJsonKey = {
  'soybean': 'soja',
  'sunflower': 'tournesol',
  'coconut': 'cocotier',
  'peanut': 'arachide',
};

class PlantInfo {
  final String display;
  final String scientific;
  final String description;
  final String oilExtraction;
  final String uses;
  final String benefits;
  final String harvestSeason;
  final String regions;

  PlantInfo({
    required this.display,
    required this.scientific,
    required this.description,
    required this.oilExtraction,
    required this.uses,
    required this.benefits,
    required this.harvestSeason,
    required this.regions,
  });

  factory PlantInfo.fromJson(Map<String, dynamic> json) {
    return PlantInfo(
      display: json['display'] as String? ?? '',
      scientific: json['scientific'] as String? ?? '',
      description: json['description'] as String? ?? '',
      oilExtraction: json['oilExtraction'] as String? ?? '',
      uses: json['uses'] as String? ?? '',
      benefits: json['benefits'] as String? ?? '',
      harvestSeason: json['harvestSeason'] as String? ?? '',
      regions: json['regions'] as String? ?? '',
    );
  }
}

class PlantInfoRepository {
  static Map<String, PlantInfo>? _cache;

  static Future<Map<String, PlantInfo>> load() async {
    if (_cache != null) return _cache!;
    final String jsonString = await rootBundle.loadString('assets/plant_info.json');
    final Map<String, dynamic> map = jsonDecode(jsonString) as Map<String, dynamic>;
    _cache = map.map((key, value) => MapEntry(key, PlantInfo.fromJson(value as Map<String, dynamic>)));
    return _cache!;
  }

  /// Retourne la fiche pour une prédiction API (plant en anglais).
  static PlantInfo? getForApiPlant(Map<String, PlantInfo> data, String apiPlantName) {
    final key = apiToJsonKey[apiPlantName];
    return key != null ? data[key] : null;
  }
}
```

Au démarrage de l’app (ou de l’écran), chargez une fois les données :

```dart
late Map<String, PlantInfo> _plantData;

@override
void initState() {
  super.initState();
  PlantInfoRepository.load().then((data) => setState(() => _plantData = data));
}
```

Après une prédiction avec `plant` (ex. `sunflower`), récupérez la fiche :

```dart
final info = PlantInfoRepository.getForApiPlant(_plantData, _result!.plant!);
```

### 5.3 Afficher la fiche dans l’UI

Exemple de carte à afficher sous le résultat de prédiction (titre + confiance) :

```dart
if (_result!.isOilseed && _result!.plant != null && _plantData.isNotEmpty) {
  final info = PlantInfoRepository.getForApiPlant(_plantData, _result!.plant!);
  if (info != null) ...[
    const SizedBox(height: 16),
    Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(info.display, style: Theme.of(context).textTheme.titleLarge),
            Text(info.scientific, style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 12),
            Text('Description', style: Theme.of(context).textTheme.titleSmall),
            Text(info.description),
            const SizedBox(height: 8),
            Text('Extraction d\'huile', style: Theme.of(context).textTheme.titleSmall),
            Text(info.oilExtraction),
            const SizedBox(height: 8),
            Text('Usages', style: Theme.of(context).textTheme.titleSmall),
            Text(info.uses),
            const SizedBox(height: 8),
            Text('Bienfaits', style: Theme.of(context).textTheme.titleSmall),
            Text(info.benefits),
            const SizedBox(height: 8),
            Text('Récolte', style: Theme.of(context).textTheme.titleSmall),
            Text(info.harvestSeason),
            const SizedBox(height: 8),
            Text('Régions', style: Theme.of(context).textTheme.titleSmall),
            Text(info.regions),
          ],
        ),
      ),
    ),
  ],
}
```

Vous pouvez remplacer la `Card` par un écran dédié (ex. `PlantDetailScreen(info)`) avec scroll si les textes sont longs.

---

## 6. Exemple d’écran (caméra / galerie)

Exemple minimal dans un écran Flutter :

```dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/oilseed_api_service.dart';

class PlantClassifierScreen extends StatefulWidget {
  const PlantClassifierScreen({super.key});

  @override
  State<PlantClassifierScreen> createState() => _PlantClassifierScreenState();
}

class _PlantClassifierScreenState extends State<PlantClassifierScreen> {
  final ImagePicker _picker = ImagePicker();
  bool _loading = false;
  OilseedPrediction? _result;
  String? _error;

  @override
  void initState() {
    super.initState();
    // Réveiller l’API Render (free tier) en arrière-plan pour réduire le délai au premier scan
    OilseedApiService.checkHealth().ignore();
  }

  Future<void> _pickAndPredict(bool fromCamera) async {
    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });

    try {
      final source = fromCamera ? ImageSource.camera : ImageSource.gallery;
      final XFile? xFile = await _picker.pickImage(source: source);
      if (xFile == null) {
        setState(() => _loading = false);
        return;
      }

      // predictFromXFile : compatible Web (Firebase) et mobile (évite l’erreur MultipartFile / dart:io)
      final prediction = await OilseedApiService.predictFromXFile(xFile);

      setState(() {
        _result = prediction;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Plantes oléagineuses')),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _loading ? null : () => _pickAndPredict(true),
              icon: const Icon(Icons.camera_alt),
              label: const Text('Prendre une photo'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: _loading ? null : () => _pickAndPredict(false),
              icon: const Icon(Icons.photo_library),
              label: const Text('Choisir depuis la galerie'),
            ),
            const SizedBox(height: 32),
            if (_loading)
              const Center(child: CircularProgressIndicator())
            else if (_error != null)
              Card(
                color: Colors.red.shade100,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text(_error!, style: const TextStyle(color: Colors.red)),
                ),
              )
            else if (_result != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _result!.displayLabel,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      if (_result!.isOilseed && _result!.confidence != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 8.0),
                          child: Text(
                            'Confiance : ${(_result!.confidence! * 100).toStringAsFixed(1)} %',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                        ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
```

Le chemin `xFile.path` retourné par `ImagePicker` est utilisé directement pour l’appel API.

---

## 7. Résumé des étapes

1. Ajouter `http` et `image_picker` dans `pubspec.yaml`, puis `flutter pub get`.
2. Configurer les permissions Android et iOS (INTERNET, CAMERA, galerie).
3. Créer `lib/services/oilseed_api_service.dart` avec l’URL de base et la méthode `predict`.
4. Copier `assets/plant_info.json` (depuis le projet BARRY) dans votre app Flutter, l’ajouter aux assets dans `pubspec.yaml`, et créer `lib/models/plant_info.dart` (modèle `PlantInfo` + `PlantInfoRepository`) pour afficher les fiches plantes après prédiction.
5. Dans votre écran : appeler `ImagePicker`, récupérer le chemin du fichier, appeler `OilseedApiService.predict(imagePath: path)` ; afficher le résultat puis, si `isOilseed`, récupérer `PlantInfoRepository.getForApiPlant(_plantData, result.plant!)` et afficher la fiche (display, scientific, description, oilExtraction, uses, benefits, harvestSeason, regions).

---

## 8. Environnements (dev / prod)

Pour changer d’URL selon l’environnement :

```dart
class OilseedApiService {
  static String get baseUrl {
    const env = String.fromEnvironment('API_ENV', defaultValue: 'prod');
    if (env == 'dev') return 'http://192.168.1.100:5000'; // votre machine en local
    return 'https://projet-barry.onrender.com';
  }
  // ...
}
```

Lancer en dev : `flutter run --dart-define=API_ENV=dev`

---

## 9. Voir aussi

- **Déploiement Render** : `DEPLOY_RENDER.md`
- **Intégration ONNX (hors ligne)** : `FLUTTER_INTEGRATION.md` (Option 2)
