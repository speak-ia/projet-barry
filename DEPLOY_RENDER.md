# Déployer l’API Oilseed sur Render

Guide pour mettre l’API Flask en ligne sur [Render](https://render.com) afin de tester les endpoints depuis le web et une app mobile (Flutter) en attendant la fin de l’entraînement.

---

## Que mettre sur GitHub ?

**Vous ne devez pas mettre le dossier complet avec les datasets.** Pour le déploiement Render, il suffit de pousser **uniquement le code** (scripts Python + config).

### À inclure dans le repo (à pousser sur GitHub)

- `api_server.py` — serveur Flask
- `model.py`, `dataset.py`, `config.py`, `inference.py` — utilisés par l’API
- `requirements.txt` — dépendances
- `render.yaml` — config Render (optionnel)
- `DEPLOY_RENDER.md`, `INSTALL.md`, `README.md` — doc (optionnel)

### À ne pas pousser (déjà dans .gitignore)

- **Datasets** (`dataset/`, dossiers `Soja/`, `Tournesol/`, etc.) — inutiles pour l’API, très lourds
- **Modèle** (`models/*.pth`) — vous l’ajouterez plus tard (Secret File Render ou commit après entraînement)
- **Environnement virtuel** (`.venv/`, `venv/`)
- **Résultats** (`results/`)

Un fichier **`.gitignore`** à la racine du projet exclut déjà ces dossiers/fichiers. Vous pouvez donc faire un **commit de tout le projet** : seuls les scripts et la config seront envoyés, pas les images ni le modèle.

Résumé : **mettez uniquement les scripts sur GitHub**, pas les datasets.

---

## Comportement sans modèle

L’API **démarre même si le fichier du modèle n’est pas présent** (`models/oilseed_mobilenet.pth`). Dans ce cas :

- **GET /health** → `"model_loaded": false`
- **POST /predict** → **503** avec un message en français : *« Le modèle n'est pas encore disponible… »*

Dès que vous déployez le fichier du modèle (voir plus bas), les prédictions fonctionnent.

---

## Option A : Déploiement via le dashboard Render

1. **Compte**  
   Créez un compte sur [render.com](https://render.com) (gratuit).

2. **Nouveau Web Service**  
   - Dashboard → **New** → **Web Service**
   - Connectez votre repo GitHub/GitLab (projet BARRY).

3. **Configuration**
   - **Name** : `oilseed-api` (ou autre)
   - **Region** : Frankfurt (ou le plus proche)
   - **Runtime** : **Python 3**
   - **Build Command** :
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command** :
     ```bash
     gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 120 api_server:app
     ```
   - **Instance Type** : Free (suffisant pour tester).

4. **Variables d’environnement**  
   Aucune obligatoire. Le port est fourni par Render via `PORT`.

5. **Déployer**  
   Cliquez sur **Create Web Service**. Render build puis démarre l’API. L’URL sera du type :  
   `https://oilseed-api-xxxx.onrender.com`

6. **Tester**
   - Navigateur : `https://votre-service.onrender.com/health`
   - Prédiction (sans modèle) :  
     `POST https://votre-service.onrender.com/predict`  
     → vous devez recevoir **503** et le message « Modèle non chargé » en JSON.

---

## Option B : Déploiement avec Blueprint (render.yaml)

Si votre dépôt contient déjà `render.yaml` à la racine :

1. Render → **New** → **Blueprint**
2. Sélectionnez le repo BARRY
3. Render lit `render.yaml` et crée le service avec les commandes définies dedans.

Vous pouvez ajuster `render.yaml` (nom du service, région, etc.) puis redéployer.

---

## Endpoints à utiliser (web / mobile)

| Méthode | URL              | Description |
|--------|-------------------|-------------|
| GET    | `/health`         | Santé + `model_loaded` (true/false) |
| GET    | `/classes`        | Liste des 4 classes (soybean, sunflower, coconut, peanut) |
| POST   | `/predict`        | Prédiction à partir d’une image |

**Exemple POST /predict (multipart, pour Flutter / Postman)**  
- **Content-Type** : `multipart/form-data`  
- **Body** : champ nommé `image` = fichier image (jpg, png, etc.)  
- **Query** (optionnel) : `?threshold=0.6` (seuil de confiance)

**Réponses possibles**  
- Plante reconnue :  
  `{"plant": "sunflower", "confidence": 0.92}`  
- Pas une plante oléagineuse :  
  `{"plant": "not_oilseed", "message": "Ce n'est pas une plante oléagineuse."}`  
- Modèle absent : **503** +  
  `{"error": "Modèle non chargé", "message": "…"}`  

---

## Activer les prédictions après l’entraînement

Quand `python train.py` a produit `models/oilseed_mobilenet.pth` :

1. **Option 1 – Commit du modèle (simple pour tester)**  
   - Ajoutez `models/oilseed_mobilenet.pth` au dépôt (attention à la taille si > 10 Mo).  
   - Poussez sur GitHub/GitLab.  
   - Redéployez le service sur Render (manuel ou auto si déploiement automatique activé).

2. **Option 2 – Secret File (recommandé si le fichier est gros)**  
   - Dans Render : **Environment** → **Secret Files**  
   - Créez un fichier dont le chemin dans le container est :  
     `models/oilseed_mobilenet.pth`  
   - Contenu = le fichier `.pth` généré en local.  
   - Redémarrez le service.

Après redéploiement, **GET /health** doit afficher `"model_loaded": true` et **POST /predict** doit renvoyer des prédictions au lieu de 503.

---

## Exemple cURL

```bash
# Santé
curl https://votre-service.onrender.com/health

# Prédiction (fichier image)
curl -X POST https://votre-service.onrender.com/predict \
  -F "image=@/chemin/vers/photo.jpg"
```

---

## Exemple Flutter (appel API)

```dart
final uri = Uri.parse('https://votre-service.onrender.com/predict');
final request = http.MultipartRequest('POST', uri);
request.files.add(await http.MultipartFile.fromPath('image', pathToImageFile));
final response = await request.send();
// Parser le JSON de la réponse
```

Une fois l’API déployée, utilisez l’URL de votre service Render dans votre app Flutter à la place de `localhost`.
