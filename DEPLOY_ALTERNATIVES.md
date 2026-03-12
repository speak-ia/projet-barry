# Alternatives à Render — déploiement pour la démo

Si Render renvoie **502 Bad Gateway** et **CORS** (l’instance free s’éteint, le proxy renvoie 502 sans en-têtes CORS), déployer l’API ailleurs garantit une démo stable.

---

## Option 1 : Google Cloud Run (recommandé pour demain)

**Avantages :** Même écosystème que Firebase (planthuile.web.app), timeout configurable (jusqu’à 60 min), pas de spin-down gratuit comme Render, CORS géré par ton Flask.

### Prérequis
- Compte Google Cloud (même compte que Firebase)
- Billing activé (Cloud Run a un quota gratuit mensuel ; une démo reste en gratuit)

### Déploiement rapide (environ 20–30 min)

1. **Installer Google Cloud SDK**  
   https://cloud.google.com/sdk/docs/install

2. **Se connecter et choisir le projet**
   ```bash
   gcloud auth login
   gcloud config set project VOTRE_PROJECT_ID
   ```
   (Le Project ID est celui de ton projet Firebase / planthuile.)

3. **Activer les APIs**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

4. **Déployer depuis le dossier du projet (BARRY)**
   ```bash
   cd /chemin/vers/BARRY
   gcloud run deploy oilseed-api \
     --source . \
     --region europe-west1 \
     --allow-unauthenticated \
     --set-env-vars "PYTHONUNBUFFERED=1" \
     --memory 2Gi \
     --timeout 300
   ```
   - `--source .` : build depuis le code (Dockerfile ou buildpacks).
   - `--allow-unauthenticated` : l’app Flutter peut appeler l’API sans token.
   - `--timeout 300` : 5 minutes par requête (évite les 502 en cas de cold start + chargement du modèle).

5. **Récupérer l’URL**  
   À la fin du déploiement, Cloud Run affiche l’URL du type :  
   `https://oilseed-api-xxxxx-ew.a.run.app`

6. **Dans Flutter**  
   Remplacer la base URL par cette URL :
   ```dart
   static const String baseUrl = 'https://oilseed-api-xxxxx-ew.a.run.app';
   ```

7. **CORS**  
   Ton `api_server.py` envoie déjà `Access-Control-Allow-Origin: *` ; Cloud Run renvoie les réponses de Flask, donc CORS fonctionne.

---

## Option 2 : Railway

**Avantages :** Déploiement depuis GitHub en quelques clics, pas de 502 “proxy” comme Render. Free tier limité mais souvent suffisant pour une démo.

### Étapes

1. Aller sur **https://railway.app** et se connecter avec GitHub.
2. **New Project** → **Deploy from GitHub repo** → choisir **speak-ia/projet-barry**.
3. **Utiliser le Dockerfile** (recommandé) : Railway détecte le `Dockerfile` à la racine. Il installe PyTorch **CPU uniquement** et `requirements-api.txt` pour rester sous le timeout de build (~5–10 min au lieu de 25+ min).
   - Si tu configures à la main : **Build Command** = laisser vide (build Docker) ou ne pas override.
   - **Start Command** : laisser vide pour utiliser le `CMD` du Dockerfile (sinon Railway peut lancer la commande sans shell et `$PORT` ne sera pas développé → erreur « '$PORT' is not a valid port number »). Si tu mets une commande custom, utilise : `sh -c 'gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 120 api_server:app'`.
4. Dans **Settings** → **Networking** → **Generate Domain** pour obtenir une URL publique.
5. Le modèle `models/oilseed_mobilenet.pth` est dans le repo, il sera inclus au build.
6. Dans Flutter, mettre `baseUrl` = l’URL Railway (ex. `https://oilseed-api-production-xxxx.up.railway.app`).

---

## Option 3 : Garder Render en “dernier recours”

- S’assurer que l’app Flutter appelle **GET /health** au chargement de l’écran (réveil).
- Attendre **1 à 2 minutes** après l’ouverture de l’app avant de lancer une prédiction.
- Timeout client **5 minutes** pour **POST /predict**.

Même avec ça, le free tier Render peut continuer à donner des 502 ; pour une démo “à tout prix”, privilégier **Cloud Run** ou **Railway**.

---

## Récap pour demain

| Solution       | Difficulté | Délai  | Stabilité démo |
|----------------|------------|--------|-----------------|
| **Cloud Run**  | Moyenne    | ~30 min| Très bonne      |
| **Railway**    | Facile     | ~15 min| Bonne           |
| **Render**     | Déjà fait  | 0      | Risquée (502)   |

Recommandation : **déployer sur Cloud Run** (ou Railway si tu préfères ne pas toucher à gcloud), changer uniquement la `baseUrl` dans Flutter, puis tester depuis planthuile.web.app. Une fois l’URL mise à jour et le déploiement terminé, les erreurs 502/CORS devraient être réglées pour la présentation.
