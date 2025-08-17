# api_flask.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from io import BytesIO
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost", "http://localhost:3000"]}})

# ---------------------------------------------------------------------
# CONFIG GLOBALE
# ---------------------------------------------------------------------
# Dossier "saved_models" à la même racine que ce fichier
ROOT = Path(__file__).parent.resolve()
SAVED_MODELS_DIR = (ROOT / "saved_models").resolve()

# Cache (thread-safe) pour éviter de recharger les modèles à chaque requête
_model_cache = {}
_model_cache_lock = threading.Lock()

# ---------------------------------------------------------------------
# OUTILS
# ---------------------------------------------------------------------
def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_list_dir(p: Path):
    return [d for d in p.iterdir() if d.is_dir()]

def _list_available_models():
    """
    Un modèle est considéré "valide" s'il contient un config.json.
    """
    models = []
    if not SAVED_MODELS_DIR.exists():
        return models
    for d in _safe_list_dir(SAVED_MODELS_DIR):
        cfg = d / "config.json"
        if cfg.exists():
            models.append(d.name)
    return sorted(models)

def _resolve_model_dir(model_name: str) -> Path:
    d = SAVED_MODELS_DIR / model_name
    if not d.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found in {SAVED_MODELS_DIR}")
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json for model '{model_name}'")
    return d

def _load_model_and_config(model_name: str):
    """
    Charge (ou récupère du cache) (model, config).
    config.json (exemple) :
    {
      "model_type": "keras_h5",          // "keras_h5" | "tf_savedmodel"
      "model_path": "model.h5",          // relatif au dossier du modèle
      "input_format": "image",           // "image" | "numeric"
      "img_size": [224, 224],            // requis si image
      "scale_0_1": true,                 // normaliser 0..1
      "channels": 3,                     // 1 ou 3 (grayscale vs RGB)
      "class_names": ["Early Blight", "Late Blight", "Healthy"], // si classification
      "feature_order": ["f1","f2", ...]  // requis si numeric
    }
    """
    with _model_cache_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]["model"], _model_cache[model_name]["config"]

        model_dir = _resolve_model_dir(model_name)
        cfg = _load_json(model_dir / "config.json")

        model_type = cfg.get("model_type", "keras_h5")
        model_path = model_dir / cfg.get("model_path", "model.h5")

        if model_type == "keras_h5":
            model = tf.keras.models.load_model(model_path)
        elif model_type == "tf_savedmodel":
            # Dossier SavedModel (contenant saved_model.pb + variables)
            model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        _model_cache[model_name] = {"model": model, "config": cfg}
        return model, cfg

def _prepare_image(data: bytes, img_size=(224, 224), scale_0_1=True, channels=3) -> np.ndarray:
    """
    Prépare une image pour prédiction: resize, channels, normalisation.
    """
    img = Image.open(BytesIO(data))
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img = img.resize(tuple(img_size))
    arr = np.array(img, dtype=np.float32)
    if scale_0_1:
        arr = arr / 255.0
    # Ajout batch dimension
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr

def _prepare_numeric(features_json: dict, feature_order: list) -> np.ndarray:
    """
    Prépare un vecteur/matrice de features à partir d'un JSON.
    - features_json doit contenir un dict {feature_name: value}
    - feature_order donne l'ordre attendu par le modèle
    Retourne un batch de shape (1, n_features).
    """
    try:
        row = [float(features_json[name]) for name in feature_order]
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"Missing feature '{missing}' in payload.")
    return np.array([row], dtype=np.float32)

def _postprocess_predictions(preds: np.ndarray, class_names=None):
    """
    Si classification multi-classes (softmax), renvoie (class, confidence).
    Sinon renvoie raw preds.
    """
    if class_names is not None and preds.ndim == 2 and preds.shape[0] == 1:
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        return {"class": class_names[idx], "confidence": conf}
    # Régression ou format non standard : retourne les valeurs
    return {"predictions": preds.tolist()}

# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.get("/ping")
def ping():
    return "Hello, I am alive"

@app.get("/models")
def list_models():
    return jsonify({"available_models": _list_available_models()})

@app.get("/models/<model_name>")
def model_info(model_name: str):
    try:
        _, cfg = _load_model_and_config(model_name)
        # Ne retourne pas tout le fichier brut (sécurité); on expose l’essentiel
        info = {
            "model_name": model_name,
            "model_type": cfg.get("model_type"),
            "input_format": cfg.get("input_format"),
            "img_size": cfg.get("img_size"),
            "scale_0_1": cfg.get("scale_0_1"),
            "channels": cfg.get("channels"),
            "class_names": cfg.get("class_names"),
            "feature_order": cfg.get("feature_order"),
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/predict")
def predict():
    
    
    """
    Usage:
    - Sélection du modèle:
        /predict?model=<nom_modele>
      (sinon, prend le premier modèle disponible)

    - input_format == "image":
        form-data: file=<image>

    - input_format == "numeric":
        body JSON: {"features": {"f1": 1.2, "f2": 3.4, ...}}
    """
    try:
        model_name = request.args.get("model")
        if not model_name:
            available = _list_available_models()
            if not available:
                return jsonify({"error": "No models available."}), 500
            model_name = available[0]

        model, cfg = _load_model_and_config(model_name)

        input_format = cfg.get("input_format", "image")
        class_names = cfg.get("class_names")
        if input_format == "image":
            if "file" not in request.files:
                return jsonify({"error": "No file field named 'file'"}), 400
            f = request.files["file"]
            if f.filename == "":
                return jsonify({"error": "Empty filename"}), 400

            img_size = cfg.get("img_size", [224, 224])
            scale_0_1 = bool(cfg.get("scale_0_1", True))
            channels = int(cfg.get("channels", 3))
            x = _prepare_image(f.read(), img_size=img_size, scale_0_1=scale_0_1, channels=channels)

        elif input_format == "numeric":
            data = request.get_json(silent=True) or {}
            feats = data.get("features")
            if not isinstance(feats, dict):
                return jsonify({"error": "JSON body must contain {'features': {...}}"}), 400
            feature_order = cfg.get("feature_order")
            if not feature_order:
                return jsonify({"error": "Model config missing 'feature_order' for numeric input."}), 500
            x = _prepare_numeric(feats, feature_order)

        else:
            return jsonify({"error": f"Unsupported input_format: {input_format}"}), 400

        preds = model.predict(x)
        out = _postprocess_predictions(preds, class_names=class_names)
        out.update({"model": model_name})
        return jsonify(out)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 0.0.0.0 pour accepter les requêtes du réseau local
    app.run(host="0.0.0.0", port=8000, debug=True)
