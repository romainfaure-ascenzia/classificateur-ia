"""
Classificateur d'images avec TensorFlow Lite et Streamlit
Pour déploiement Kubernetes multi-env avec ArgoCD
"""

import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
import os

# Configuration de la page
st.set_page_config(
    page_title="Classificateur IA",
    page_icon="🔍",
    layout="centered"
)

# URLs des fichiers modèle
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"
MODEL_PATH = "mobilenet_v1_1.0_224_quant.tflite"
LABELS_PATH = "labels_mobilenet_quant_v1_224.txt"

@st.cache_resource
def download_model():
    """Télécharge le modèle et les labels si nécessaire"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        st.info("📥 Téléchargement du modèle MobileNet...")
        import zipfile
        zip_path = "model.zip"
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(zip_path)
        st.success("✅ Modèle téléchargé !")

@st.cache_resource
def load_labels():
    """Charge les labels ImageNet"""
    download_model()
    with open(LABELS_PATH, 'r') as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_model():
    """Charge le modèle TensorFlow Lite"""
    download_model()
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, target_size=(224, 224)):
    """Prétraite l'image pour le modèle"""
    img = image.resize(target_size, Image.Resampling.LANCZOS)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    return np.expand_dims(img_array, axis=0)

def classify_image(interpreter, image, labels, top_k=5):
    """Classifie une image et retourne les top-k prédictions"""
    input_data = preprocess_image(image)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    predictions = output_data[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = predictions[idx]
        if predictions.max() > 1:
            score = (score / 255.0) * 100
        else:
            score = score * 100
        results.append({
            'label': labels[idx] if idx < len(labels) else f"Classe {idx}",
            'score': score
        })
    
    return results

# === Interface Streamlit ===
st.title("🔍 Classificateur d'Images IA")
st.markdown("*Utilise TensorFlow Lite avec MobileNet*")

# Afficher l'environnement
env = os.getenv("APP_ENV", "development")
st.sidebar.markdown(f"**Environnement:** `{env}`")
st.sidebar.markdown("---")

# Charger le modèle et les labels
with st.spinner("Chargement du modèle..."):
    try:
        interpreter = load_model()
        labels = load_labels()
        st.sidebar.success("✅ Modèle prêt")
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {e}")
        st.stop()

# Upload d'image
st.markdown("### 📤 Chargez une image")
uploaded_file = st.file_uploader(
    "Choisissez une image...",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Formats supportés: JPG, PNG, WebP"
)

# Option caméra (pour mobile)
use_camera = st.checkbox("📷 Utiliser la caméra")
if use_camera:
    camera_image = st.camera_input("Prenez une photo")
    if camera_image:
        uploaded_file = camera_image

# Traitement de l'image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Image uploadée", use_container_width=True)
    
    with st.spinner("🔄 Analyse en cours..."):
        results = classify_image(interpreter, image, labels)
    
    with col2:
        st.markdown("### 🎯 Résultats")
        for i, res in enumerate(results):
            label = res['label'].replace('_', ' ').title()
            score = res['score']
            
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📌"
            
            st.markdown(f"{emoji} **{label}**")
            st.progress(min(score / 100, 1.0))
            st.caption(f"Confiance: {score:.1f}%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Propulsé par TensorFlow Lite • Déployé sur Kubernetes avec ArgoCD"
    "</div>",
    unsafe_allow_html=True
)