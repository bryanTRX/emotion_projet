# 🧠 Emotion Detection Project

Détection et reconnaissance **des émotions faciales en temps réel** à partir du flux d’une caméra, basée sur un **modèle CNN** entraîné sur des images de visages.  

Le projet combine **MediaPipe**, **OpenCV** et **PyTorch** pour détecter les visages, prétraiter les images, puis prédire les émotions en direct avec un modèle de deep learning optimisé.

Le système :

1. Détecte les visages avec **MediaPipe**
2. Recadre et prétraite la région d’intérêt
3. Passe l’image dans ton modèle CNN
4. Affiche la **prédiction et la probabilité** en temps réel

## ⚙️ Installation

### Cloner le dépôt

```bash
git clone https://github.com/bryanTRX/emotion_projet
cd emotion_projet
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Prédiction en temps réel

Lancer la détection des émotions via la webcam :

```bash
python src/app_model.py
```

L’application :

- détecte automatiquement les visages,

- prétraite l’image,

- prédit l’émotion via ton modèle CNN,

- affiche le label et la confiance en direct.

Appuie sur la lettre  **q** pour quitter.
