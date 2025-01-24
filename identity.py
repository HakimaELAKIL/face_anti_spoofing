# import numpy as np
# import cv2
# import pickle
# from sklearn.svm import SVC
# from sklearn.preprocessing import LabelEncoder
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# import os

# # Initialiser FaceNet et le détecteur de visage (MTCNN)
# facenet = FaceNet()
# detector = MTCNN()

# # Répertoire contenant les images de visages connus
# KNOWN_FACES_DIR = './known_faces'
# MODEL_OUTPUT_FILE = 'svm_face_recognition.pkl'

# # Extraire les embeddings des visages connus
# def extract_embeddings(directory):
#     embeddings = []
#     labels = []

#     for label in os.listdir(directory):
#         label_dir = os.path.join(directory, label)

#         if not os.path.isdir(label_dir):
#             continue

#         for image_name in os.listdir(label_dir):
#             image_path = os.path.join(label_dir, image_name)

#             # Lire et traiter l'image
#             img = cv2.imread(image_path)
#             if img is None:
#                 continue

#             rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             faces = detector.detect_faces(rgb_img)
#             if len(faces) == 0:
#                 print(f"No face detected in {image_name}")
#                 continue

#             # Extraire le visage
#             x, y, w, h = faces[0]['box']
#             x, y = max(0, x), max(0, y)  # S'assurer que les coordonnées sont positives
#             face_crop = rgb_img[y:y + h, x:x + w]

#             # Redimensionner pour FaceNet
#             face_resized = cv2.resize(face_crop, (160, 160))

#             # Calculer l'embedding
#             embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
#             embeddings.append(embedding)
#             labels.append(label)

#     return embeddings, labels

# # Charger les embeddings et labels
# embeddings, labels = extract_embeddings(KNOWN_FACES_DIR)

# # Encoder les labels (convertir les noms en indices numériques)
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(labels)

# # Entraîner un SVM
# svm_model = SVC(kernel='rbf', probability=True)
# svm_model.fit(embeddings, encoded_labels)

# # Sauvegarder le modèle SVM et l'encodeur
# with open(MODEL_OUTPUT_FILE, 'wb') as f:
#     pickle.dump((svm_model, label_encoder), f)

# print(f"SVM model saved to {MODEL_OUTPUT_FILE}")
# 

import numpy as np
import cv2
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from keras_facenet import FaceNet
from mtcnn import MTCNN
import os

# Initialiser FaceNet et le détecteur de visage (MTCNN)
facenet = FaceNet()
detector = MTCNN()

# Répertoire contenant les images de visages connus
KNOWN_FACES_DIR = './known_faces'
MODEL_OUTPUT_FILE = 'svm_face_recognition_with_pca.pkl'

# Extraire les embeddings des visages connus
def extract_embeddings(directory):
    embeddings = []
    labels = []

    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)

        if not os.path.isdir(label_dir):
            continue

        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)

            # Lire et traiter l'image
            img = cv2.imread(image_path)
            if img is None:
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_img)
            if len(faces) == 0:
                print(f"No face detected in {image_name}")
                continue

            # Extraire le visage
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)  # S'assurer que les coordonnées sont positives
            face_crop = rgb_img[y:y + h, x:x + w]

            # Redimensionner pour FaceNet
            face_resized = cv2.resize(face_crop, (160, 160))

            # Calculer l'embedding
            embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
            embeddings.append(embedding)
            labels.append(label)

    return embeddings, labels

# Charger les embeddings et labels
embeddings, labels = extract_embeddings(KNOWN_FACES_DIR)

# Appliquer PCA pour réduire la dimensionnalité des embeddings
pca = PCA(n_components=0.95)  # 95% de variance expliquée
reduced_embeddings = pca.fit_transform(embeddings)

# Encoder les labels (convertir les noms en indices numériques)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Entraîner un SVM avec les embeddings réduits
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(reduced_embeddings, encoded_labels)

# Sauvegarder le modèle SVM, l'encodeur et le modèle PCA
with open(MODEL_OUTPUT_FILE, 'wb') as f:
    pickle.dump((svm_model, label_encoder, pca), f)

print(f"SVM model with PCA saved to {MODEL_OUTPUT_FILE}")

# Fonction pour tester la reconnaissance sur un ensemble d'images
def test_face_recognition(directory, model_file):
    with open(model_file, 'rb') as f:
        svm_model, label_encoder, pca = pickle.load(f)

    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)

        if not os.path.isdir(label_dir):
            continue

        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)

            # Lire et traiter l'image
            img = cv2.imread(image_path)
            if img is None:
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_img)
            if len(faces) == 0:
                print(f"No face detected in {image_name}")
                continue

            # Extraire le visage
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)  # S'assurer que les coordonnées sont positives
            face_crop = rgb_img[y:y + h, x:x + w]

            # Redimensionner pour FaceNet
            face_resized = cv2.resize(face_crop, (160, 160))

            # Calculer l'embedding
            embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

            # Réduire l'embedding avec PCA
            reduced_embedding = pca.transform([embedding])

            # Prédire avec SVM
            predicted_label_index = svm_model.predict(reduced_embedding)[0]
            predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

            print(f"Image: {image_name} | Predicted: {predicted_label} | Actual: {label}")

# Test sur le même ensemble de données
test_face_recognition(KNOWN_FACES_DIR, MODEL_OUTPUT_FILE)