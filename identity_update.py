import os
import cv2
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from keras_facenet import FaceNet
from mtcnn import MTCNN
import numpy as np

# Initialiser FaceNet et le détecteur de visage (MTCNN)
facenet = FaceNet()
detector = MTCNN()

# Répertoire contenant les images de visages connus
KNOWN_FACES_DIR = './known_faces'
MODEL_OUTPUT_FILE = 'svm_face_recognition_with_pca.pkl'


def extract_embeddings(directory):
    """
    Extraire les embeddings des visages connus à partir du répertoire.
    """
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


def train_and_save_model(embeddings, labels, model_file):
    """
    Entraîner un SVM avec PCA et sauvegarder le modèle.
    """
    # Réduction de la dimensionnalité avec PCA
    pca = PCA(n_components=0.95)  # 95% de variance expliquée
    reduced_embeddings = pca.fit_transform(embeddings)

    # Encoder les labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Entraîner un SVM
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(reduced_embeddings, encoded_labels)

    # Sauvegarder le modèle, le PCA, et l'encodeur
    with open(model_file, 'wb') as f:
        pickle.dump((svm_model, label_encoder, pca), f)

    print(f"Model saved to {model_file}")


def update_model_if_needed(directory, model_file):
    """
    Vérifier et mettre à jour le modèle SVM si de nouveaux visages sont ajoutés.
    """
    embeddings, labels = extract_embeddings(directory)

    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            svm_model, label_encoder, pca = pickle.load(f)

        # Vérifier si le modèle doit être mis à jour
        if len(labels) != len(label_encoder.classes_):
            print("New faces detected. Updating the model...")
            train_and_save_model(embeddings, labels, model_file)
        else:
            print("No new faces detected. Model is up-to-date.")
    else:
        print("Model not found. Training a new model...")
        train_and_save_model(embeddings, labels, model_file)

update_model_if_needed(KNOWN_FACES_DIR, MODEL_OUTPUT_FILE)