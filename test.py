# from flask import Flask, render_template
# from flask_socketio import SocketIO
# import os
# from datetime import datetime

# # Flask app and SocketIO configuration
# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Directory to save images
# UPLOAD_FOLDER = "images"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return "WebSocket Server Running"

# @socketio.on('connect')
# def handle_connect():
#     print("Client connected")

# @socketio.on('disconnect')
# def handle_disconnect():
#     print("Client disconnected")

# @socketio.on('image')
# def handle_image(data):
#     print("Image received!")
#     # Save the image to a file
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filepath = os.path.join(UPLOAD_FOLDER, f"image_{timestamp}.jpg")
#     with open(filepath, 'wb') as f:
#         f.write(data)
#     print(f"Image saved as {filepath}")

# if __name__ == '__name__':
#     socketio.run(app, host='0.0.0.0', port=5000)

    # app.run(host='192.168.137.1', port=5000) 

# import os
# import pickle
# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import Label, Button
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import LabelEncoder
# from keras_facenet import FaceNet
# from mtcnn import MTCNN

# # Chemins des modèles
# MODEL_PATH = "./svm_face_recognition_with_pca.pkl"

# # Charger les modèles nécessaires
# def load_models():
#     with open(MODEL_PATH, 'rb') as f:
#         svm_model, label_encoder, pca = pickle.load(f)
#     return svm_model, label_encoder, pca

# # Détection et reconnaissance faciale
# def recognize_face(frame, detector, facenet, svm_model, pca, label_encoder):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(rgb_frame)

#     if len(faces) == 0:
#         return "No face detected"

#     x, y, w, h = faces[0]['box']
#     x, y = max(0, x), max(0, y)
#     face_crop = rgb_frame[y:y + h, x:x + w]
#     face_resized = cv2.resize(face_crop, (160, 160))

#     embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
#     reduced_embedding = pca.transform([embedding])

#     predicted_label_index = svm_model.predict(reduced_embedding)[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#     probability = max(svm_model.predict_proba(reduced_embedding)[0])

#     return f"Label: {predicted_label}, Probability: {probability:.2f}"

# # Interface Tkinter pour afficher les résultats en temps réel
# class FaceRecognitionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Face Recognition in Real-Time")

#         # Charger les modèles
#         self.svm_model, self.label_encoder, self.pca = load_models()
#         self.facenet = FaceNet()
#         self.detector = MTCNN()

#         # Vidéo
#         self.cap = cv2.VideoCapture(0)

#         # Interface utilisateur
#         self.label = Label(root, text="Initializing...", font=("Arial", 14))
#         self.label.pack()

#         self.start_button = Button(root, text="Start Recognition", command=self.start_recognition)
#         self.start_button.pack()

#         self.stop_button = Button(root, text="Stop", command=self.stop_recognition)
#         self.stop_button.pack()

#         self.running = False

#     def start_recognition(self):
#         self.running = True
#         self.update_frame()

#     def stop_recognition(self):
#         self.running = False
#         self.cap.release()
#         self.root.destroy()

#     def update_frame(self):
#         if not self.running:
#             return

#         ret, frame = self.cap.read()
#         if not ret:
#             self.label.config(text="Failed to capture video.")
#             return

#         result_text = recognize_face(frame, self.detector, self.facenet, self.svm_model, self.pca, self.label_encoder)
#         self.label.config(text=result_text)

#         # Afficher la vidéo en direct dans une fenêtre séparée
#         cv2.imshow("Face Recognition", frame)

#         # Mettre à jour le prochain frame
#         self.root.after(10, self.update_frame)

# if __name__ == '__main__':
#     root = tk.Tk()
#     app = FaceRecognitionApp(root)
#     root.mainloop()

# import os
# import pickle
# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import Label, Button
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import LabelEncoder
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from scipy.spatial import distance

# # Chemins des modèles
# MODEL_PATH = "./svm_face_recognition_with_pca.pkl"

# # Charger les modèles nécessaires
# def load_models():
#     with open(MODEL_PATH, 'rb') as f:
#         svm_model, label_encoder, pca = pickle.load(f)
#     return svm_model, label_encoder, pca

# # Fonction pour calculer l'aspect ratio des yeux
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Détection et reconnaissance faciale
# def recognize_face(frame, detector, facenet, svm_model, pca, label_encoder):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(rgb_frame)

#     if len(faces) == 0:
#         return "No face detected"

#     x, y, w, h = faces[0]['box']
#     x, y = max(0, x), max(0, y)
#     face_crop = rgb_frame[y:y + h, x:x + w]
#     face_resized = cv2.resize(face_crop, (160, 160))

#     embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
#     reduced_embedding = pca.transform([embedding])

#     predicted_label_index = svm_model.predict(reduced_embedding)[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#     probability = max(svm_model.predict_proba(reduced_embedding)[0])

#     # Vérification anti-usurpation (simple exemple basé sur la variance des gradients)
#     gray_face = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
#     laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
#     if laplacian_var < 100:  # Seuil ajustable
#         return "Spoofing detected: Low gradient variance"

#     return f"Label: {predicted_label}, Probability: {probability:.2f}"

# # Interface Tkinter pour afficher les résultats en temps réel
# class FaceRecognitionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Face Recognition in Real-Time")

#         # Charger les modèles
#         self.svm_model, self.label_encoder, self.pca = load_models()
#         self.facenet = FaceNet()
#         self.detector = MTCNN()

#         # Vidéo
#         self.cap = cv2.VideoCapture(0)

#         # Interface utilisateur
#         self.label = Label(root, text="Initializing...", font=("Arial", 14))
#         self.label.pack()

#         self.start_button = Button(root, text="Start Recognition", command=self.start_recognition)
#         self.start_button.pack()

#         self.stop_button = Button(root, text="Stop", command=self.stop_recognition)
#         self.stop_button.pack()

#         self.running = False

#     def start_recognition(self):
#         self.running = True
#         self.update_frame()

#     def stop_recognition(self):
#         self.running = False
#         self.cap.release()
#         self.root.destroy()

#     def update_frame(self):
#         if not self.running:
#             return

#         ret, frame = self.cap.read()
#         if not ret:
#             self.label.config(text="Failed to capture video.")
#             return

#         result_text = recognize_face(frame, self.detector, self.facenet, self.svm_model, self.pca, self.label_encoder)
#         self.label.config(text=result_text)

#         # Afficher la vidéo en direct dans une fenêtre séparée
#         cv2.imshow("Face Recognition", frame)

#         # Mettre à jour le prochain frame
#         self.root.after(10, self.update_frame)

# if __name__ == '__main__':
#     root = tk.Tk()
#     app = FaceRecognitionApp(root)
#     root.mainloop()

# import cv2
# import pickle
# import numpy as np
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from collections import defaultdict
# import onnxruntime as ort  # For anti-spoofing model

# # Initialize models
# facenet = FaceNet()
# detector = MTCNN()

# # Eye status tracker for each face label
# eye_status_tracker = defaultdict(list)

# # Model paths
# SVM_MODEL_PATH = './svm_face_recognition.pkl'
# ANTI_SPOOFING_MODEL_PATH = './AntiSpoofing_bin_1.5_128.onnx'

# # Load the SVM model and encoder
# def load_svm_model():
#     with open(SVM_MODEL_PATH, 'rb') as f:
#         return pickle.load(f)

# # Load the anti-spoofing model (ONNX)
# def load_anti_spoofing_model():
#     return ort.InferenceSession(ANTI_SPOOFING_MODEL_PATH)

# # Preprocessing for anti-spoofing model
# def preprocess_for_anti_spoofing(frame):
#     frame_resized = cv2.resize(frame, (128, 128))
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#     frame_normalized = frame_rgb.astype(np.float32) / 255.0
#     frame_input = np.transpose(frame_normalized, (2, 0, 1))
#     frame_input = np.expand_dims(frame_input, axis=0)
#     return frame_input

# # Eye blink detection using Haar Cascade
# def detect_haar_eyes(frame):
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
#     eye_status = []

#     for (ex, ey, ew, eh) in eyes:
#         eye_region = gray_frame[ey:ey + eh, ex:ex + ew]
#         _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
#         white_pixel_ratio = cv2.countNonZero(threshold_eye) / (ew * eh)

#         if white_pixel_ratio < 0.2:
#             eye_status.append("Closed")
#         else:
#             eye_status.append("Open")

#         cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#     return "Blink detected" if eye_status.count("Closed") > 0 else "No Blink"

# # Face recognition with SVM
# def recognize_face_with_svm(face_embedding, svm_model, label_encoder):
#     predicted_label_index = svm_model.predict([face_embedding])[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#     probability = max(svm_model.predict_proba([face_embedding])[0])
#     return predicted_label, probability

# # Verify face, blink, and anti-spoofing
# def verify_face_and_blink(frame, svm_model, label_encoder, anti_spoofing_model):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(rgb_frame)

#     results = []
#     for face in faces:
#         x, y, w, h = face['box']
#         x, y = max(0, x), max(0, y)

#         # Extract and prepare the face
#         face_crop = rgb_frame[y:y + h, x:x + w]
#         face_resized = cv2.resize(face_crop, (160, 160))
#         embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

#         # Face recognition with SVM
#         label, probability = recognize_face_with_svm(embedding, svm_model, label_encoder)

#         # Eye blink detection
#         blink_status = detect_haar_eyes(frame)

#         # Anti-spoofing verification
#         frame_input = preprocess_for_anti_spoofing(face_crop)
#         inputs = {anti_spoofing_model.get_inputs()[0].name: frame_input}
#         spoofing_output = anti_spoofing_model.run(None, inputs)
#         is_real_face = spoofing_output[0][0][0] > 5
#         spoofing_status = "Real Face" if is_real_face else "Spoofed Face"

#         # Eye status tracking
#         if label in eye_status_tracker:
#             previous_status = eye_status_tracker[label]
#             if previous_status == blink_status:
#                 eye_tracker_status = "Spoof Attack Detected"
#             else:
#                 eye_tracker_status = f"Eye status varies: {blink_status}"
#         else:
#             eye_tracker_status = f"First detection: {blink_status}"

#         eye_status_tracker[label] = blink_status

#         # Annotate the image
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         label_text = f"{label} ({probability:.2f}) - {spoofing_status} - {eye_tracker_status}"
#         cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         results.append({"label": label, "probability": probability, "spoofing_status": spoofing_status, "eye_status": eye_tracker_status})

#     return frame, results

# # Real-time webcam video processing
# def start_video_stream():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     # Load models
#     svm_model, label_encoder = load_svm_model()
#     anti_spoofing_model = load_anti_spoofing_model()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Process frame for face recognition, blink detection, and anti-spoofing
#         frame_with_annotations, results = verify_face_and_blink(frame, svm_model, label_encoder, anti_spoofing_model)

#         # Display the resulting frame
#         cv2.imshow('Video Stream', frame_with_annotations)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Start the video stream
# if __name__ == '__main__':
#     start_video_stream()

# import cv2
# import numpy as np
# import onnxruntime as ort  # For anti-spoofing model
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from collections import defaultdict
# import pickle

# # Initialize models
# facenet = FaceNet()
# detector = MTCNN()

# # Eye status tracker for each face label
# eye_status_tracker = defaultdict(list)

# # Model paths
# SVM_MODEL_PATH = './svm_face_recognition.pkl'
# ANTI_SPOOFING_MODEL_PATH = './AntiSpoofing_bin_1.5_128.onnx'

# # Load the SVM model and encoder
# def load_svm_model():
#     with open(SVM_MODEL_PATH, 'rb') as f:
#         return pickle.load(f)

# # Load the anti-spoofing model (ONNX)
# def load_anti_spoofing_model():
#     return ort.InferenceSession(ANTI_SPOOFING_MODEL_PATH)

# # Preprocessing for anti-spoofing model
# def preprocess_for_anti_spoofing(frame):
#     frame_resized = cv2.resize(frame, (128, 128))
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#     frame_normalized = frame_rgb.astype(np.float32) / 255.0
#     frame_input = np.transpose(frame_normalized, (2, 0, 1))
#     frame_input = np.expand_dims(frame_input, axis=0)
#     return frame_input

# # Eye blink detection using Haar Cascade
# def detect_haar_eyes(frame):
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
#     eye_status = []

#     for (ex, ey, ew, eh) in eyes:
#         eye_region = gray_frame[ey:ey + eh, ex:ex + ew]
#         _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
#         white_pixel_ratio = cv2.countNonZero(threshold_eye) / (ew * eh)

#         if white_pixel_ratio < 0.2:
#             eye_status.append("Closed")
#         else:
#             eye_status.append("Open")

#         cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#     return "Blink detected" if eye_status.count("Closed") > 0 else "No Blink"

# # Face recognition with SVM
# def recognize_face_with_svm(face_embedding, svm_model, label_encoder):
#     predicted_label_index = svm_model.predict([face_embedding])[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#     probability = max(svm_model.predict_proba([face_embedding])[0])
#     return predicted_label, probability

# # Verify face, blink, and anti-spoofing
# def verify_face_and_blink(frame, svm_model, label_encoder, anti_spoofing_model):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(rgb_frame)

#     results = []
#     for face in faces:
#         x, y, w, h = face['box']
#         x, y = max(0, x), max(0, y)

#         # Extract and prepare the face
#         face_crop = rgb_frame[y:y + h, x:x + w]
#         face_resized = cv2.resize(face_crop, (160, 160))
#         embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

#         # Face recognition with SVM
#         label, probability = recognize_face_with_svm(embedding, svm_model, label_encoder)

#         # Eye blink detection
#         blink_status = detect_haar_eyes(frame)

#         # Anti-spoofing verification
#         frame_input = preprocess_for_anti_spoofing(face_crop)
#         inputs = {anti_spoofing_model.get_inputs()[0].name: frame_input}
#         spoofing_output = anti_spoofing_model.run(None, inputs)
#         is_real_face = spoofing_output[0][0][0] > 0.5
#         spoofing_status = "Real Face" if is_real_face else "Spoofed Face"

#         # Eye status tracking
#         if label in eye_status_tracker:
#             previous_status = eye_status_tracker[label]
#             if previous_status == blink_status:
#                 eye_tracker_status = "Spoof Attack Detected"
#             else:
#                 eye_tracker_status = f"Eye status varies: {blink_status}"
#         else:
#             eye_tracker_status = f"First detection: {blink_status}"

#         eye_status_tracker[label] = blink_status

#         # Annotate the image
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         label_text = f"{label} ({probability:.2f}) - {spoofing_status} - {eye_tracker_status}"
#         cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         results.append({"label": label, "probability": probability, "spoofing_status": spoofing_status, "eye_status": eye_tracker_status})

#     return frame, results

# # Real-time webcam video processing
# def start_video_stream():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     # Load models
#     svm_model, label_encoder = load_svm_model()
#     anti_spoofing_model = load_anti_spoofing_model()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Process frame for face recognition, blink detection, and anti-spoofing
#         frame_with_annotations, results = verify_face_and_blink(frame, svm_model, label_encoder, anti_spoofing_model)

#         # Display the resulting frame
#         cv2.imshow('Video Stream', frame_with_annotations)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Start the video stream
# if __name__ == '__main__':
#     start_video_stream()


# LAST VERSION 


# import cv2
# import numpy as np
# import onnxruntime as ort  # Pour le modèle anti-spoofing
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from collections import defaultdict
# import pickle

# # Initialiser les modèles
# facenet = FaceNet()
# detector = MTCNN()

# # Suivi du statut des yeux pour chaque étiquette de visage
# eye_status_tracker = defaultdict(list)

# # Chemins des modèles
# SVM_MODEL_PATH = './svm_face_recognition.pkl'
# ANTI_SPOOFING_MODEL_PATH = './AntiSpoofing_bin_1.5_128.onnx'

# # Charger le modèle SVM et l'encodeur
# def load_svm_model():
#     with open(SVM_MODEL_PATH, 'rb') as f:
#         return pickle.load(f)

# # Charger le modèle anti-spoofing (ONNX)
# def load_anti_spoofing_model():
#     return ort.InferenceSession(ANTI_SPOOFING_MODEL_PATH)

# # Prétraitement pour le modèle anti-spoofing
# def preprocess_for_anti_spoofing(frame):
#     frame_resized = cv2.resize(frame, (128, 128))
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#     frame_normalized = frame_rgb.astype(np.float32) / 255.0
#     frame_input = np.transpose(frame_normalized, (2, 0, 1))
#     frame_input = np.expand_dims(frame_input, axis=0)
#     return frame_input

# # Détection de clignement des yeux avec Haar Cascade
# def detect_haar_eyes(frame):
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
#     eye_status = []

#     for (ex, ey, ew, eh) in eyes:
#         eye_region = gray_frame[ey:ey + eh, ex:ex + ew]
#         _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
#         white_pixel_ratio = cv2.countNonZero(threshold_eye) / (ew * eh)

#         if white_pixel_ratio < 0.2:
#             eye_status.append("Closed")
#         else:
#             eye_status.append("Open")

#         cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#     return "Blink detected" if eye_status.count("Closed") > 0 else "No Blink"

# # Reconnaissance faciale avec SVM
# def recognize_face_with_svm(face_embedding, svm_model, label_encoder):
#     predicted_label_index = svm_model.predict([face_embedding])[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#     probability = max(svm_model.predict_proba([face_embedding])[0])
#     return predicted_label, probability

# # Vérification de l'anti-spoofing
# def verify_anti_spoofing(frame, model_anti_spoofing):
#     input_data = preprocess_for_anti_spoofing(frame)
#     inputs = {model_anti_spoofing.get_inputs()[0].name: input_data}
#     result = model_anti_spoofing.run(None, inputs)
#     spoofing_score = result[0][0][0]
#     return spoofing_score  # Spoofing_score près de 0 signifie spoofing, plus élevé signifie authentique

# # Vérification combinée de l'anti-spoofing et du clignement des yeux
# def verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(rgb_frame)

#     results = []
#     for face in faces:
#         x, y, w, h = face['box']
#         x, y = max(0, x), max(0, y)

#         # Extraire et préparer le visage
#         face_crop = rgb_frame[y:y + h, x:x + w]
#         face_resized = cv2.resize(face_crop, (160, 160))
#         embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

#         # Reconnaissance faciale avec SVM
#         label, probability = recognize_face_with_svm(embedding, svm_model, label_encoder)

#         # Détection du clignement des yeux
#         blink_status = detect_haar_eyes(frame)

#         # Vérification anti-spoofing
#         spoofing_score = verify_anti_spoofing(frame, anti_spoofing_model)
#         is_authentic = spoofing_score > 0.5  # Si le score d'authenticité est > 0.5, on considère que ce n'est pas un spoofing

#         # Condition combinée : Si l'anti-spoofing est authentique et qu'un clignement des yeux est détecté
#         if is_authentic and blink_status == "Blink detected":
#             combined_status = f"{label} ({probability:.2f}) - Blink Detected and Authentic"
#         elif is_authentic and blink_status == "No Blink":
#             combined_status = f"{label} ({probability:.2f}) - No Blink, but Authentic"
#         elif not is_authentic:
#             combined_status = f"Spoofing detected!"

#         # Annoter l'image
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, combined_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         results.append({"label": label, "probability": probability, "eye_status": blink_status, "anti_spoofing": is_authentic})

#     return frame, results

# # Stream vidéo en temps réel
# def start_video_stream():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     # Charger les modèles
#     svm_model, label_encoder = load_svm_model()
#     anti_spoofing_model = load_anti_spoofing_model()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Traiter le cadre pour la reconnaissance faciale, le clignement des yeux et l'anti-spoofing
#         frame_with_annotations, results = verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model)

#         # Afficher le cadre résultant
#         cv2.imshow('Video Stream', frame_with_annotations)

#         # Quitter la boucle si 'q' est pressé
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Démarrer le stream vidéo
# if __name__ == '__main__':
#     start_video_stream()


# import cv2
# import numpy as np
# import onnxruntime as ort  # Pour le modèle anti-spoofing
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from collections import defaultdict
# import pickle
# import sqlite3
# from datetime import datetime

# # Initialiser les modèles
# facenet = FaceNet()
# detector = MTCNN()

# # Suivi du statut des yeux pour chaque étiquette de visage
# eye_status_tracker = defaultdict(list)

# # Chemins des modèles
# SVM_MODEL_PATH = './svm_face_recognition.pkl'
# ANTI_SPOOFING_MODEL_PATH = './AntiSpoofing_bin_1.5_128.onnx'

# # Charger le modèle SVM et l'encodeur
# def load_svm_model():
#     with open(SVM_MODEL_PATH, 'rb') as f:
#         return pickle.load(f)

# # Charger le modèle anti-spoofing (ONNX)
# def load_anti_spoofing_model():
#     return ort.InferenceSession(ANTI_SPOOFING_MODEL_PATH)

# # Prétraitement pour le modèle anti-spoofing
# def preprocess_for_anti_spoofing(frame):
#     frame_resized = cv2.resize(frame, (128, 128))
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#     frame_normalized = frame_rgb.astype(np.float32) / 255.0
#     frame_input = np.transpose(frame_normalized, (2, 0, 1))
#     frame_input = np.expand_dims(frame_input, axis=0)
#     return frame_input

# # Détection de clignement des yeux avec Haar Cascade
# def detect_haar_eyes(frame):
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
#     eye_status = []

#     for (ex, ey, ew, eh) in eyes:
#         eye_region = gray_frame[ey:ey + eh, ex:ex + ew]
#         _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
#         white_pixel_ratio = cv2.countNonZero(threshold_eye) / (ew * eh)

#         if white_pixel_ratio < 0.4:
#             eye_status.append("Closed")
#         else:
#             eye_status.append("Open")

#         cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#     return "Blink detected" if eye_status.count("Closed") > 0 else "No Blink"

# # Reconnaissance faciale avec SVM
# def recognize_face_with_svm(face_embedding, svm_model, label_encoder):
#     predicted_label_index = svm_model.predict([face_embedding])[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#     probability = max(svm_model.predict_proba([face_embedding])[0])
#     return predicted_label, probability

# # Vérification de l'anti-spoofing
# def verify_anti_spoofing(frame, model_anti_spoofing):
#     input_data = preprocess_for_anti_spoofing(frame)
#     inputs = {model_anti_spoofing.get_inputs()[0].name: input_data}
#     result = model_anti_spoofing.run(None, inputs)
#     spoofing_score = result[0][0][0]
#     return spoofing_score  # Spoofing_score près de 0 signifie spoofing, plus élevé signifie authentique

# # Vérification combinée de l'anti-spoofing et du clignement des yeux
# def verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(rgb_frame)

#     results = []
#     for face in faces:
#         x, y, w, h = face['box']
#         x, y = max(0, x), max(0, y)

#         # Extraire et préparer le visage
#         face_crop = rgb_frame[y:y + h, x:x + w]
#         face_resized = cv2.resize(face_crop, (160, 160))
#         embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

#         # Reconnaissance faciale avec SVM
#         label, probability = recognize_face_with_svm(embedding, svm_model, label_encoder)

#         # Détection du clignement des yeux
#         blink_status = detect_haar_eyes(frame)

#         # Vérification anti-spoofing
#         spoofing_score = verify_anti_spoofing(frame, anti_spoofing_model)
#         is_authentic = spoofing_score > 0.5  # Si le score d'authenticité est > 0.5, on considère que ce n'est pas un spoofing

#         # Condition combinée : Si l'anti-spoofing est authentique et qu'un clignement des yeux est détecté
#         if is_authentic and blink_status == "Blink detected":
#             combined_status = f"{label} ({probability:.2f}) - Blink Detected and Authentic"
#             # Insérer dans la base de données
#             insert_into_db(label)
#         elif is_authentic and blink_status == "No Blink":
#             combined_status = f"{label} ({probability:.2f}) - No Blink, but Authentic"
#         elif not is_authentic:
#             combined_status = f"Spoofing detected!"

#         # Annoter l'image
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, combined_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         results.append({"label": label, "probability": probability, "eye_status": blink_status, "anti_spoofing": is_authentic})

#     return frame, results

# # Insérer les données dans la base de données SQLite
# def insert_into_db(name):
#     conn = sqlite3.connect('face_recognition.db')
#     cursor = conn.cursor()

#     # Créer la table si elle n'existe pas
#     cursor.execute('''CREATE TABLE IF NOT EXISTS user_data (name TEXT, timestamp TEXT)''')

#     # Insérer les informations dans la table
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     cursor.execute("INSERT INTO user_data (name, timestamp) VALUES (?, ?)", (name, timestamp))

#     # Commit et fermer la connexion
#     conn.commit()
#     conn.close()

# # Stream vidéo en temps réel
# def start_video_stream():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     # Charger les modèles
#     svm_model, label_encoder = load_svm_model()
#     anti_spoofing_model = load_anti_spoofing_model()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Traiter le cadre pour la reconnaissance faciale, le clignement des yeux et l'anti-spoofing
#         frame_with_annotations, results = verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model)

#         # Afficher le cadre résultant
#         cv2.imshow('Video Stream', frame_with_annotations)

#         # Quitter la boucle si 'q' est pressé
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Démarrer le stream vidéo
# if __name__ == '__main__':
#     start_video_stream()

# import os
# import cv2
# import numpy as np
# import pickle
# import sqlite3
# from datetime import datetime, timedelta
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# import mediapipe as mp
# import onnxruntime as ort
# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA
# from collections import defaultdict
# import joblib

# # Initialisation
# facenet = FaceNet()
# detector = MTCNN()
# eye_tracker = defaultdict(list)

# # Charger le modèle anti-spoofing ONNX
# anti_spoofing_model = ort.InferenceSession('./AntiSpoofing_bin_1.5_128.onnx')

# # Charger le classificateur Haar pour la détection des yeux
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# # Initialisation de Mediapipe pour le mesh du visage
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)

# # Charger le modèle SVM
# svm_model, label_encoder, pca = joblib.load('./svm_face_recognition_with_pca.pkl')  # Remplacez par le chemin de votre modèle

# # Initialisation de la base de données SQLite
# def log_to_database(label):
#     conn = sqlite3.connect('face_recognition.db')
#     cursor = conn.cursor()
#     cursor.execute('''CREATE TABLE IF NOT EXISTS logs (name TEXT, timestamp TEXT)''')
    
#     # Vérifier la dernière entrée enregistrée pour le même label
#     cursor.execute('SELECT timestamp FROM logs WHERE name = ? ORDER BY timestamp DESC LIMIT 1', (label,))
#     last_entry = cursor.fetchone()
    
#     if last_entry:
#         last_time = datetime.strptime(last_entry[0], '%Y-%m-%d %H:%M:%S.%f')
#         if datetime.now() - last_time < timedelta(minutes=2):
#             conn.close()
#             return False  # Sauter l'enregistrement si moins de 2 minutes se sont écoulées

#     # Enregistrer l'entrée actuelle
#     cursor.execute('INSERT INTO logs (name, timestamp) VALUES (?, ?)', (label, datetime.now()))
#     conn.commit()
#     conn.close()
#     return True

# # Détection de clignement des yeux avec Haar Cascade
# def detect_haar_blink(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
#     return "Blink Detected" if len(eyes) == 2 else "No Blink"

# # Vérification anti-spoofing
# def verify_anti_spoofing(frame):
#     resized = cv2.resize(frame, (128, 128))
#     normalized = resized.astype(np.float32) / 255.0
#     input_data = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)
#     spoofing_score = anti_spoofing_model.run(None, {anti_spoofing_model.get_inputs()[0].name: input_data})[0][0][0]
#     return spoofing_score > 0.5

# # Reconnaissance faciale avec SVM
# def recognize_face(embedding):
#     # Appliquer la transformation PCA pour réduire les dimensions
#     reduced_embedding = pca.transform([embedding])
    
#     # Prédire avec le modèle SVM
#     prediction = svm_model.predict(reduced_embedding)
#     confidence = max(svm_model.predict_proba(reduced_embedding)[0])
    
#     # Décoder le label en texte (si nécessaire)
#     label = label_encoder.inverse_transform([prediction[0]])[0]
    
#     return label, confidence

# # Boucle principale de reconnaissance
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
    
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Détection du clignement des yeux
#             haar_blink_status = detect_haar_blink(frame)
            
#             # Vérification anti-spoofing
#             is_authentic = verify_anti_spoofing(frame)
            
#             # Reconnaissance faciale
#             faces = detector.detect_faces(rgb_frame)
#             if len(faces) > 0:
#                 # Extraire le premier visage détecté (vous pouvez améliorer cela pour plusieurs visages si nécessaire)
#                 x, y, w, h = faces[0]['box']
#                 x, y = max(0, x), max(0, y)
#                 face_crop = rgb_frame[y:y + h, x:x + w]

#                 # Redimensionner pour FaceNet
#                 face_resized = cv2.resize(face_crop, (160, 160))

#                 # Calculer l'embedding
#                 embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

#                 # Reconnaître le visage
#                 label, confidence = recognize_face(embedding)
                
#                 # Combiner les résultats
#                 if is_authentic and haar_blink_status == "Blink Detected":
#                     if log_to_database(label):
#                         status = f"Label: {label} ({confidence:.2f}) - Blink & Authentic"
#                     else:
#                         status = "Duplicate Log Skipped"
#                 elif not is_authentic:
#                     status = "Spoofing Detected"
#                 else:
#                     status = "No Blink or Low Confidence"
                
#                 cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     cv2.imshow("Face Recognition and Blink Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Appuyez sur 'Q' pour quitter
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import onnxruntime as ort  # Pour le modèle anti-spoofing
from keras_facenet import FaceNet
from mtcnn import MTCNN
from collections import defaultdict
import pickle
import sqlite3
from datetime import datetime
import requests
import http.client
import json


# Initialiser les modèles
facenet = FaceNet()
detector = MTCNN()

# Suivi du statut des yeux pour chaque étiquette de visage
eye_status_tracker = defaultdict(list)

# Chemins des modèles
SVM_MODEL_PATH = './svm_face_recognition.pkl'
ANTI_SPOOFING_MODEL_PATH = './AntiSpoofing_bin_1.5_128.onnx'

# Charger le modèle SVM et l'encodeur
def load_svm_model():
    with open(SVM_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# Charger le modèle anti-spoofing (ONNX)
def load_anti_spoofing_model():
    return ort.InferenceSession(ANTI_SPOOFING_MODEL_PATH)

# Prétraitement pour le modèle anti-spoofing
def preprocess_for_anti_spoofing(frame):
    frame_resized = cv2.resize(frame, (128, 128))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_input = np.transpose(frame_normalized, (2, 0, 1))
    frame_input = np.expand_dims(frame_input, axis=0)
    return frame_input

# Détection de clignement des yeux avec Haar Cascade
def detect_haar_eyes(frame):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
    eye_status = []

    for (ex, ey, ew, eh) in eyes:
        eye_region = gray_frame[ey:ey + eh, ex:ex + ew]
        _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
        white_pixel_ratio = cv2.countNonZero(threshold_eye) / (ew * eh)

        if white_pixel_ratio < 0.4:
            eye_status.append("Closed")
        else:
            eye_status.append("Open")

        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return "Blink detected" if eye_status.count("Closed") > 0 else "No Blink"

# Reconnaissance faciale avec SVM
def recognize_face_with_svm(face_embedding, svm_model, label_encoder):
    predicted_label_index = svm_model.predict([face_embedding])[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    probability = max(svm_model.predict_proba([face_embedding])[0])
    return predicted_label, probability

# Vérification de l'anti-spoofing
def verify_anti_spoofing(frame, model_anti_spoofing):
    input_data = preprocess_for_anti_spoofing(frame)
    inputs = {model_anti_spoofing.get_inputs()[0].name: input_data}
    result = model_anti_spoofing.run(None, inputs)
    spoofing_score = result[0][0][0]
    return spoofing_score  # Spoofing_score près de 0 signifie spoofing, plus élevé signifie authentique

# Fonction pour envoyer le SMS
def send_sms(message):
    conn = http.client.HTTPSConnection("d9ydxr.api.infobip.com")
    payload = json.dumps({
        "messages": [
            {
                "destinations": [{"to": "212688552708"}],  # Numéro de destinataire
                "from": "ServiceSMS",
                "text": message
            }
        ]
    })
    headers = {
        'Authorization': 'App afe5bc66d5b3be2cc4c2c52cd0d395c4-893a5145-1842-4c3a-be84-ff70d3ebfb42',  # Remplacez par votre clé API
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    conn.request("POST", "/sms/2/text/advanced", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))

# Vérification combinée de l'anti-spoofing et du clignement des yeux
def verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    results = []
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        # Extraire et préparer le visage
        face_crop = rgb_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, (160, 160))
        embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

        # Reconnaissance faciale avec SVM
        label, probability = recognize_face_with_svm(embedding, svm_model, label_encoder)

        # Détection du clignement des yeux
        blink_status = detect_haar_eyes(frame)

        # Vérification anti-spoofing
        spoofing_score = verify_anti_spoofing(frame, anti_spoofing_model)
        is_authentic = spoofing_score > 0.5  # Si le score d'authenticité est > 0.5, on considère que ce n'est pas un spoofing

        # Condition combinée : Si l'anti-spoofing est authentique et qu'un clignement des yeux est détecté
        if is_authentic and blink_status == "Blink detected":
            combined_status = f"{label} ({probability:.2f}) - Blink Detected and Authentic"
            # Insérer dans la base de données
            insert_into_db(label)
        elif is_authentic and blink_status == "No Blink":
            combined_status = f"{label} ({probability:.2f}) - No Blink, but Authentic"
        elif not is_authentic and blink_status == "No Blink":
            print(f"Spoofing detected for {label}!")  # Print statement for debugging
            combined_status = f"Spoofing detected for {label}!"
            # Envoi du SMS avec le label de spoofing
            send_sms(f"Spoofing detected for {label}!")  # Envoi du message SMS
        elif not is_authentic:
            combined_status = f"Spoofing detected!"

        # Annoter l'image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, combined_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        results.append({"label": label, "probability": probability, "eye_status": blink_status, "anti_spoofing": is_authentic})

    return frame, results


import sqlite3
from datetime import datetime, timedelta

# Fonction pour insérer des données dans la base SQLite seulement après 2 minutes
def insert_into_db(name):
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    # Créer la table si elle n'existe pas
    cursor.execute('''CREATE TABLE IF NOT EXISTS et (
                        name TEXT,
                        timestamp TEXT,
                        status INTEGER
                    )''')

    # Vérifier la dernière détection enregistrée pour ce nom
    cursor.execute("SELECT timestamp FROM et WHERE name = ? ORDER BY timestamp DESC LIMIT 1", (name,))
    last_entry = cursor.fetchone()

    if last_entry:
        # Convertir le dernier horodatage en objet datetime
        last_time = datetime.strptime(last_entry[0], '%Y-%m-%d %H:%M:%S')
        
        # Vérifier si moins de 2 minutes se sont écoulées
        if datetime.now() - last_time < timedelta(minutes=2):
            print(f"{name} a déjà été enregistré récemment. Enregistrement ignoré.")
            conn.close()
            return  # Ignorer l'enregistrement

    # Déterminer le nouveau statut (alterner entre 0 et 1)
    cursor.execute("SELECT status FROM et WHERE name = ? ORDER BY timestamp DESC LIMIT 1", (name,))
    last_status = cursor.fetchone()
    new_status = 1 if last_status is None or last_status[0] == 0 else 0

    # Insérer les informations dans la base
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO et (name, timestamp, status) VALUES (?, ?, ?)", (name, timestamp, new_status))
    conn.commit()
    conn.close()

    print(f"Enregistrement effectué pour {name} avec le statut {new_status}.")


# Stream vidéo en temps réel depuis l'ESP32-CAM
def start_video_stream_from_esp32():
    URL = "http://192.168.137.122"  # Remplace par l'URL de ton ESP32-CAM
    
    # Charger les modèles
    svm_model, label_encoder = load_svm_model()
    anti_spoofing_model = load_anti_spoofing_model()

    while True:
        try:
            # Effectuer une requête HTTP pour récupérer l'image
            response = requests.get(f"{URL}/capture", stream=True)
            
            # Vérifier si la requête a réussi
            if response.status_code == 200:
                # Convertir l'image binaire en tableau NumPy
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                
                # Décoder l'image en utilisant OpenCV
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Traiter le cadre pour la reconnaissance faciale, le clignement des yeux et l'anti-spoofing
                    frame_with_annotations, results = verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model)

                    # Afficher le cadre résultant
                    cv2.imshow('ESP32 Video Stream', frame_with_annotations)

                    # Quitter la boucle si 'q' est pressé
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Erreur lors du décodage de l'image")
            else:
                print("Erreur lors de la récupération de l'image.")
        
        except Exception as e:
            print(f"Erreur lors de la récupération de l'image: {e}")
            break

    cv2.destroyAllWindows()

# Démarrer le stream vidéo depuis l'ESP32-CAM
if __name__ == '__main__':
    start_video_stream_from_esp32()
