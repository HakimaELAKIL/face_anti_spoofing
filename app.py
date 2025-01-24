# import os
# import cv2
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from collections import defaultdict
# import onnxruntime as ort  # Pour le modèle anti-spoofing

# # Configuration Flask
# app = Flask(__name__)
# UPLOAD_FOLDER = './uploads'
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Initialisation des modèles
# facenet = FaceNet()
# detector = MTCNN()

# # Suivi des états des yeux pour chaque label de visage
# eye_status_tracker = defaultdict(list)

# # Chemins des modèles
# SVM_MODEL_PATH = 'svm_face_recognition.pkl'
# ANTI_SPOOFING_MODEL_PATH = 'AntiSpoofing_bin_1.5_128.onnx'

# # Vérifier les extensions de fichiers autorisées
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Charger le modèle SVM et l'encodeur
# def load_svm_model():
#     with open(SVM_MODEL_PATH, 'rb') as f:
#         return pickle.load(f)

# # Charger le modèle Anti-spoofing (ONNX)
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

# # Détection de clignement des yeux
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

# # Reconnaissance faciale avec le SVM
# def recognize_face_with_svm(face_embedding, svm_model, label_encoder):
#     predicted_label_index = svm_model.predict([face_embedding])[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#     probability = max(svm_model.predict_proba([face_embedding])[0])
#     return predicted_label, probability

# # Vérification visage, clignement et anti-spoofing
# def verify_face_and_blink(frame, svm_model, label_encoder, anti_spoofing_model):
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

#         # Reconnaissance faciale avec le SVM
#         label, probability = recognize_face_with_svm(embedding, svm_model, label_encoder)

#         # Détection de clignement des yeux
#         blink_status = detect_haar_eyes(frame)

#         # Vérification anti-spoofing
#         frame_input = preprocess_for_anti_spoofing(face_crop)
#         inputs = {anti_spoofing_model.get_inputs()[0].name: frame_input}
#         spoofing_output = anti_spoofing_model.run(None, inputs)
#         is_real_face = spoofing_output[0][0][0] > 0.5
#         spoofing_status = "Real Face" if is_real_face else "Spoofed Face"

#         # Suivi des états des yeux
#         if label in eye_status_tracker:
#             previous_status = eye_status_tracker[label]
#             if previous_status == blink_status:
#                 eye_tracker_status = "Spoof Attack Detected"
#             else:
#                 eye_tracker_status = f"Eye status varies: {blink_status}"
#         else:
#             eye_tracker_status = f"First detection: {blink_status}"

#         eye_status_tracker[label] = blink_status

#         # Annoter l'image
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         label_text = f"{label} ({probability:.2f}) - {spoofing_status} - {eye_tracker_status}"
#         cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         results.append({"label": label, "probability": probability, "spoofing_status": spoofing_status, "eye_status": eye_tracker_status})

#     return frame, results

# # Route pour traiter une image
# @app.route('/upload_frames', methods=['POST'])
# def upload_image():
#     if 'imageFile' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['imageFile']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Charger les modèles
#         svm_model, label_encoder = load_svm_model()
#         anti_spoofing_model = load_anti_spoofing_model()

#         # Lire l'image et effectuer les traitements
#         img = cv2.imread(filepath)
#         img_with_annotations, results = verify_face_and_blink(img, svm_model, label_encoder, anti_spoofing_model)

#         # Sauvegarder l'image annotée
#         processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
#         cv2.imwrite(processed_filepath, img_with_annotations)

#         return jsonify({"message": "Image processed successfully", "results": results}), 200

#     return jsonify({"error": "Invalid file type"}), 400

# # Lancer l'application Flask
# if __name__ == '__main__':
#     app.run(debug=True, host='192.168.1.58', port=9000)

# import os
# import cv2
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from collections import defaultdict
# import onnxruntime as ort  # For anti-spoofing model

# # Configuration Flask
# app = Flask(__name__)
# UPLOAD_FOLDER = './uploads'
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Initialization of models
# facenet = FaceNet()
# detector = MTCNN()

# # Eye status tracker for each face label
# eye_status_tracker = defaultdict(list)

# # Model paths
# SVM_MODEL_PATH = 'svm_face_recognition.pkl'
# ANTI_SPOOFING_MODEL_PATH = 'AntiSpoofing_bin_1.5_128.onnx'

# # Global variables to store results of the first and second requests
# first_request_results = None
# second_request_results = None

# # Check for allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

# # Route to handle image upload
# @app.route('/upload_frames', methods=['POST'])
# def upload_image():
#     global first_request_results, second_request_results
    
#     if 'imageFile' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['imageFile']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Load models
#         svm_model, label_encoder = load_svm_model()
#         anti_spoofing_model = load_anti_spoofing_model()

#         # Read image and process it
#         img = cv2.imread(filepath)
#         img_with_annotations, results = verify_face_and_blink(img, svm_model, label_encoder, anti_spoofing_model)

#         # Save annotated image
#         processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
#         cv2.imwrite(processed_filepath, img_with_annotations)

#         # Store results of the first request
#         if first_request_results is None:
#             first_request_results = results
#             return jsonify({"message": "First request processed successfully", "results": results}), 200
#         else:
#             # Compare with second request results
#             second_request_results = results

#             # Detect any change in spoofing status
#             changes_detected = []
#             for first_result, second_result in zip(first_request_results, second_request_results):
#                 if first_result['spoofing_status'] != second_result['spoofing_status']:
#                     changes_detected.append({
#                         'label': first_result['label'],
#                         'first_spoofing_status': first_result['spoofing_status'],
#                         'second_spoofing_status': second_result['spoofing_status']
#                     })

#             if changes_detected:
#                 return jsonify({"message": "Spoofing status changed", "changes": changes_detected}), 200
#             else:
#                 return jsonify({"message": "No change in spoofing status between requests"}), 200

#     return jsonify({"error": "Invalid file type"}), 400

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True, host='192.168.1.58', port=9000)

# import os
# import cv2
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from collections import defaultdict
# import onnxruntime as ort  # For anti-spoofing model
# from datetime import datetime
# import json

# # Configuration Flask
# app = Flask(__name__)
# UPLOAD_FOLDER = './uploads'
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Initialization of models
# facenet = FaceNet()
# detector = MTCNN()

# # Eye status tracker for each face label
# eye_status_tracker = defaultdict(list)

# # Model paths
# SVM_MODEL_PATH = 'svm_face_recognition.pkl'
# ANTI_SPOOFING_MODEL_PATH = 'AntiSpoofing_bin_1.5_128.onnx'

# # Global variable to store the first request results
# first_request_results = None

# # File paths for storing logs
# REAL_FACES_LOG_PATH = 'real_faces_log.json'
# SPOOFED_FACES_LOG_PATH = 'spoofed_faces_log.json'

# # Check for allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

#         # Log to the appropriate file based on spoofing status
#         log_entry = {
#             "label": label,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "spoofing_status": spoofing_status
#         }

#         if spoofing_status == "Spoofed Face":
#             # Save to spoofed faces log
#             log_to_json(SPOOFED_FACES_LOG_PATH, log_entry)
#         else:
#             # Save to real faces log
#             log_to_json(REAL_FACES_LOG_PATH, log_entry)

#     return frame, results

# # Function to log data to a JSON file
# def log_to_json(file_path, log_entry):
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             logs = json.load(f)
#     else:
#         logs = []

#     logs.append(log_entry)

#     with open(file_path, 'w') as f:
#         json.dump(logs, f, indent=4)

# # Route to handle image upload
# @app.route('/upload_frames', methods=['POST'])
# def upload_image():
#     global first_request_results
    
#     if 'imageFile' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['imageFile']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Load models
#         svm_model, label_encoder = load_svm_model()
#         anti_spoofing_model = load_anti_spoofing_model()

#         # Read image and process it
#         img = cv2.imread(filepath)
#         img_with_annotations, results = verify_face_and_blink(img, svm_model, label_encoder, anti_spoofing_model)

#         # Save annotated image
#         processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
#         cv2.imwrite(processed_filepath, img_with_annotations)

#         # If the identity is not spoofed, no need to treat the second request
#         if first_request_results is None:
#             first_request_results = results
#             return jsonify({"message": "First request processed successfully", "results": results}), 200
#         else:
#             # Compare only if the identity is spoofed
#             for result in results:
#                 if result['spoofing_status'] == 'Spoofed Face':
#                     return jsonify({"message": "Spoofing detected in second request", "results": results}), 200
#             return jsonify({"message": "Identity is real, no further treatment needed."}), 200

#     return jsonify({"error": "Invalid file type"}), 400

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True, host='192.168.1.58', port=9000)

from flask import Flask, request, jsonify, render_template
import sqlite3
import base64
import os

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_images', methods=['POST'])
def upload_images():
    if not request.is_json:
        return jsonify({"message": "Invalid request format"}), 400

    data = request.get_json()
    images = data.get('images', [])

    if not images or len(images) != 10:
        return jsonify({"message": "10 images are required"}), 400

    for i, img_base64 in enumerate(images):
        try:
            # Décoder l'image base64
            img_data = base64.b64decode(img_base64)
            img_path = os.path.join(UPLOAD_FOLDER, f'image_{i+1}.jpg')

            # Sauvegarder l'image
            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)

        except Exception as e:
            return jsonify({"message": f"Error saving image {i+1}: {str(e)}"}), 500

    return jsonify({"message": "Images saved successfully"}), 200
# Fonction pour se connecter à la base de données SQLite
def get_db_connection():
    conn = sqlite3.connect('faces.db')
    conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
    return conn

# Route pour afficher les données
@app.route('/')
def index():
    try:
        conn = get_db_connection()
        users = conn.execute('SELECT * FROM user_data').fetchall()  # Récupérer toutes les lignes de la table users
        conn.close()
        if users:
            return render_template('index.html', users=users)
        else:
            return "Aucun utilisateur trouvé dans la base de données."
    except Exception as e:
        return f"Une erreur est survenue : {e}"


@app.route('/Hakima')
def index1():
    try:
        conn = get_db_connection()
        # Récupérer tous les utilisateurs de la base de données
        users = conn.execute('SELECT * FROM et').fetchall()
        conn.close()

        # Groupement des utilisateurs par nom
        grouped_users = {}
        for user in users:
            group_name = user['name']
            if group_name not in grouped_users:
                grouped_users[group_name] = []
            grouped_users[group_name].append(user)

        if users:
            return render_template('index1.html', users=grouped_users)
        else:
            return "Aucun utilisateur trouvé dans la base de données."
    except Exception as e:
        return f"Une erreur est survenue : {e}"

    

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)