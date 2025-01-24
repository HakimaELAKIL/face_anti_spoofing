# import cv2
# import numpy as np
# import onnxruntime as ort  # For anti-spoofing model
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# from collections import defaultdict
# import pickle
# import sqlite3
# from datetime import datetime, timedelta
# import requests
# import http.client
# import json
# from flask import Flask, Response, render_template_string

# app = Flask(__name__)

# # URL of ESP32-CAM
# URL = "http://192.168.137.140"  # Replace with your ESP32-CAM URL

# # Initializing models
# facenet = FaceNet()
# detector = MTCNN()

# # Models paths
# SVM_MODEL_PATH = './svm_face_recognition.pkl'
# ANTI_SPOOFING_MODEL_PATH = './AntiSpoofing_bin_1.5_128.onnx'

# # Loading the models
# def load_svm_model():
#     with open(SVM_MODEL_PATH, 'rb') as f:
#         return pickle.load(f)

# def load_anti_spoofing_model():
#     return ort.InferenceSession(ANTI_SPOOFING_MODEL_PATH)

# svm_model, label_encoder = load_svm_model()
# anti_spoofing_model = load_anti_spoofing_model()

# # Preprocess function for anti-spoofing
# def preprocess_for_anti_spoofing(frame):
#     frame_resized = cv2.resize(frame, (128, 128))
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#     frame_normalized = frame_rgb.astype(np.float32) / 255.0
#     frame_input = np.transpose(frame_normalized, (2, 0, 1))
#     frame_input = np.expand_dims(frame_input, axis=0)
#     return frame_input

# # Eye blink detection
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

# # Recognize face using SVM
# def recognize_face_with_svm(face_embedding, svm_model, label_encoder):
#     predicted_label_index = svm_model.predict([face_embedding])[0]
#     predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#     probability = max(svm_model.predict_proba([face_embedding])[0])
#     return predicted_label, probability

# # Anti-spoofing verification
# def verify_anti_spoofing(frame, model_anti_spoofing):
#     input_data = preprocess_for_anti_spoofing(frame)
#     inputs = {model_anti_spoofing.get_inputs()[0].name: input_data}
#     result = model_anti_spoofing.run(None, inputs)
#     spoofing_score = result[0][0][0]
#     return spoofing_score  # Score near 0 means spoofing, higher means authentic

# # Function to send SMS when spoofing is detected
# def send_sms(message):
#     conn = http.client.HTTPSConnection("d9ydxr.api.infobip.com")
#     payload = json.dumps({
#         "messages": [
#             {
#                 "destinations": [{"to": "212688552708"}],  # Recipient number
#                 "from": "ServiceSMS",
#                 "text": message
#             }
#         ]
#     })
#     headers = {
#         'Authorization': 'App afe5bc66d5b3be2cc4c2c52cd0d395c4-893a5145-1842-4c3a-be84-ff70d3ebfb42',  # Your API key
#         'Content-Type': 'application/json',
#         'Accept': 'application/json'
#     }
#     conn.request("POST", "/sms/2/text/advanced", payload, headers)
#     res = conn.getresponse()
#     data = res.read()
#     print(data.decode("utf-8"))

# # Combine face, blink, and anti-spoofing checks
# def verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(rgb_frame)

#     results = []
#     for face in faces:
#         x, y, w, h = face['box']
#         x, y = max(0, x), max(0, y)

#         # Extract and prepare face
#         face_crop = rgb_frame[y:y + h, x:x + w]
#         face_resized = cv2.resize(face_crop, (160, 160))
#         embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

#         # Recognize face with SVM
#         label, probability = recognize_face_with_svm(embedding, svm_model, label_encoder)

#         # Eye blink detection
#         blink_status = detect_haar_eyes(frame)

#         # Anti-spoofing check
#         spoofing_score = verify_anti_spoofing(frame, anti_spoofing_model)
#         is_authentic = spoofing_score > 0.5  # If the authenticity score is > 0.5, it's considered authentic

#         # Combined condition: If the anti-spoofing is authentic and blink is detected
#         if is_authentic and blink_status == "Blink detected":
#             combined_status = f"{label} ({probability:.2f}) - Blink Detected and Authentic"
#             insert_into_db(label)
#         elif is_authentic and blink_status == "No Blink":
#             combined_status = f"{label} ({probability:.2f}) - No Blink, but Authentic"
#         elif not is_authentic and blink_status == "No Blink":
#             print(f"Spoofing detected for {label}!")
#             combined_status = f"Spoofing detected for {label}!"
#             send_sms(f"Spoofing detected for {label}!")  # Send SMS on spoofing
#         elif not is_authentic:
#             combined_status = f"Spoofing detected!"

#         # Annotate image
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, combined_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         results.append({"label": label, "probability": probability, "eye_status": blink_status, "anti_spoofing": is_authentic})

#     return frame, results

# # Database insertion function (to avoid duplicate entries within 2 minutes)
# def insert_into_db(name):
#     conn = sqlite3.connect('faces.db')
#     cursor = conn.cursor()

#     cursor.execute('''CREATE TABLE IF NOT EXISTS et (
#                         name TEXT,
#                         timestamp TEXT,
#                         status INTEGER
#                     )''')

#     cursor.execute("SELECT timestamp FROM et WHERE name = ? ORDER BY timestamp DESC LIMIT 1", (name,))
#     last_entry = cursor.fetchone()

#     if last_entry:
#         last_time = datetime.strptime(last_entry[0], '%Y-%m-%d %H:%M:%S')
#         if datetime.now() - last_time < timedelta(minutes=2):
#             print(f"{name} has been recorded recently. Skipping entry.")
#             conn.close()
#             return  # Skip the record

#     cursor.execute("SELECT status FROM et WHERE name = ? ORDER BY timestamp DESC LIMIT 1", (name,))
#     last_status = cursor.fetchone()
#     new_status = 1 if last_status is None or last_status[0] == 0 else 0

#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     cursor.execute("INSERT INTO et (name, timestamp, status) VALUES (?, ?, ?)", (name, timestamp, new_status))
#     conn.commit()
#     conn.close()
#     print(f"Record added for {name} with status {new_status}.")

# # Video stream from ESP32-CAM
# def start_video_stream_from_esp32():
#     while True:
#         try:
#             response = requests.get(f"{URL}/capture", stream=True)
#             if response.status_code == 200:
#                 image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
#                 frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#                 if frame is not None:
#                     frame_with_annotations, results = verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model)
#                     cv2.imshow('ESP32 Video Stream', frame_with_annotations)

#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
#                 else:
#                     print("Error decoding image.")
#             else:
#                 print("Error fetching image.")
#         except Exception as e:
#             print(f"Error fetching image: {e}")
#             break

#     cv2.destroyAllWindows()

# # Function to generate frames from the ESP32-CAM
# def generate_frames():
#     while True:
#         try:
#             # Request the image stream from the ESP32-CAM
#             response = requests.get(f"{URL}/capture", stream=True)
#             if response.status_code == 200:
#                 # Convert the byte content into an image frame
#                 image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
#                 frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
#                 if frame is not None:
#                     # Process the frame (e.g., face recognition, blink detection)
#                     frame_with_annotations, results = verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model)
                    
#                     # Encode the frame to JPEG for streaming
#                     ret, buffer = cv2.imencode('.jpg', frame_with_annotations)
#                     frame_bytes = buffer.tobytes()

#                     # Yield the frame in a format that Flask can stream
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
#                 else:
#                     print("Error decoding image.")
#             else:
#                 print("Error fetching image.")
#         except Exception as e:
#             print(f"Error fetching image: {e}")
#             break

# # Flask route to stream video
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     html = """
#     <!doctype html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>ESP32-CAM Video Stream</title>
#         <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
#         <style>
#             body {
#                 background-color: #f8f9fa;
#             }
#             .stream-container {
#                 display: flex;
#                 justify-content: center;
#                 align-items: center;
#                 height: 80vh;
#                 flex-direction: column;
#             }
#             img {
#                 border: 2px solid #6c757d;
#                 border-radius: 8px;
#                 box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
#             }
#             footer {
#                 text-align: center;
#                 margin-top: 20px;
#                 font-size: 0.9rem;
#                 color: #6c757d;
#             }
#         </style>
#     </head>
#     <body>
#         <header class="bg-dark text-white text-center py-3">
#             <h1>ESP32-CAM Video Stream</h1>
#         </header>
#         <div class="stream-container">
#             <img src="/video_feed" class="img-fluid" alt="ESP32-CAM Stream">
#             <p class="mt-3">Real-time video with face recognition and blink detection.</p>
#             <button class="btn btn-primary" onclick="location.reload()">Refresh Stream</button>
#         </div>
#         <footer>
#             <p>&copy; 2024 - ESP32-CAM Project by Hakima</p>
#         </footer>
#     </body>
#     </html>
#     """
#     return render_template_string(html)


# if __name__ == '__main__':
#     app.run(debug=True, port='1000')

import cv2
import numpy as np
import onnxruntime as ort  # For anti-spoofing model
from keras_facenet import FaceNet
from mtcnn import MTCNN
from collections import defaultdict
import pickle
import sqlite3
from datetime import datetime, timedelta
import requests
import http.client
import json
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# URL of ESP32-CAM
URL = "http://192.168.137.122:81"  # Replace with your ESP32-CAM URL

# Initializing models
facenet = FaceNet()
detector = MTCNN()

# Models paths
SVM_MODEL_PATH = './svm_face_recognition.pkl'
ANTI_SPOOFING_MODEL_PATH = './AntiSpoofing_bin_1.5_128.onnx'

# Loading the models
def load_svm_model():
    with open(SVM_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def load_anti_spoofing_model():
    return ort.InferenceSession(ANTI_SPOOFING_MODEL_PATH)

svm_model, label_encoder = load_svm_model()
anti_spoofing_model = load_anti_spoofing_model()

# Preprocess function for anti-spoofing
def preprocess_for_anti_spoofing(frame):
    frame_resized = cv2.resize(frame, (128, 128))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_input = np.transpose(frame_normalized, (2, 0, 1))
    frame_input = np.expand_dims(frame_input, axis=0)
    return frame_input

# Eye blink detection
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

# Recognize face using SVM
def recognize_face_with_svm(face_embedding, svm_model, label_encoder):
    predicted_label_index = svm_model.predict([face_embedding])[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    probability = max(svm_model.predict_proba([face_embedding])[0])
    return predicted_label, probability

# Anti-spoofing verification
def verify_anti_spoofing(frame, model_anti_spoofing):
    input_data = preprocess_for_anti_spoofing(frame)
    inputs = {model_anti_spoofing.get_inputs()[0].name: input_data}
    result = model_anti_spoofing.run(None, inputs)
    spoofing_score = result[0][0][0]
    return spoofing_score  # Score near 0 means spoofing, higher means authentic

# Function to send SMS when spoofing is detected
def send_sms(message):
    conn = http.client.HTTPSConnection("d9ydxr.api.infobip.com")
    payload = json.dumps({
        "messages": [
            {
                "destinations": [{"to": "212688552708"}],  # Recipient number
                "from": "ServiceSMS",
                "text": message
            }
        ]
    })
    headers = {
        'Authorization': 'App afe5bc66d5b3be2cc4c2c52cd0d395c4-893a5145-1842-4c3a-be84-ff70d3ebfb42',  # Your API key
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    conn.request("POST", "/sms/2/text/advanced", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))

# Combine face, blink, and anti-spoofing checks
def verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    results = []
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        # Extract and prepare face
        face_crop = rgb_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, (160, 160))
        embedding = facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]

        # Recognize face with SVM
        label, probability = recognize_face_with_svm(embedding, svm_model, label_encoder)

        # Eye blink detection
        blink_status = detect_haar_eyes(frame)

        # Anti-spoofing check
        spoofing_score = verify_anti_spoofing(frame, anti_spoofing_model)
        is_authentic = spoofing_score > 0.5  # If the authenticity score is > 0.5, it's considered authentic

        # Combined condition: If the anti-spoofing is authentic and blink is detected
        if is_authentic and blink_status == "Blink detected":
            combined_status = f"{label} ({probability:.2f}) - Blink Detected and Authentic"
            insert_into_db(label)
        elif is_authentic and blink_status == "No Blink":
            combined_status = f"{label} ({probability:.2f}) - No Blink, but Authentic"
        elif not is_authentic and blink_status == "No Blink":
            print(f"Spoofing detected for {label}!")
            combined_status = f"Spoofing detected for {label}!"
            send_sms(f"Spoofing detected for {label}!")  # Send SMS on spoofing
        elif not is_authentic:
            combined_status = f"Spoofing detected!"

        # Annotate image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, combined_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        results.append({"label": label, "probability": probability, "eye_status": blink_status, "anti_spoofing": is_authentic})

    return frame, results

# Database insertion function (to avoid duplicate entries within 2 minutes)
def insert_into_db(name):
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS et (
                        name TEXT,
                        timestamp TEXT,
                        status INTEGER
                    )''')

    cursor.execute("SELECT timestamp FROM et WHERE name = ? ORDER BY timestamp DESC LIMIT 1", (name,))
    last_entry = cursor.fetchone()

    if last_entry:
        last_time = datetime.strptime(last_entry[0], '%Y-%m-%d %H:%M:%S')
        if datetime.now() - last_time < timedelta(minutes=2):
            print(f"{name} has been recorded recently. Skipping entry.")
            conn.close()
            return  # Skip the record

    cursor.execute("SELECT status FROM et WHERE name = ? ORDER BY timestamp DESC LIMIT 1", (name,))
    last_status = cursor.fetchone()
    new_status = 1 if last_status is None or last_status[0] == 0 else 0

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO et (name, timestamp, status) VALUES (?, ?, ?)", (name, timestamp, new_status))
    conn.commit()
    conn.close()
    print(f"Record added for {name} with status {new_status}.")

# Video stream from ESP32-CAM
def start_video_stream_from_esp32():
    while True:
        try:
            response = requests.get(f"{URL}/stream", stream=True)
            if response.status_code == 200:
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_with_annotations, results = verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model)
                    cv2.imshow('ESP32 Video Stream', frame_with_annotations)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Error decoding image.")
            else:
                print("Error fetching image.")
        except Exception as e:
            print(f"Error fetching image: {e}")
            break

    cv2.destroyAllWindows()

def generate_frames():
    # Utilisation de cv2.VideoCapture pour lire le flux continu
    stream_url = f"{URL}/stream"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir le flux vidéo.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Erreur : Impossible de lire une image du flux.")
            break

        # Traitement du frame (reconnaissance faciale, détection de clignements, etc.)
        frame_with_annotations, results = verify_face_and_blink_and_anti_spoofing(frame, svm_model, label_encoder, anti_spoofing_model)

        # Encodage en JPEG pour la diffusion
        ret, buffer = cv2.imencode('.jpg', frame_with_annotations)
        frame_bytes = buffer.tobytes()

        # Génération des frames pour Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()


# Flask route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ESP32-CAM Video Stream</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
            }
            .stream-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 80vh;
                flex-direction: column;
            }
            img {
                border: 2px solid #6c757d;
                border-radius: 8px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }
            footer {
                text-align: center;
                margin-top: 20px;
                font-size: 0.9rem;
                color: #6c757d;
            }
        </style>
    </head>
    <body>
        <header class="bg-dark text-white text-center py-3">
            <h1>ESP32-CAM Video Stream</h1>
        </header>
        <div class="stream-container">
            <img src="/video_feed" class="img-fluid" alt="ESP32-CAM Stream">
            <p class="mt-3">Real-time video with face recognition and blink detection.</p>
            <button class="btn btn-primary" onclick="location.reload()">Refresh Stream</button>
        </div>
        <footer>
            <p>&copy; 2024 - ESP32-CAM</p>
        </footer>
    </body>
    </html>
    """
    return render_template_string(html)


if __name__ == '__main__':
    app.run(debug=True, port='1000')