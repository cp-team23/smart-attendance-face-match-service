import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from flask import Flask, request, jsonify
from flask_cors import CORS
from insightface.app import FaceAnalysis
import cv2
import numpy as np

API_KEY = "SMART_ATTENDANCE_FACE_KEY"

app = Flask(__name__)
CORS(app)

face_app = FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(320, 320))

dummy = np.zeros((320, 320, 3), dtype=np.uint8)
for _ in range(2):
    face_app.get(dummy)

print("InsightFace loaded & warmed up")

def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def read_image(file):
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def normalize_image(img):
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > 800:
        scale = 800 / max_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

@app.before_request
def check_api_key():
    client_key = request.headers.get("X-API-KEY")

    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401


@app.route("/face-match", methods=["POST"])
def face_match():

    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Both images required"}), 400

    img1 = read_image(request.files["image1"])
    img2 = read_image(request.files["image2"])

    if img1 is None or img2 is None:
        return jsonify({"error": "Invalid image"}), 400

    img1 = normalize_image(img1)
    img2 = normalize_image(img2)

    faces1 = face_app.get(img1)
    faces2 = face_app.get(img2)

    if not faces1 or not faces2:
        return jsonify({"error": "No face detected"}), 400

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    similarity = cosine_similarity(emb1, emb2)

    return jsonify({
        "similarity": similarity,
        "same_person": bool(similarity >= 0.5)
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )

# python -m venv venv
# venv\scripts\activate
# pip install -r requirements.txt
# python app.py