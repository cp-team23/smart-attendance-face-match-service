import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from flask import Flask, request, jsonify
from flask_cors import CORS
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import dlib

app = Flask(__name__)
CORS(app)

# Global variables - lazy loaded
face_app = None
dlib_detector = None
dlib_predictor = None


def load_models():
    global face_app, dlib_detector, dlib_predictor
    if face_app is None:
        print("Loading models...")
        face_app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "antifake"]
        )
        face_app.prepare(ctx_id=0, det_size=(320, 320))
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(2):
            face_app.get(dummy)
        dlib_detector = dlib.get_frontal_face_detector()
        dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("Models loaded!")


# ─────────────────────────────────────────────
#  LIVENESS HELPERS
# ─────────────────────────────────────────────

def compute_lbp_score(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    radius, n_points = 1, 8
    rows, cols = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center = gray[i, j]
            binary = ""
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                ni = int(round(i - radius * np.sin(angle)))
                nj = int(round(j + radius * np.cos(angle)))
                binary += "1" if gray[ni, nj] >= center else "0"
            lbp[i, j] = int(binary, 2)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return float(np.clip((entropy - 3.0) / (7.5 - 3.0), 0.0, 1.0))


def compute_reflection_score(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    _, highlight_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    highlight_ratio = np.sum(highlight_mask > 0) / (gray.size + 1e-7)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(highlight_mask)
    blob_count = num_labels - 1
    if highlight_ratio < 0.001:
        return 0.35
    elif highlight_ratio > 0.20:
        return 0.20
    elif 0.005 <= highlight_ratio <= 0.10 and 1 <= blob_count <= 8:
        return 0.80
    return 0.50


def compute_geometry_score(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rect = dlib.rectangle(0, 0, w, h)
    try:
        shape = dlib_predictor(gray, rect)
    except Exception:
        return 0.5

    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    center_x = np.mean(pts[:, 0])
    left_pts = pts[pts[:, 0] < center_x]
    right_pts = pts[pts[:, 0] >= center_x]
    left_spread = np.std(left_pts[:, 0]) if len(left_pts) > 0 else 0
    right_spread = np.std(right_pts[:, 0]) if len(right_pts) > 0 else 0
    symmetry_score = float(np.clip(
        min(left_spread, right_spread) / (max(left_spread, right_spread) + 1e-7), 0.0, 1.0
    ))

    def eye_aspect_ratio(eye_pts):
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (A + B) / (2.0 * C + 1e-7)

    left_ear = eye_aspect_ratio(pts[36:42])
    right_ear = eye_aspect_ratio(pts[42:48])
    avg_ear = (left_ear + right_ear) / 2.0
    ear_score = float(np.clip(1.0 - abs(avg_ear - 0.30) / 0.20, 0.0, 1.0))

    bbox_w = pts[:, 0].max() - pts[:, 0].min()
    bbox_h = pts[:, 1].max() - pts[:, 1].min()
    aspect = min(bbox_w, bbox_h) / (max(bbox_w, bbox_h) + 1e-7)
    aspect_score = float(np.clip(1.0 - abs(aspect - 0.72) / 0.3, 0.0, 1.0))

    return float(symmetry_score * 0.4 + ear_score * 0.4 + aspect_score * 0.2)


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


def check_liveness(img, face):
    result = {"is_live": True, "reason": None, "score": 1.0}

    if hasattr(face, "antifake") and face.antifake is not None:
        spoof_score = float(face.antifake)
        result["score"] = spoof_score
        if spoof_score < 0.6:
            result["is_live"] = False
            result["reason"] = f"Spoof detected (antifake score: {spoof_score:.2f})"
            return result

    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        return result

    face_resized = cv2.resize(face_crop, (256, 256))

    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 30:
        result["is_live"] = False
        result["reason"] = f"Image too blurry (sharpness: {laplacian_var:.1f})"
        return result

    lbp_score = compute_lbp_score(face_resized)
    if lbp_score < 0.38:
        result["is_live"] = False
        result["reason"] = f"Flat texture detected (LBP score: {lbp_score:.2f})"
        return result

    ref_score = compute_reflection_score(face_resized)
    if ref_score < 0.25:
        result["is_live"] = False
        result["reason"] = f"Screen glare detected (reflection score: {ref_score:.2f})"
        return result

    geo_score = compute_geometry_score(face_resized)
    if geo_score < 0.30:
        result["is_live"] = False
        result["reason"] = f"Implausible face geometry (geometry score: {geo_score:.2f})"
        return result

    final_score = (0.45 * lbp_score) + (0.35 * geo_score) + (0.20 * ref_score)
    result["score"] = round(final_score, 4)

    if final_score < 0.45:
        result["is_live"] = False
        result["reason"] = f"Combined liveness check failed (score: {final_score:.2f})"

    return result


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/health")
def health():
    return {"status": "face matching server is running"}


@app.route("/face-match", methods=["POST"])
def face_match():
    load_models()

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

    liveness = check_liveness(img2, faces2[0])
    if not liveness["is_live"]:
        return jsonify({
            "similarity": 0.0,
            "same_person": False
        }), 200

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding
    similarity = cosine_similarity(emb1, emb2)

    return jsonify({
        "similarity": round(similarity, 4),
        "same_person": bool(similarity >= 0.5)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
