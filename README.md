# smart-attendance-face-match-service

A Python microservice that handles face verification for the Smart Attendance System. It receives two images — the student's stored photo and the selfie taken during attendance — runs a liveness check on the selfie, and compares the two faces. The Spring Boot backend calls this service over HTTP after a valid QR scan.

Part of a three-repo system:

- **[smart-attendance](https://github.com/cp-team23/smart-attendance)** — Spring Boot backend + web UI
- **[SmartAttendanceApp](https://github.com/cp-team23/SmartAttendanceApp)** — Kotlin Android app for students
- **[smart-attendance-face-match-service](https://github.com/cp-team23/smart-attendance-face-match-service)** ← you are here

**Live:** https://smart-attendance-face-match-service.onrender.com/health

---

## What it does

When a student scans a QR code and takes a selfie, the Spring Boot backend forwards both images here — the stored photo (image1) and the selfie (image2). The service then does two things in sequence:

**Liveness check** — before comparing faces, it verifies that image2 is a real person in front of a camera and not a printed photo or screen recording. Several signals are checked:

- **Antifake score** from InsightFace's built-in anti-spoofing module — if the score is below 0.6, it's rejected immediately
- **Sharpness** — blurry images (Laplacian variance below 30) are flagged as likely screen captures
- **Texture** — LBP (Local Binary Pattern) analysis checks for flat, uniform textures typical of printed photos
- **Reflection** — highlight distribution on the face is checked for screen glare patterns
- **Face geometry** — dlib's 68-point facial landmark model checks for plausible facial symmetry, eye aspect ratio, and face proportions

If any check fails, the service returns `same_person: false` without proceeding to face comparison.

**Face comparison** — if liveness passes, InsightFace extracts embeddings from both images and computes cosine similarity. A similarity of 0.5 or above is considered a match.

Models are lazy-loaded on the first request so the service starts up quickly.

---

## API

### `GET /health`

Simple health check.

```json
{ "status": "ok" }
```

### `POST /face-match`

Accepts a multipart form with two image files.

| Field | Type | Description |
|---|---|---|
| `image1` | file | Stored student photo |
| `image2` | file | Selfie taken during attendance |

**Response (match):**
```json
{
  "similarity": 0.7812,
  "same_person": true
}
```

**Response (liveness failed or no match):**
```json
{
  "similarity": 0.0,
  "same_person": false
}
```

**Response (no face detected or missing image):**
```json
{ "error": "No face detected" }
```

---

## Tech stack

| | |
|---|---|
| Framework | Flask + Flask-CORS |
| Face detection & recognition | InsightFace (buffalo_sc model, CPU) |
| Anti-spoofing | InsightFace antifake module |
| Landmark detection | dlib (shape_predictor_68_face_landmarks) |
| Image processing | OpenCV, NumPy |
| Hosting | Render.com |

Thread counts are capped via environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) to keep CPU usage reasonable on a free-tier instance.

---

## Running locally

### Prerequisites

- Python 3.9+
- `shape_predictor_68_face_landmarks.dat` in the project root (download from the [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2))

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

The service starts on port `10000` by default. Set the `PORT` environment variable to change it.

```bash
PORT=8000 python app.py
```

### Test the endpoint

```bash
curl -X POST http://localhost:10000/face-match \
  -F "image1=@stored_photo.jpg" \
  -F "image2=@selfie.jpg"
```

---

## Related repositories

- Backend: https://github.com/cp-team23/smart-attendance
- Android app: https://github.com/cp-team23/SmartAttendanceApp
