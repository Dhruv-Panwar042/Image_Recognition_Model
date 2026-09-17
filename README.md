# VisionMind AI: Full-Stack Computer Vision Service

[![CI](https://github.com/Dhruv-Panwar042/VisionMind-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhruv-Panwar042/VisionMind-AI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8n-00599C?style=flat)](https://ultralytics.com)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini_Flash-4285F4?style=flat&logo=google)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)

A decoupled computer vision service that pairs fast local object detection (**Ultralytics YOLOv8n**) with multimodal LLM reasoning (**Google Gemini**) and automated reporting (**ReportLab**). 

The application is structured as a modular FastAPI backend exposing 12 REST endpoints, served alongside a responsive, light-mode web dashboard.

---

### 🌐 Live Demo & Documentation

* **Live Web App**: [image-recognition-model-9l2e.onrender.com](https://image-recognition-model-9l2e.onrender.com)  
  *(Hosted on Render's free tier with 0.5 shared vCPU and 512 MB RAM. If the instance was idling, please allow ~30 seconds for initial container cold start).*
* **Interactive OpenAPI/Swagger Docs**: [image-recognition-model-9l2e.onrender.com/docs](https://image-recognition-model-9l2e.onrender.com/docs)

---

## Architecture Overview

The system is organized into modular service layers with clear separation of concerns:

```
                    ┌────────────────────────────────────────┐
                    │      Client Dashboard / API Consumer   │
                    │   (HTML5 Canvas Downscaling + Web UI)  │
                    └───────────────────┬────────────────────┘
                                        │ HTTP / JSON Multipart
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Service Layer                                 │
│                                                                                 │
│  [Latency Profiler: X-Process-Time-Ms] ──> [Pydantic v2 Request Validation]     │
│                                                                                 │
│  API Endpoints (/api/v1):                                                       │
│  ├── POST /detect              (Single-pass YOLO detection + base64 overlay)    │
│  ├── POST /detect/batch        (Multi-image sequential processing)              │
│  ├── POST /annotated-image     (JPEG stream with rendered bounding boxes)       │
│  ├── POST /heatmap             (Gaussian confidence density matrix)             │
│  ├── POST /distances           (Normalized centroid Euclidean proximity)        │
│  ├── POST /scene-description   (Natural language scene narrative)               │
│  ├── POST /risk-assessment     (Structured hazard evaluation & recommendations) │
│  ├── POST /qa                  (Visual question answering)                      │
│  ├── POST /compare             (Two-image object delta tracking)                │
│  ├── POST /export/json         (Structured audit JSON export)                   │
│  └── POST /export/pdf          (ReportLab PDF report compilation)               │
│                                                                                 │
│  Internal Services:                                                             │
│  ├── YOLOService               (Singleton YOLOv8n model, imgsz=416 on CPU)      │
│  ├── CVService                 (OpenCV matrix heatmaps, box rendering, math)    │
│  ├── GeminiService             (Multimodal vision with multi-model failover)    │
│  └── ReportService             (ReportLab dynamic table & document compiler)    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Details

### 1. Latency & Performance Optimization
Running computer vision workloads on a resource-constrained cloud container (0.5 vCPU, 512 MB RAM) introduced initial response times of ~35 seconds when testing with large smartphone photos. Two targeted adjustments brought round-trip response times down to **~1.2 seconds**:
* **Client-Side Canvas Downscaling**: Before uploading, photos are automatically resized to a maximum dimension of 800px on an offscreen HTML5 `<canvas>` in the browser. This shrinks image payload from ~4MB down to ~80KB (a 97% reduction), eliminating upload delays and reducing CPU image decode overhead.
* **Single-Pass In-Memory Annotation**: Instead of making two separate network requests (one for detection coordinates and a second to render boxes), the backend renders bounding boxes in memory during the initial detection pass and bundles the resulting JPEG in base64 within the JSON response.
* **CPU Inference Tuning**: Model input size is clamped to `imgsz=416`, cutting floating-point operations (FLOPs) by ~55% on CPU compared to standard 640px input.

### 2. LLM Rate-Limit Handling (HTTP 429 Failover)
Free-tier developer quotas on Google AI Studio enforce strict daily caps (20 requests/day on `gemini-2.5-flash`). To prevent service interruption:
* `GeminiService` implements an automatic failover chain:
  $$\text{gemini-2.5-flash} \longrightarrow \text{gemini-flash-lite-latest} \longrightarrow \text{gemini-flash-latest}$$
* If a 429 quota exhaustion or 503 high-demand exception occurs, the service catches the error and retries with the next available model in the chain before returning a response.

### 3. Spatial Distance & Density Heatmaps
* **Pairwise Centroid Distances**: Calculates normalized Euclidean distances between all detected object centroids $cx = \frac{x_1 + x_2}{2W}$, $cy = \frac{y_1 + y_2}{2H}$, categorizing relative distances into `Very Close`, `Close`, or `Far`.
* **Density Heatmap**: Accumulates detection confidence matrices across the image grid, applies an OpenCV Gaussian blur (`kernel=(51,51)`), normalizes intensities, and blends the resulting `COLORMAP_JET` overlay onto the original image.

---

## REST API Reference

All endpoints are prefixed with `/api/v1`. Interactive documentation is available at `/docs`.

| Method | Endpoint | Description | Input | Output |
|---|---|---|---|---|
| `GET` | `/health` | Service liveness probe | None | `{"status": "healthy"}` |
| `GET` | `/api/v1/classes` | List of 80 detectable COCO classes | None | `["person", "car", ...]` |
| `POST` | `/api/v1/detect` | Object detection + base64 image overlay | `multipart/form-data` | `DetectionResponse` (JSON) |
| `POST` | `/api/v1/detect/batch` | Sequential multi-image batch detection | `multipart/form-data` | `BatchDetectionResponse` (JSON) |
| `POST` | `/api/v1/annotated-image` | Direct JPEG stream with bounding boxes | `multipart/form-data` | `image/jpeg` stream |
| `POST` | `/api/v1/heatmap` | Object density heatmap overlay | `multipart/form-data` | `image/jpeg` stream |
| `POST` | `/api/v1/distances` | Pairwise object proximity calculation | `multipart/form-data` | `DistanceResponse` (JSON) |
| `POST` | `/api/v1/scene-description` | Natural language scene narrative | `multipart/form-data` | `SceneDescriptionResponse` (JSON) |
| `POST` | `/api/v1/risk-assessment` | Scene hazard & safety risk evaluation | `multipart/form-data` | `RiskAssessmentResponse` (JSON) |
| `POST` | `/api/v1/qa` | Grounded visual question answering | `multipart/form-data` | `QAResponse` (JSON) |
| `POST` | `/api/v1/compare` | Two-image class count delta comparison | `multipart/form-data` | `ComparisonResponse` (JSON) |
| `POST` | `/api/v1/export/json` | Full audit data export | `multipart/form-data` | `FullAnalysisExport` (JSON) |
| `POST` | `/api/v1/export/pdf` | Formatted PDF summary report | `multipart/form-data` | `application/pdf` download |

---

## Local Setup & Quickstart

### Prerequisites
* Python 3.10+
* A Google Gemini API Key ([Get one free on Google AI Studio](https://aistudio.google.com))

### 1. Clone the Repository
```bash
git clone https://github.com/Dhruv-Panwar042/VisionMind-AI.git
cd VisionMind-AI
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt / PowerShell):
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment file and add your Gemini API key:
```bash
# On Linux / macOS:
cp .env.example .env

# On Windows:
copy .env.example .env
```
Open `.env` and set your key:
```env
GOOGLE_API_KEY=your_actual_gemini_key_here
GEMINI_MODEL=gemini-2.5-flash
YOLO_MODEL_PATH=yolov8n.pt
YOLO_IMG_SIZE=416
DEFAULT_CONFIDENCE=0.25
PORT=8000
```

### 5. Run the Application
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, open:
* **Web Dashboard**: `http://localhost:8000`
* **Swagger Documentation**: `http://localhost:8000/docs`

---

## Running Tests

Automated tests cover API endpoints, class parsers, schema validations, and error handlers:
```bash
pytest tests/
```

---

## Docker Deployment

To build and run using Docker:
```bash
# Build the Docker image
docker build -t visionmind-ai .

# Run the container
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_actual_gemini_key_here visionmind-ai
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Dhruv Panwar**  
B.Tech Computer Science & Engineering (E-Commerce Technology), Vellore Institute of Technology (VIT)  
[GitHub](https://github.com/Dhruv-Panwar042) · [LinkedIn](https://linkedin.com/in/dhruv-panwar) · [LeetCode](https://leetcode.com/u/DhruvPanwar)
