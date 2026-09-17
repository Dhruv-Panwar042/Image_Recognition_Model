# 🔍 VisionMind AI — Intelligent Image Analysis System (v2.0)

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8n-00599C?style=flat)](https://ultralytics.com)
[![Google Gemini](https://img.shields.io/badge/Multimodal-Gemini_Flash_Fleet-4285F4?style=flat&logo=google)](https://ai.google.dev)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![Live Deployment](https://img.shields.io/badge/Live_Cloud-Render_Online-46E3B7?style=flat&logo=render&logoColor=white)](https://image-recognition-model-9l2e.onrender.com)

An enterprise-grade, decoupled computer vision and multimodal intelligence system. Combines **YOLOv8 Nano** for real-time edge/CPU object detection with **Google Gemini Multimodal Resiliency Fleet** for natural language scene understanding, public safety risk assessment, and spatial proximity reasoning.

Architected with a **production-ready FastAPI REST API backend** serving an **integrated modern light dashboard UI**, featuring automatic Swagger documentation, SLA latency profiling, client-side canvas downscaling, multi-model quota fallback, and dual-format (PDF & JSON) data export.

---

### 🌐 Live Production Deployment
* **Live Interactive Dashboard**: [https://image-recognition-model-9l2e.onrender.com](https://image-recognition-model-9l2e.onrender.com)
* **Interactive OpenAPI/Swagger Docs**: [https://image-recognition-model-9l2e.onrender.com/docs](https://image-recognition-model-9l2e.onrender.com/docs)

---

## 🏛️ System Architecture

```
                       ┌────────────────────────────────────────┐
                       │           Client Applications          │
                       │   (Streamlit Web UI / Mobile / cURL)   │
                       └──────────────────┬─────────────────────┘
                                          │ HTTP / JSON Multipart
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Microservice (Backend)                         │
│                                                                                 │
│  [Latency Middleware: X-Process-Time-Ms] ──> [CORS & Input Validation (Pydantic)]│
│                                                                                 │
│  API Routes (/api/v1):                                                          │
│  ├── POST /detect              (Single image detection & SLA breakdown)         │
│  ├── POST /detect/batch        (Multi-image batch processing)                   │
│  ├── POST /annotated-image     (JPEG stream with class bounding boxes)          │
│  ├── POST /heatmap             (Gaussian confidence density matrix)             │
│  ├── POST /distances           (Normalized centroid Euclidean proximity)        │
│  ├── POST /scene-description   (Gemini multimodal scene narrative)              │
│  ├── POST /risk-assessment     (Structured safety risk & mitigation points)     │
│  ├── POST /qa                  (Grounded visual question answering)             │
│  ├── POST /compare             (Two-frame object delta tracking)                │
│  ├── POST /export/json         (Standardized JSON schema audit export)          │
│  └── POST /export/pdf          (ReportLab executive PDF generation)             │
│                                                                                 │
│  Service Layer:                                                                 │
│  ├── YOLOService               (Ultralytics YOLOv8n singleton inference)        │
│  ├── CVService                 (OpenCV matrix heatmaps, box rendering, math)    │
│  ├── GeminiService             (Google GenAI SDK multimodal vision requests)    │
│  └── ReportService             (ReportLab dynamic table & document compiler)    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Engineering Highlights

1. **Decoupled Client-Server Separation:** The frontend has **zero machine learning dependencies** (no PyTorch, no OpenCV, no YOLO). It acts as a pure consumer communicating with the FastAPI REST backend via HTTP.
2. **SLA & Latency Profiling:** Every inference call provides high-resolution microsecond latency breakdowns: `preprocess_ms`, `inference_ms`, `postprocess_ms`, `total_ms`, and compute device identification (`cpu` / `cuda`).
3. **Target Class Filtering:** Allows clients to pass a `filter_classes` parameter (e.g. `person,car`), eliminating unwanted detections and saving downstream bandwidth.
4. **Batch Processing Pipeline:** Dedicated `POST /api/v1/detect/batch` endpoint to ingest multiple image files in a single request and return aggregated statistics.
5. **Object Density Heatmap:** Accumulates bounding box confidence matrices, applies OpenCV Gaussian blur (`kernel=(51,51)`), normalizes intensities, and blends with the original image using `cv2.COLORMAP_JET` and `cv2.addWeighted`.
6. **Spatial Proximity Geometry:** Computes normalized 2D centroids $cx = \frac{x_1 + x_2}{2W}$, $cy = \frac{y_1 + y_2}{2H}$ and evaluates Euclidean distances between all detected object pairs, categorizing them into `Very Close`, `Close`, or `Far`.
7. **Dual-Format Interoperability:** Generates executive **PDF reports** via ReportLab for human decision-makers, and exports standardized **JSON schemas** for automated ingestion by enterprise data pipelines.

---

## 📡 REST API Reference

All endpoints are prefixed with `/api/v1`. Interactive OpenAPI/Swagger documentation is available at `http://localhost:8000/docs`.

| Method | Endpoint | Description | Request Format | Response Format |
|---|---|---|---|---|
| `GET` | `/health` | Container liveness probe | None | `{"status": "healthy"}` |
| `GET` | `/api/v1/classes` | List of 80 detectable COCO classes | None | `["person", "bicycle", ...]` |
| `POST` | `/api/v1/detect` | Single image object detection | `multipart/form-data` | `DetectionResponse` (JSON) |
| `POST` | `/api/v1/detect/batch` | Batch multi-image processing | `multipart/form-data` | `BatchDetectionResponse` (JSON) |
| `POST` | `/api/v1/annotated-image` | Rendered bounding box overlay | `multipart/form-data` | `image/jpeg` stream |
| `POST` | `/api/v1/heatmap` | Object density heatmap overlay | `multipart/form-data` | `image/jpeg` stream |
| `POST` | `/api/v1/distances` | Pairwise spatial distance estimation | `multipart/form-data` | `DistanceResponse` (JSON) |
| `POST` | `/api/v1/scene-description` | Gemini visual scene narrative | `multipart/form-data` | `SceneDescriptionResponse` (JSON) |
| `POST` | `/api/v1/risk-assessment` | Structured safety hazard assessment | `multipart/form-data` | `RiskAssessmentResponse` (JSON) |
| `POST` | `/api/v1/qa` | Grounded visual question answering | `multipart/form-data` | `QAResponse` (JSON) |
| `POST` | `/api/v1/compare` | Two-image object delta comparison | `multipart/form-data` | `ComparisonResponse` (JSON) |
| `POST` | `/api/v1/export/json` | Comprehensive JSON audit data | `multipart/form-data` | `FullAnalysisExport` (JSON) |
| `POST` | `/api/v1/export/pdf` | Programmatic executive PDF report | `multipart/form-data` | `application/pdf` download |

---

## 🚀 Local Quickstart

### Prerequisites
- Python 3.10+
- A Google Gemini API Key ([Get one free on Google AI Studio](https://aistudio.google.com))

### 1. Configure Environment
In `D:\Image Analysis`:
Your existing `.env` file already contains your `GOOGLE_API_KEY`.

### 2. Run Backend (FastAPI)
```bash
cd "D:\Image Analysis"
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* ReDoc Specification: `http://localhost:8000/redoc`

### 3. Run Frontend (Streamlit)
In a new terminal:
```bash
cd "D:\Image Analysis"
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```
* Web UI: `http://localhost:8501`

---

## ☁️ Cloud Deployment Guide (Railway / Render)

### Option 1: Deploy to Railway (Recommended)
1. Commit and push these changes to GitHub:
   ```bash
   git add .
   git commit -m "feat: Decouple into FastAPI backend and lightweight Streamlit frontend"
   git push origin main
   ```
2. Log in to [Railway.app](https://railway.app) and create a **New Project** -> **Deploy from GitHub repo**.
3. **Deploy Backend Service:**
   - In service settings, set **Dockerfile Path** to `backend/Dockerfile`.
   - In **Variables**, add:
     - `GOOGLE_API_KEY`: `<your_gemini_api_key>`
     - `GEMINI_MODEL`: `gemini-2.5-flash`
     - `PORT`: `8000`
   - In **Settings** -> **Networking**, click **Generate Domain** (e.g. `https://vision-backend-production.up.railway.app`).
4. **Deploy Frontend Service:**
   - In the same project, click **+ New** -> **GitHub Repo** -> choose `VisionMind-AI`.
   - Set **Dockerfile Path** to `frontend/Dockerfile`.
   - In **Variables**, add:
     - `BACKEND_URL`: `https://vision-backend-production.up.railway.app`
   - Generate a public domain. Done!

### Option 2: Deploy to Render via Blueprint (`render.yaml`)
1. Log in to [Render.com](https://render.com).
2. Go to **Blueprints** -> **New Blueprint Instance**.
3. Connect your `VisionMind-AI` repository. Render will automatically configure both services using `render.yaml`.
4. Enter your `GOOGLE_API_KEY` when prompted.

---

## 🛠️ Tech Stack & Trade-offs

| Component | Technology | Rationale & Trade-off |
|---|---|---|
| **API Framework** | FastAPI | High asynchronous throughput, native OpenAPI/Swagger generation, strict Pydantic v2 data validation. |
| **Object Detector** | Ultralytics YOLOv8 Nano | Chosen for sub-200ms CPU inference and minimal memory footprint (~3M parameters), enabling free-tier cloud hosting without costly GPU instances. |
| **Spatial Engine** | OpenCV & NumPy | Deterministic matrix operations for confidence heatmaps and normalized Euclidean centroid geometry. |
| **Multimodal LLM** | Google Gemini 2.5 Flash | High visual reasoning capability with minimal token latency for natural language descriptions and risk assessments. |
| **Document Compiler**| ReportLab | Direct programmatic PDF compilation with corporate table styling, removing dependencies on browser headless printing. |

---

## 👤 Author
**Dhruv Panwar**  
B.Tech CSE, Vellore Institute of Technology (VIT)  
[GitHub](https://github.com/Dhruv-Panwar042) · [LinkedIn](https://linkedin.com/in/dhruv-panwar) · [LeetCode](https://leetcode.com/u/DhruvPanwar)