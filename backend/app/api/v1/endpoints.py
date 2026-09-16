import io
import time
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from PIL import Image

from app.core.config import settings
from app.schemas.analysis import (
    DetectionResponse,
    DistanceResponse,
    SceneDescriptionResponse,
    RiskAssessmentResponse,
    QAResponse,
    ComparisonResponse,
    BatchDetectionItem,
    BatchDetectionResponse,
    FullAnalysisExport,
)
from app.services.yolo_service import yolo_service
from app.services.cv_service import cv_service
from app.services.gemini_service import gemini_service
from app.services.report_service import report_service

router = APIRouter()


def _read_image(upload_file: UploadFile) -> Image.Image:
    """Safely decodes an uploaded file into a PIL RGB Image."""
    try:
        contents = upload_file.file.read()
        return cv_service.bytes_to_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file '{upload_file.filename}': {str(e)}")


def _parse_filter_classes(filter_classes_raw: Optional[str]) -> Optional[List[str]]:
    """Splits comma-separated or list-style class names."""
    if not filter_classes_raw:
        return None
    return [c.strip() for c in filter_classes_raw.split(",") if c.strip()]


@router.get("/classes", response_model=List[str], tags=["Detection"])
def get_supported_classes():
    """Returns the list of 80 detectable COCO classes supported by YOLOv8."""
    return yolo_service.get_class_names()


@router.post("/detect", response_model=DetectionResponse, tags=["Detection"])
def detect_objects(
    file: UploadFile = File(..., description="Image to analyze"),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
    filter_classes: Optional[str] = Form(None, description="Comma-separated class names to filter (e.g. 'person,car')"),
):
    """
    Runs YOLOv8 object detection on an uploaded image with performance latency breakdown.
    """
    image = _read_image(file)
    allowed = _parse_filter_classes(filter_classes)
    detection, _ = yolo_service.run_detection(image, confidence_threshold, allowed)
    return detection


@router.post("/detect/batch", response_model=BatchDetectionResponse, tags=["Detection"])
def detect_objects_batch(
    files: List[UploadFile] = File(..., description="Multiple images to process sequentially"),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
):
    """
    Batch detection endpoint supporting multiple image uploads.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    start_total = time.perf_counter()
    batch_results: List[BatchDetectionItem] = []

    for file in files:
        try:
            image = _read_image(file)
            det, _ = yolo_service.run_detection(image, confidence_threshold)
            batch_results.append(
                BatchDetectionItem(
                    filename=file.filename or "unknown",
                    total_objects=det.total_objects,
                    unique_classes=det.unique_classes,
                    class_counts=det.class_counts,
                    average_confidence=det.average_confidence,
                    metrics=det.metrics,
                )
            )
        except Exception:
            continue

    total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)
    return BatchDetectionResponse(
        total_images_processed=len(batch_results),
        results=batch_results,
        total_processing_time_ms=total_time_ms,
    )


@router.post("/annotated-image", tags=["Detection"])
def get_annotated_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
    filter_classes: Optional[str] = Form(None),
):
    """
    Returns the image with bounding boxes, labels, and confidence tags rendered.
    """
    image = _read_image(file)
    allowed = _parse_filter_classes(filter_classes)
    detection, _ = yolo_service.run_detection(image, confidence_threshold, allowed)
    annotated = cv_service.draw_bounding_boxes(image, detection.detections)
    img_bytes = cv_service.image_to_bytes(annotated, format="JPEG")
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


@router.post("/heatmap", tags=["Spatial Analysis"])
def get_object_density_heatmap(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
):
    """
    Generates a Gaussian-smoothed object density heatmap overlaid on the original image.
    """
    image = _read_image(file)
    detection, _ = yolo_service.run_detection(image, confidence_threshold)
    heatmap_img = cv_service.generate_heatmap(image, detection.detections)
    img_bytes = cv_service.image_to_bytes(heatmap_img, format="JPEG")
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


@router.post("/distances", response_model=DistanceResponse, tags=["Spatial Analysis"])
def calculate_spatial_distances(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
    max_pairs: int = Query(10, ge=1, le=50),
):
    """
    Calculates normalized Euclidean distances between bounding box centroids and outputs proximity ratings.
    """
    image = _read_image(file)
    detection, _ = yolo_service.run_detection(image, confidence_threshold)
    return cv_service.estimate_distances(detection.detections, max_pairs=max_pairs)


@router.post("/scene-description", response_model=SceneDescriptionResponse, tags=["Multimodal Intelligence"])
def generate_scene_description(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
):
    """
    Combines YOLO structured detection results with Gemini multimodal vision for natural language scene understanding.
    """
    image = _read_image(file)
    detection, _ = yolo_service.run_detection(image, confidence_threshold)
    img_bytes = cv_service.image_to_bytes(image)

    objects_summary = ", ".join([f"{box.class_name} ({box.confidence*100:.1f}%)" for box in detection.detections])
    if not objects_summary:
        objects_summary = "No major objects detected above threshold."

    return gemini_service.get_scene_description(img_bytes, objects_summary)


@router.post("/risk-assessment", response_model=RiskAssessmentResponse, tags=["Multimodal Intelligence"])
def generate_risk_assessment(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
):
    """
    Evaluates safety hazards, risk level (Low/Medium/High), and mitigation recommendations via Gemini.
    """
    image = _read_image(file)
    detection, _ = yolo_service.run_detection(image, confidence_threshold)
    img_bytes = cv_service.image_to_bytes(image)

    objects_summary = ", ".join([f"{box.class_name} ({box.confidence*100:.1f}%)" for box in detection.detections])
    if not objects_summary:
        objects_summary = "No objects detected."

    return gemini_service.get_risk_assessment(img_bytes, objects_summary)


@router.post("/qa", response_model=QAResponse, tags=["Multimodal Intelligence"])
def visual_question_answering(
    file: UploadFile = File(...),
    question: str = Form(..., min_length=2, description="Question about the image"),
):
    """
    Answers arbitrary questions grounded in the visual evidence of the uploaded image.
    """
    image = _read_image(file)
    img_bytes = cv_service.image_to_bytes(image)
    return gemini_service.ask_question(img_bytes, question)


@router.post("/compare", response_model=ComparisonResponse, tags=["Detection"])
def compare_two_images(
    file1: UploadFile = File(..., description="First image baseline"),
    file2: UploadFile = File(..., description="Second image comparison"),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
):
    """
    Compares object distributions between two images and returns frequency deltas.
    """
    img1 = _read_image(file1)
    img2 = _read_image(file2)
    return yolo_service.compare_images(img1, img2, confidence_threshold)


@router.post("/export/json", response_model=FullAnalysisExport, tags=["Export"])
def export_analysis_json(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
    include_scene_desc: bool = Form(False),
    include_risk_assess: bool = Form(False),
):
    """
    Exports a comprehensive structured JSON audit schema for downstream microservices and data lakes.
    """
    image = _read_image(file)
    detection, _ = yolo_service.run_detection(image, confidence_threshold)
    distances = cv_service.estimate_distances(detection.detections).distances

    scene_desc = None
    risk_assess = None

    if include_scene_desc or include_risk_assess:
        img_bytes = cv_service.image_to_bytes(image)
        objects_summary = ", ".join([f"{box.class_name} ({box.confidence*100:.1f}%)" for box in detection.detections])

        if include_scene_desc:
            scene_desc = gemini_service.get_scene_description(img_bytes, objects_summary).description
        if include_risk_assess:
            risk_assess = gemini_service.get_risk_assessment(img_bytes, objects_summary)

    return FullAnalysisExport(
        system_version=settings.VERSION,
        detection_summary=detection,
        distances=distances,
        scene_description=scene_desc,
        risk_assessment=risk_assess,
    )


@router.post("/export/pdf", tags=["Export"])
def export_analysis_pdf(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25, ge=0.05, le=0.95),
    include_scene_desc: bool = Form(True),
    include_risk_assess: bool = Form(True),
):
    """
    Generates and downloads an executive PDF report containing detection tables, annotated image, and LLM insights.
    """
    image = _read_image(file)
    detection, _ = yolo_service.run_detection(image, confidence_threshold)
    annotated = cv_service.draw_bounding_boxes(image, detection.detections)
    distances = cv_service.estimate_distances(detection.detections).distances

    scene_desc = "N/A"
    risk_assess = None

    if include_scene_desc or include_risk_assess:
        try:
            img_bytes = cv_service.image_to_bytes(image)
            objects_summary = ", ".join([f"{box.class_name} ({box.confidence*100:.1f}%)" for box in detection.detections])
            if include_scene_desc:
                scene_desc = gemini_service.get_scene_description(img_bytes, objects_summary).description
            if include_risk_assess:
                risk_assess = gemini_service.get_risk_assessment(img_bytes, objects_summary)
        except Exception:
            pass

    pdf_bytes = report_service.generate_pdf(
        image=annotated,
        detection=detection,
        distances=distances,
        scene_desc=scene_desc,
        risk_assess=risk_assess,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=image_analysis_report.pdf"},
    )
