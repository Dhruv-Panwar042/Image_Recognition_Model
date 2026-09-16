import time
from collections import Counter
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
from ultralytics import YOLO

from app.core.config import settings
from app.schemas.analysis import (
    BoundingBox,
    PerformanceMetrics,
    DetectionResponse,
    ClassDelta,
    ComparisonResponse,
)


class YOLOService:
    _instance: Optional["YOLOService"] = None
    _model: Optional[YOLO] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Loads the YOLOv8 model weights."""
        print(f"Loading YOLO model from: {settings.YOLO_MODEL_PATH}...")
        self._model = YOLO(settings.YOLO_MODEL_PATH)
        self.device = "cuda" if self._model.device.type == "cuda" else "cpu"
        self.names = self._model.names

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._load_model()
        return self._model

    def get_class_names(self) -> List[str]:
        """Returns the list of 80 COCO classes supported."""
        return list(self.names.values())

    def run_detection(
        self,
        image: Image.Image,
        confidence_threshold: float = settings.DEFAULT_CONFIDENCE,
        filter_classes: Optional[List[str]] = None,
    ) -> Tuple[DetectionResponse, any]:
        """
        Runs YOLOv8 object detection on a PIL Image.
        Returns the structured DetectionResponse schema and raw Ultralytics result.
        """
        start_time = time.perf_counter()
        
        # Run inference with optimized dimension for sub-second CPU latency
        results = self.model(image, imgsz=settings.YOLO_IMG_SIZE, verbose=False)
        raw_result = results[0]
        
        # Prepare filter list if specified
        allowed_classes = None
        if filter_classes:
            allowed_classes = set(c.lower().strip() for c in filter_classes)

        width, height = image.size
        detections: List[BoundingBox] = []
        confidences: List[float] = []
        classes_found: List[str] = []

        # Parse boxes
        for box in raw_result.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            class_id = int(box.cls[0])
            class_name = self.names[class_id]

            if allowed_classes and class_name.lower() not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = round((x1 + x2) / 2 / width, 4)
            cy = round((y1 + y2) / 2 / height, 4)

            detections.append(
                BoundingBox(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=round(conf, 4),
                    box=[x1, y1, x2, y2],
                    normalized_centroid=[cx, cy],
                )
            )
            confidences.append(conf)
            classes_found.append(class_name)

        # Performance timing breakdown
        speed = getattr(raw_result, "speed", {})
        preprocess_ms = round(speed.get("preprocess", 0.0), 2)
        inference_ms = round(speed.get("inference", 0.0), 2)
        postprocess_ms = round(speed.get("postprocess", 0.0), 2)
        total_ms = round((time.perf_counter() - start_time) * 1000, 2)

        metrics = PerformanceMetrics(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
            device=self.device,
        )

        class_counts = dict(Counter(classes_found))
        avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0

        response = DetectionResponse(
            total_objects=len(detections),
            unique_classes=len(class_counts),
            average_confidence=avg_conf,
            image_width=width,
            image_height=height,
            class_counts=class_counts,
            detections=detections,
            metrics=metrics,
        )

        return response, raw_result

    def compare_images(
        self,
        image1: Image.Image,
        image2: Image.Image,
        confidence_threshold: float = settings.DEFAULT_CONFIDENCE,
    ) -> ComparisonResponse:
        """
        Runs detection on two images and outputs class delta comparison.
        """
        start_time = time.perf_counter()
        resp1, _ = self.run_detection(image1, confidence_threshold)
        resp2, _ = self.run_detection(image2, confidence_threshold)

        all_classes = set(resp1.class_counts.keys()) | set(resp2.class_counts.keys())
        changes: List[ClassDelta] = []

        for cls in all_classes:
            c1 = resp1.class_counts.get(cls, 0)
            c2 = resp2.class_counts.get(cls, 0)
            changes.append(
                ClassDelta(
                    object=cls,
                    image1_count=c1,
                    image2_count=c2,
                    change=c2 - c1,
                )
            )

        changes.sort(key=lambda x: abs(x.change), reverse=True)
        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ComparisonResponse(
            image1_total=resp1.total_objects,
            image2_total=resp2.total_objects,
            changes=changes,
            metrics=PerformanceMetrics(
                preprocess_ms=round(resp1.metrics.preprocess_ms + resp2.metrics.preprocess_ms, 2),
                inference_ms=round(resp1.metrics.inference_ms + resp2.metrics.inference_ms, 2),
                postprocess_ms=round(resp1.metrics.postprocess_ms + resp2.metrics.postprocess_ms, 2),
                total_ms=total_time_ms,
                device=self.device,
            ),
        )


yolo_service = YOLOService()
