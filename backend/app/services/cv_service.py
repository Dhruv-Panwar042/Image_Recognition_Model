import io
from typing import List
import cv2
import numpy as np
from PIL import Image

from app.schemas.analysis import BoundingBox, DistanceItem, DistanceResponse


class CVService:
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
        """Converts PIL Image to binary bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    @staticmethod
    def bytes_to_image(image_bytes: bytes) -> Image.Image:
        """Converts binary bytes to RGB PIL Image."""
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def draw_bounding_boxes(image: Image.Image, detections: List[BoundingBox]) -> Image.Image:
        """
        Renders bounding boxes and labels onto the image with distinct class colors.
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Deterministic color generation per class
        np.random.seed(42)
        color_cache = {}

        for det in detections:
            cid = det.class_id
            if cid not in color_cache:
                color_cache[cid] = tuple(int(c) for c in np.random.randint(60, 220, 3))
            color = color_cache[cid]

            x1, y1, x2, y2 = det.box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img_cv, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(th + 8, y1)), color, -1)
            cv2.putText(
                img_cv,
                label,
                (x1 + 3, max(th + 4, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    @staticmethod
    def generate_heatmap(image: Image.Image, detections: List[BoundingBox]) -> Image.Image:
        """
        Accumulates detection confidence matrices and applies Gaussian blur with JET colormap.
        """
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        for det in detections:
            x1, y1, x2, y2 = det.box
            heatmap[y1:y2, x1:x2] += det.confidence

        # Gaussian smoothing
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(img_np, 0.5, heatmap_rgb, 0.5, 0)
        return Image.fromarray(blended)

    @staticmethod
    def estimate_distances(detections: List[BoundingBox], max_pairs: int = 10) -> DistanceResponse:
        """
        Computes pairwise normalized Euclidean distance between bounding box centroids.
        """
        distances: List[DistanceItem] = []
        n = len(detections)

        for i in range(n):
            for j in range(i + 1, n):
                d1 = detections[i]
                d2 = detections[j]

                cx1, cy1 = d1.normalized_centroid
                cx2, cy2 = d2.normalized_centroid

                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                if dist < 0.3:
                    proximity = "Very Close"
                elif dist < 0.5:
                    proximity = "Close"
                else:
                    proximity = "Far"

                distances.append(
                    DistanceItem(
                        obj1=d1.class_name.capitalize(),
                        obj2=d2.class_name.capitalize(),
                        distance=round(float(dist), 4),
                        proximity=proximity,
                    )
                )

        distances.sort(key=lambda x: x.distance)

        return DistanceResponse(
            total_pairs=len(distances),
            distances=distances[:max_pairs],
        )


cv_service = CVService()
