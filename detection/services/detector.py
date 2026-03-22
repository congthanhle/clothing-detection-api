"""
Detector service for clothing detection.
Uses a two-stage pipeline:
1. yolov8n.pt to ensure at least one person is in the image.
2. deepfashion2_yolov8s-seg.pt to detect clothing items, returning bounding boxes.
"""

from ultralytics import YOLO
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# --- CONSTANTS ---

PERSON_MODEL_PATH   = "yolov8n.pt"
CLOTHING_MODEL_PATH = settings.CLOTHING_MODEL_PATH

PERSON_CLASS_ID          = 0      # COCO class 0 = person
PERSON_CONF_THRESHOLD    = 0.50
CLOTHING_CONF_THRESHOLD  = 0.35

DEEPFASHION2_CLASSES = {
    0:  "Short sleeve top",
    1:  "Long sleeve top",
    2:  "Short sleeve outwear",
    3:  "Long sleeve outwear",
    4:  "Vest",
    5:  "Sling",
    6:  "Shorts",
    7:  "Trousers",
    8:  "Skirt",
    9:  "Short sleeve dress",
    10: "Long sleeve dress",
    11: "Vest dress",
    12: "Sling dress",
}

DEEPFASHION2_CLASSES_VI = {
    0:  "Áo tay ngắn",
    1:  "Áo tay dài",
    2:  "Áo khoác tay ngắn",
    3:  "Áo khoác tay dài",
    4:  "Áo vest",
    5:  "Áo hai dây",
    6:  "Quần short",
    7:  "Quần dài",
    8:  "Váy",
    9:  "Đầm tay ngắn",
    10: "Đầm tay dài",
    11: "Đầm vest",
    12: "Đầm hai dây",
}

# --- EXCEPTIONS ---

class NoPersonDetectedError(Exception):
    """Raised when no person is found in the uploaded image."""
    pass

class DetectionError(Exception):
    """Raised when the clothing model inference fails."""
    pass

# --- SINGLETON LOADERS ---

_person_model   = None
_clothing_model = None

def _get_person_model() -> YOLO:
    """Returns a singleton instance of the person guard model."""
    global _person_model
    if _person_model is None:
        logger.info("[Detector] Loading person guard model...")
        _person_model = YOLO(PERSON_MODEL_PATH)
    return _person_model

def _get_clothing_model() -> YOLO:
    """Returns a singleton instance of the DeepFashion2 clothing model."""
    global _clothing_model
    if _clothing_model is None:
        import os
        from huggingface_hub import hf_hub_download

        if not os.path.exists(CLOTHING_MODEL_PATH):
            logger.info("[Detector] Model not found locally, downloading from HuggingFace...")
            model_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="deepfashion2_yolov8s-seg.pt")
        else:
            model_path = CLOTHING_MODEL_PATH

        logger.info("[Detector] Loading DeepFashion2 model...")
        _clothing_model = YOLO(model_path)
    return _clothing_model

# --- MAIN detect() FUNCTION ---

def detect(image_path: str) -> list[dict]:
    """
    Two-stage detection pipeline:

    Stage 1 — Person guard:
      Run yolov8n.pt to confirm at least one person exists.
      Raises NoPersonDetectedError if no person found (conf >= 0.50).

    Stage 2 — Clothing detection:
      Run deepfashion2_yolov8s-seg.pt.
      Extract ONLY bbox from each detection (ignore segmentation masks).
      Filter by confidence >= CLOTHING_CONF_THRESHOLD.
      Map class index to friendly label via DEEPFASHION2_CLASSES.

    Returns:
    [
      {
        "label":      "Long sleeve top",
        "confidence": 0.87,
        "bbox": { "x1": 42, "y1": 80, "x2": 310, "y2": 420 }
      },
      ...
    ]

    Raises:
      NoPersonDetectedError  — no person in image
      DetectionError         — model inference failure
    """

    # STAGE 1 — Person guard
    try:
        person_results = _get_person_model().predict(
            source=image_path,
            verbose=False,
            conf=PERSON_CONF_THRESHOLD
        )
        persons = [
            b for b in person_results[0].boxes
            if int(b.cls) == PERSON_CLASS_ID
        ]
    except Exception as e:
        raise DetectionError(f"Person check failed: {e}")

    if not persons:
        raise NoPersonDetectedError(
            "No person detected in this image. "
            "Please upload a photo of a person wearing clothing."
        )

    logger.info("[Detector] Stage 1 passed — %d person(s) found", len(persons))

    # STAGE 2 — Clothing detection
    # Note: model returns Results with .boxes (bbox) and .masks (segmentation).
    # We access only .boxes and ignore .masks entirely.
    try:
        clothing_results = _get_clothing_model().predict(
            source=image_path,
            verbose=False,
            conf=CLOTHING_CONF_THRESHOLD
        )

        detections = []
        boxes = clothing_results[0].boxes  # access bbox only, masks ignored

        for box in boxes:
            class_id   = int(box.cls)
            confidence = round(float(box.conf), 4)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "label":      DEEPFASHION2_CLASSES_VI.get(class_id, f"Class {class_id}"),
                "confidence": confidence,
                "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

    except Exception as e:
        raise DetectionError(f"Clothing detection failed: {e}")

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    logger.info("[Detector] Stage 2 complete — %d item(s) detected", len(detections))
    return detections

class ClothingDetector:
    """Wrapper class to maintain compatibility with existing view imports."""
    def detect(self, image_path: str) -> list[dict]:
        """Runs the standalone detect function for compatibility."""
        return detect(image_path)

# --- MOCK FUNCTION ---

def mock_detect(image_path: str) -> list[dict]:
    """Hardcoded detections for testing without running models."""
    return [
        {"label": "Áo tay dài",
         "confidence": 0.92,
         "bbox": {"x1": 80,  "y1": 60,  "x2": 320, "y2": 380}},
        {"label": "Quần dài",
         "confidence": 0.87,
         "bbox": {"x1": 100, "y1": 390, "x2": 300, "y2": 580}},
        {"label": "Áo khoác tay ngắn",
         "confidence": 0.74,
         "bbox": {"x1": 60,  "y1": 50,  "x2": 340, "y2": 400}},
    ]
