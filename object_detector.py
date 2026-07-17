"""MobileNet-SSD object detection using OpenCV's DNN module."""
import uuid
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROTOTXT = BASE_DIR / "models" / "MobileNetSSD_deploy.prototxt"
MODEL = BASE_DIR / "models" / "MobileNetSSD_deploy.caffemodel"
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
]
_network = None


class DetectionError(Exception):
    """An expected detection problem suitable for the web interface."""


def _get_network():
    global _network
    if not PROTOTXT.is_file() or not MODEL.is_file():
        raise DetectionError(
            "MobileNet-SSD model files are missing. Please place the required model files inside the models folder."
        )
    if _network is None:
        try:
            _network = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(MODEL))
        except cv2.error as error:
            raise DetectionError(f"MobileNet-SSD model files could not be loaded: {error}")
    return _network


def detect_objects(image_path, processed_dir, threshold=0.80):
    image = cv2.imread(str(image_path))
    if image is None:
        raise DetectionError("OpenCV could not load the uploaded image.")
    network = _get_network()
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    network.setInput(blob)
    predictions = network.forward()
    output = image.copy()
    detections = []

    for index in range(predictions.shape[2]):
        confidence = float(predictions[0, 0, index, 2])
        if confidence < threshold:
            continue
        class_id = int(predictions[0, 0, index, 1])
        if class_id <= 0 or class_id >= len(CLASSES):
            continue
        x1, y1, x2, y2 = (predictions[0, 0, index, 3:7] * np.array([width, height, width, height])).astype(int)
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        label = CLASSES[class_id]
        text = f"{label}: {confidence * 100:.1f}%"
        cv2.rectangle(output, (x1, y1), (x2, y2), (36, 113, 163), 2)
        label_y = y1 - 10 if y1 > 25 else y1 + 22
        cv2.putText(output, text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36, 113, 163), 2)
        detections.append({"label": label, "confidence": round(confidence * 100, 1)})

    processed_name = f"detection_{uuid.uuid4().hex}.jpg"
    if not cv2.imwrite(str(Path(processed_dir) / processed_name), output):
        raise DetectionError("Could not save the detected output image.")
    return {"processed_image": f"processed/{processed_name}", "detections": detections, "count": len(detections)}
