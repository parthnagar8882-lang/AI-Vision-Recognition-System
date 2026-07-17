"""Beginner-friendly OpenCV preprocessing and Tesseract OCR helpers."""
import os
import uuid
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output


class OCRProcessingError(Exception):
    """An expected OCR problem that can be shown safely in the interface."""


def _configure_tesseract():
    """Allow an optional TESSERACT_CMD environment variable on Windows."""
    command = os.getenv("TESSERACT_CMD")
    if command:
        pytesseract.pytesseract.tesseract_cmd = command


def process_ocr(image_path, processed_dir):
    _configure_tesseract()
    image = cv2.imread(str(image_path))
    if image is None:
        raise OCRProcessingError("OpenCV could not load the uploaded image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu automatically chooses a useful cut-off value for many printed-text images.
    _value, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    processed_name = f"ocr_{uuid.uuid4().hex}.png"
    output_path = Path(processed_dir) / processed_name
    if not cv2.imwrite(str(output_path), thresholded):
        raise OCRProcessingError("Could not save the preprocessed image.")

    try:
        config = "--psm 6"
        text = pytesseract.image_to_string(thresholded, config=config).strip()
        data = pytesseract.image_to_data(thresholded, config=config, output_type=Output.DICT)
    except pytesseract.TesseractNotFoundError:
        raise OCRProcessingError(
            "Tesseract OCR was not found. Install it and add it to PATH, or set TESSERACT_CMD."
        )
    except pytesseract.TesseractError as error:
        raise OCRProcessingError(f"Tesseract could not process this image: {error}")

    confidences = []
    for confidence in data.get("conf", []):
        try:
            value = float(confidence)
            if value >= 0:
                confidences.append(value)
        except (TypeError, ValueError):
            continue

    return {
        "text": text,
        "confidence": round(sum(confidences) / len(confidences), 1) if confidences else None,
        "preprocessed_image": f"processed/{processed_name}",
        "no_text": not bool(text),
    }
