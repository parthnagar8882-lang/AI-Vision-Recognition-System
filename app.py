"""Flask application for the AI Vision Recognition System."""
import os
import uuid
from pathlib import Path

import cv2
from flask import Flask, abort, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ocr_engine import OCRProcessingError, process_ocr
from object_detector import DetectionError, detect_objects

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
PROCESSED_DIR = BASE_DIR / "static" / "processed"
TEXT_DIR = PROCESSED_DIR / "text"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
DEFAULT_THRESHOLD = 0.80

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

for directory in (UPLOAD_DIR, PROCESSED_DIR, TEXT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_valid_image(uploaded_file):
    """Validate an uploaded image and save it under a collision-safe name."""
    if not uploaded_file or not uploaded_file.filename:
        raise ValueError("Please choose an image file.")
    if not allowed_file(uploaded_file.filename):
        raise ValueError("Unsupported file type. Please upload JPG, JPEG, PNG, or BMP.")

    original_name = secure_filename(uploaded_file.filename)
    if not original_name:
        raise ValueError("The uploaded filename is not valid.")
    extension = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"
    destination = UPLOAD_DIR / unique_name
    uploaded_file.save(destination)

    # A valid extension alone is not enough; ensure OpenCV can read the file.
    image = cv2.imread(str(destination))
    if image is None:
        destination.unlink(missing_ok=True)
        raise ValueError("This file could not be read as a valid image.")
    return destination, unique_name


def page(**context):
    context.setdefault("default_threshold", int(DEFAULT_THRESHOLD * 100))
    return render_template("index.html", **context)


@app.route("/")
def index():
    return page()


@app.post("/ocr")
def ocr():
    try:
        image_path, image_name = save_valid_image(request.files.get("ocr_image"))
        result = process_ocr(image_path, PROCESSED_DIR)
        text_filename = f"{Path(image_name).stem}_{uuid.uuid4().hex[:8]}.txt"
        (TEXT_DIR / text_filename).write_text(result["text"], encoding="utf-8")
        return page(
            active_mode="ocr", ocr_result=result, original_image=f"uploads/{image_name}",
            text_filename=text_filename,
        )
    except (ValueError, OCRProcessingError) as error:
        return page(active_mode="ocr", error=str(error))


@app.post("/detect")
def detect():
    try:
        image_path, image_name = save_valid_image(request.files.get("detect_image"))
        try:
            threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
        except ValueError:
            raise ValueError("Please select a valid confidence threshold.")
        if threshold not in {0.5, 0.6, 0.7, 0.8, 0.9}:
            raise ValueError("Please select one of the available confidence thresholds.")
        result = detect_objects(image_path, PROCESSED_DIR, threshold)
        return page(
            active_mode="detect", detect_result=result, original_image=f"uploads/{image_name}",
            selected_threshold=int(threshold * 100),
        )
    except (ValueError, DetectionError) as error:
        return page(active_mode="detect", error=str(error), selected_threshold=request.form.get("threshold", "0.8"))


@app.get("/download-text/<filename>")
def download_text(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename or not safe_name.endswith(".txt") or not (TEXT_DIR / safe_name).is_file():
        abort(404)
    return send_from_directory(TEXT_DIR, safe_name, as_attachment=True, download_name="extracted_text.txt")


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    return page(error="Image is too large. The maximum upload size is 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
