# AI Vision Recognition System

Artificial Intelligence Internship Project 4: a simple Flask application with two computer-vision features: OCR text recognition and MobileNet-SSD object detection. It uses existing pre-trained tools; it does not train a model from scratch.

## Features

- Upload JPG, JPEG, PNG, or BMP images (up to 10 MB).
- OCR: grayscale conversion, Gaussian blur, Otsu thresholding, Tesseract text extraction, and average word-level OCR confidence.
- Object detection: OpenCV DNN, MobileNet-SSD, configurable 50–90% threshold (80% default), bounding boxes, labels, and model confidence values.
- Secure unique upload names, image-content validation, responsive interface, and friendly errors.

## How it works

### OCR module

The image is loaded with OpenCV, converted to grayscale, lightly denoised with Gaussian blur, and binarized with Otsu thresholding. Tesseract then reads the preprocessed image using `--psm 6`. The displayed **Average OCR Confidence** is an average of valid Tesseract word confidence values; it is not a scientific accuracy measurement.

### Object detection module

OpenCV creates a 300×300 image blob and sends it to the pre-trained Caffe MobileNet-SSD network. Predictions at or above the selected confidence threshold are drawn onto a saved output image. Supported labels are the 20 PASCAL VOC classes included in the model, including person, bicycle, car, cat, dog, chair, bus, train, and tv monitor.

Higher thresholds can reduce false positives, but may also hide valid objects with lower confidence.

## Project structure

```
ai-vision-recognition/
├── app.py                 # Flask routes and secure file uploads
├── ocr_engine.py          # OpenCV preprocessing and Tesseract OCR
├── object_detector.py     # MobileNet-SSD inference
├── models/                # Downloaded model files go here
├── static/css/style.css
├── static/js/script.js
├── static/uploads/        # temporary originals (ignored by Git)
├── static/processed/      # generated results (ignored by Git)
└── templates/index.html
```

## Installation on Windows

1. Open PowerShell in this project folder.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install Python packages:

   ```powershell
   pip install -r requirements.txt
   ```

4. Install the **Tesseract OCR engine** separately. Use a trusted Windows installer, ensure its install folder is in your `PATH`, then verify:

   ```powershell
   tesseract --version
   ```

   If it is not on PATH, set it for the current PowerShell session before launching Flask (adjust the path to your own installation):

   ```powershell
   $env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

5. Download the two legitimate pre-trained MobileNet-SSD Caffe files and put them in `models/` with these exact names:

   - `MobileNetSSD_deploy.prototxt`
   - `MobileNetSSD_deploy.caffemodel`

   The model is deliberately not supplied or fabricated in this project. OCR will still work before the model files are installed; detection will show a clear message.

6. Start the site:

   ```powershell
   python app.py
   ```

7. Open `http://127.0.0.1:5000` in a browser.

## Testing

- **OCR 1:** upload a clear English text screenshot. Extracted text and a preprocessed image should appear.
- **OCR 2:** upload a document photograph with readable text.
- **OCR 3:** upload an image with no readable text. A friendly no-text message should appear.
- **Detection 1:** upload a clear person image. A `person` box appears if confidence meets the threshold.
- **Detection 2:** upload a car image. A `car` box appears if confidence meets the threshold.
- **Detection 3:** upload multiple supported objects; all predictions over the threshold are listed.
- **Invalid file:** upload a PDF or TXT file. The application should safely reject it.

## Troubleshooting

- **`TesseractNotFoundError`:** install Tesseract, run `tesseract --version`, then add it to PATH or set `TESSERACT_CMD` as shown above.
- **Missing MobileNet-SSD files:** confirm both files are in `models/` and their names match exactly. Detection is unavailable until then; OCR remains available.
- **OpenCV cannot load image:** use a real JPG, JPEG, PNG, or BMP image that is not corrupted.
- **Invalid upload format:** PDFs and text files are intentionally unsupported.
- **No OCR text:** try a sharper, better-lit image with larger printed English text.
- **No object above 80%:** select a lower threshold (such as 70% or 60%), or use a clearer image of a supported class.

## Limitations and future improvements

OCR is best for printed English text and may struggle with handwriting, blur, or rotated pages. MobileNet-SSD supports only its listed classes and can miss small or unusual objects. Future versions could add image rotation, language selection, model selection, and automatic cleanup of old uploads.
