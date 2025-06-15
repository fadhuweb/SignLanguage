import os
import io
import json
import base64
import numpy as np
import cv2
import torch
import tempfile
import requests
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from model import SignLanguageTranslator

app = Flask(__name__)
CORS(app)
app.static_folder = 'static'

# === Model Auto-Download ===
MODEL_URL = "https://huggingface.co/datasets/fadhuweb/best_slt_model/resolve/main/best_slt_model.pt"
MODEL_PATH = "/tmp/best_slt_model.pt"  # Use /tmp for compatibility on Render and other platforms

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Downloading model from Hugging Face...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # Basic content validation
        with open(MODEL_PATH, "rb") as f:
            head = f.read(10)
            if head.startswith(b"<!DOCTYPE") or b"<html" in head:
                raise RuntimeError("Downloaded file is HTML (not a model). Check Hugging Face permissions or URL.")
        print("✅ Model downloaded to", MODEL_PATH)

download_model()

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# === Load model ===
try:
    translator = SignLanguageTranslator(MODEL_PATH, device=DEVICE)
    print(f"✅ Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    translator = None

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": translator is not None})

@app.route('/api/model-info', methods=['GET'])
def model_info():
    if translator is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        return jsonify(translator.get_model_info())
    except Exception as e:
        return jsonify({"error": f"Failed to get model info: {str(e)}"}), 500

@app.route('/api/translate', methods=['POST'])
def translate_sign_language():
    if translator is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        translation = translator.translate_image(image)
        confidence = 0.8
        return jsonify({"translation": translation, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

@app.route('/api/translate-sequence', methods=['POST'])
def translate_sequence():
    if translator is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        if not data or 'frames' not in data or not data['frames']:
            return jsonify({"error": "No frames provided"}), 400

        frames = []
        for b64 in data['frames']:
            image_data = base64.b64decode(b64)
            image = Image.open(io.BytesIO(image_data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frames.append(image)

        translation = translator.translate_sequence(frames)
        confidence = 0.8
        return jsonify({"translation": translation, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": f"Sequence translation failed: {str(e)}"}), 500

@app.route('/api/translate-live-frame', methods=['POST'])
def translate_live_frame():
    if translator is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame provided"}), 400

        image_data = base64.b64decode(data['frame'])
        image = Image.open(io.BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        translation = translator.translate_image(image)
        confidence = 0.8
        return jsonify({"translation": translation, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": f"Live frame processing failed: {str(e)}"}), 500

@app.route('/api/translate-live-buffer', methods=['POST'])
def translate_live_buffer():
    if translator is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        if not data or 'frames' not in data or not data['frames']:
            return jsonify({"error": "No frames provided"}), 400

        frames = []
        for b64 in data['frames']:
            image_data = base64.b64decode(b64)
            image = Image.open(io.BytesIO(image_data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frames.append(image)

        translation = translator.translate_sequence(frames)
        confidence = 0.8
        return jsonify({"translation": translation, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": f"Live buffer processing failed: {str(e)}"}), 500

@app.route('/api/upload-info', methods=['GET'])
def upload_info():
    return jsonify({
        "allowed_image_types": list(ALLOWED_IMAGE_EXTENSIONS),
        "allowed_video_types": list(ALLOWED_VIDEO_EXTENSIONS),
        "max_file_size_bytes": MAX_CONTENT_LENGTH,
        "max_file_size_mb": MAX_CONTENT_LENGTH / (1024 * 1024)
    })

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_frames_from_video(video_path, max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Error opening video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30

    frame_interval = max(1, total_frames // max_frames)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0 and len(frames) < max_frames:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if translator is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(temp_dir, filename)

        try:
            file.save(file_path)

            if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
                image = cv2.imread(file_path)
                if image is None:
                    return jsonify({"error": "Could not read image file"}), 400

                translation = translator.translate_image(image)
                confidence = 0.8
                return jsonify({
                    "translation": translation,
                    "confidence": confidence,
                    "file_type": "image"
                })

            elif allowed_file(filename, ALLOWED_VIDEO_EXTENSIONS):
                max_frames = request.form.get('max_frames', default=30, type=int)
                frames = extract_frames_from_video(file_path, max_frames)
                if not frames:
                    return jsonify({"error": "Could not extract frames"}), 400

                translation = translator.translate_sequence(frames)
                confidence = 0.8
                return jsonify({
                    "translation": translation,
                    "confidence": confidence,
                    "file_type": "video",
                    "frames_processed": len(frames)
                })
            else:
                return jsonify({
                    "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS)}"
                }), 400
        finally:
            try:
                os.remove(file_path)
                os.rmdir(temp_dir)
            except:
                pass
    except Exception as e:
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
