import os
import io
import base64
import traceback

from flask import Flask, request, jsonify, render_template
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_CKPT    = os.path.join(os.path.dirname(__file__), "model", "resnet50.pth")
# Default: look for model/resnet50.pth next to this script. Can be overridden with MODEL_PATH.
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", DEFAULT_CKPT)
IMG_SIZE        = 224
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_FILE_MB  = 10   # raised from 5 — some images can be large

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

# ── Override ALL Flask error pages to return JSON (fixes the HTML error bug) ──
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": f"Bad request: {e.description}"}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": f"File too large. Maximum allowed size is {MAX_FILE_MB} MB."}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ── Build & Load Model ────────────────────────────────────────────────────────
def load_model(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'.\n"
            f"Place resnet50.pth in a 'model' folder next to app.py (default),\n"
            f"or set the MODEL_PATH environment variable to point to your checkpoint."
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Auto-detect fc output size from the checkpoint itself — never hardcode
    fc_weight   = state_dict["fc.weight"]
    num_classes = fc_weight.shape[0]

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    meta = ckpt.get("config", {})

    print("=" * 60)
    print("[MODEL LOADED]")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Device      : {DEVICE}")
    print(f"  FC output   : {num_classes} classes")
    print(f"  Test AP     : {ckpt.get('test_ap', 'n/a')}")
    print(f"  Test Acc    : {ckpt.get('test_accuracy', 'n/a')}")
    print("=" * 60)

    return model, meta, num_classes

# ── Preprocessing ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Load model once at startup ────────────────────────────────────────────────
model, model_meta, NUM_CLASSES = load_model(CHECKPOINT_PATH)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, img: Image.Image):
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)                              # [1, 1] or [1, 2]

        if NUM_CLASSES == 1:
            prob_fake = torch.sigmoid(logits[0, 0]).item()
            prob_real = 1.0 - prob_fake
        else:
            probs     = torch.softmax(logits, dim=1)[0]
            prob_real = probs[0].item()
            prob_fake = probs[1].item()

    label      = "AI-Generated" if prob_fake >= 0.5 else "Real"
    confidence = max(prob_real, prob_fake)

    return {
        "label"      : label,
        "confidence" : round(confidence * 100, 2),
        "prob_fake"  : round(prob_fake   * 100, 2),
        "prob_real"  : round(prob_real   * 100, 2),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_meta=model_meta)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    try:
        img    = Image.open(file.stream)
        result = predict(model, img)

        thumb  = img.convert("RGB")
        thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        result["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()   # full traceback visible in your terminal
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE),
                    "num_classes": NUM_CLASSES, "model": "ResNet50-Wang2020"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)