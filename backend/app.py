from __future__ import annotations

import io
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dermassist_model import CLASSES, load_model, load_calibration, to_tensor, tta_logits, calibrate
from decision_layer import decide_from_probabilities
from gradcam_utils import GradCAM, get_module_by_name

MAX_BYTES = 12 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
CHECKPOINT = Path(os.getenv("DERMASSIST_CHECKPOINT", ROOT / "models/P2b_seed0_best.pt"))
CALIBRATION = Path(os.getenv("DERMASSIST_CALIBRATION", ROOT / "models/temperature_calibration.pt"))
OUT = ROOT / "outputs" / "explainability"
OUT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="DermAssist V2 API", version="2.0.0")
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins], allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")
_model = None
_calibration = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resources():
    global _model, _calibration
    if _model is None:
        if not CHECKPOINT.is_file() or not CALIBRATION.is_file():
            raise RuntimeError("Model checkpoint or its matching calibration artifact is missing.")
        _model = load_model(str(CHECKPOINT), _device)
        _calibration = load_calibration(str(CALIBRATION))
    return _model, _calibration


def explain(model, image: Image.Image, x: torch.Tensor, cls: int, uid: str):
    camgen = GradCAM(model, get_module_by_name(model, "layer4"))
    heat = camgen.generate(x, cls)
    heat_img = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255)).resize(image.size, Image.Resampling.BILINEAR)
    arr = np.asarray(heat_img, dtype=np.float32) / 255
    jet = np.stack([np.clip(1.5 - np.abs(4*arr-3), 0, 1), np.clip(1.5 - np.abs(4*arr-2), 0, 1), np.clip(1.5 - np.abs(4*arr-1), 0, 1)], axis=-1)
    heat_rgb = Image.fromarray(np.uint8(jet * 255))
    overlay = Image.blend(image.convert("RGB"), heat_rgb, 0.42)
    names = {"original": f"{uid}_original.jpg", "heatmap": f"{uid}_heatmap.png", "overlay": f"{uid}_overlay.jpg"}
    image.convert("RGB").save(OUT / names["original"], quality=90)
    heat_rgb.save(OUT / names["heatmap"])
    overlay.save(OUT / names["overlay"], quality=90)
    return {key: f"/outputs/{name}" for key, name in names.items()}


@app.get("/api/health")
def health():
    try:
        model, (temp, bias) = resources()
        return {"status": "ok", "model_loaded": model is not None, "calibration_loaded": True, "temperature": temp, "mel_bias": bias, "classes": CLASSES, "device": str(_device)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED:
        raise HTTPException(415, "Upload a JPEG, PNG or WebP image.")
    raw = await image.read(MAX_BYTES + 1)
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image must be 12 MB or smaller.")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "The selected file is not a readable image.") from exc
    try:
        model, (temp, bias) = resources()
        x = to_tensor(pil).to(_device)
        with torch.no_grad():
            probs = calibrate(tta_logits(model, x), temp, bias)[0].cpu().tolist()
        probmap = {c: float(probs[i]) for i, c in enumerate(CLASSES)}
        order = sorted(range(len(CLASSES)), key=lambda i: probs[i], reverse=True)
        uid = uuid.uuid4().hex
        visual = explain(model, pil, x, order[0], uid)
        decision = decide_from_probabilities(probmap).to_dict()
        return {"class_names": CLASSES, "probabilities": probmap, "model_top1": {"class_name": CLASSES[order[0]], "probability": float(probs[order[0]])}, "decision": decision, "calibration": {"temperature": temp, "mel_bias": bias, "method": "4-flip TTA; softmax((logits + mel_bias × melanoma_indicator) / temperature)"}, "explainability": visual, "disclaimer": "Research output only; not a diagnosis."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Prediction could not be completed: {exc}") from exc
