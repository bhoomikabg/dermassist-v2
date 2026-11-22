# src/inference/dermassist_model.py
# Shared inference helpers: model loading, preprocessing, TTA logits, temperature + melanoma-bias calibration.
# Used by predict_single_image.py and verify_pipeline.py.

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torchvision import models

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMG = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_model(checkpoint_path, device):
    """ResNet18 whose fc is the trained head, so model(x) == head(backbone(x)) and model.layer4 exists for Grad-CAM."""
    ck = torch.load(checkpoint_path, map_location="cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(CLASSES)))
    # backbone was saved with fc=Identity (no fc.* keys); head keys are 1.weight / 1.bias -> fc.1.weight / fc.1.bias
    model.load_state_dict({**ck["backbone"], **{"fc." + k: v for k, v in ck["head"].items()}}, strict=True)
    return model.to(device).eval()


def load_calibration(path):
    """temperature_calibration.pt -> (T, melanoma logit bias beta)."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        T = float(obj.get("temperature", obj.get("T", obj.get("temp"))))
        beta = float(obj.get("mel_bias", 0.0))
    else:
        T, beta = float(obj), 0.0
    assert T > 0, f"temperature must be positive, got {T}"
    return T, beta


def to_tensor(pil_img, size=IMG):
    """PIL RGB -> normalised [1,3,H,W]. Same resize as training: PIL .resize((size,size))."""
    a = np.asarray(pil_img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    x = torch.from_numpy(a).permute(2, 0, 1)
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def preprocess(image_path, size=IMG):
    return to_tensor(Image.open(image_path), size)


@torch.no_grad()
def tta_logits(model, x):
    """Mean logits over 4 flips (identity, horizontal, vertical, both): same TTA as ENS_logits.npz."""
    views = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])]
    return torch.stack([model(v).float() for v in views]).mean(0)


def calibrate(logits, T, beta, mel_index=CLASSES.index("mel")):
    """softmax((z + beta * e_mel) / T)  -- temperature scaling with the melanoma logit bias."""
    z = logits.clone()
    z[..., mel_index] += beta
    return torch.softmax(z / T, dim=-1)
