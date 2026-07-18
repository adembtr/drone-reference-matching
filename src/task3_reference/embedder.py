#!/usr/bin/env python3
"""
DINOv3 ViT-L MASKELİ embedding (Görev 3 kimlik çıkarıcı — TEK model, cross-modal).
RGB kare → renkli embedding, termal kare → gri embedding (referans da gri).
DINOv3 fp16'da NaN üretir → fp32 zorunlu.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from src.task3_reference import paths as P

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 16
NUM_PREFIX = 5   # 1 CLS + 4 register (DINOv3 ViT-L)


def load_dino():
    from transformers import DINOv3ViTModel
    dev = P.DEVICE if torch.cuda.is_available() else "cpu"
    return DINOv3ViTModel.from_pretrained(P.DINOV3_PATH, dtype=torch.float32).to(dev).eval()


def to_gray3(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def to_edge3(img, mask_bool=None):
    """SAF ŞEKİL girdi 1: Canny kenar haritası (3 kanal). Doku/renk/modalite gitmez,
    sadece kontur kalır → termal-RGB cross-modal için."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    lo = max(20, int(np.median(g) * 0.66))
    e = cv2.Canny(g, lo, lo * 2)
    if mask_bool is not None:
        e[~mask_bool] = 0                      # nesne dışı kenarları at
    e = cv2.dilate(e, np.ones((2, 2), np.uint8))
    return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)


def to_shape3(mask_bool):
    """SAF ŞEKİL girdi 2: ikili maske silueti (beyaz nesne/siyah zemin, 3 kanal).
    Görüntü pikseli hiç kullanılmaz → tamamen modalite-bağımsız."""
    m = (mask_bool.astype(np.uint8)) * 255
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def _preprocess(crop_bgr, size):
    size = (size // PATCH) * PATCH
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0), size


@torch.inference_mode()
def embed_crop(model, crop_bgr, mask_bool=None, size=None):
    """Maske içi patch token ortalaması (L2-normalize). mask yoksa whole-crop."""
    size = size or P.EMB_SIZE
    dev = next(model.parameters()).device
    t, size = _preprocess(crop_bgr, size)
    t = t.to(dev, dtype=torch.float32)
    out = model(pixel_values=t).last_hidden_state[0]
    gh = gw = size // PATCH
    patches = out[NUM_PREFIX:, :].float()
    if mask_bool is not None and mask_bool.any():
        m = cv2.resize(mask_bool.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        w = torch.from_numpy((m > 0.4).astype(np.float32)).to(patches.device).reshape(-1)
        if w.sum() < 1:
            w = torch.from_numpy((m > 0.2).astype(np.float32)).to(patches.device).reshape(-1)
        if w.sum() < 1:
            w = torch.ones(gh * gw, device=patches.device)
        emb = (patches * w.unsqueeze(-1)).sum(0) / w.sum()
    else:
        emb = patches.mean(0)
    return F.normalize(emb, dim=0).cpu().numpy()


def cosine(a, b):
    return float(np.dot(a, b))
