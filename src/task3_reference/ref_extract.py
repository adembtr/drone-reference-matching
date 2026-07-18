#!/usr/bin/env python3
"""
Görev 3 — Referans crop'tan nesne çıkarma (arka plan silme).
HQ-SAM kutu+nokta prompt ile referans görseldeki ana nesneyi maskeler.
Çıktı: maske (bool), maskeli kesim (arka plan siyah), overlay.

Referans RGB de termal de olabilir (aynı model, cross-modal). Modalite frame adından gelir;
referans çıkarma her ikisinde de aynı HQ-SAM ile yapılır.

Kullanım (run from the repo root so the `src` package resolves):
  python -m src.task3_reference.ref_extract img1.jpg img2.jpg --out offline_data/ref_extract_out
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import numpy as np
import cv2
import torch

from src.task3_reference import paths as P

# Ağırlık DRONE_CODE/models/shared altında (paths.py). Eski sabit yol kaldırıldı.
HQSAM_CKPT = P.HQSAM_CKPT
HQSAM_TYPE = P.HQSAM_TYPE


class RefExtractor:
    def __init__(self, ckpt: str = HQSAM_CKPT, model_type: str = HQSAM_TYPE):
        from segment_anything_hq import sam_model_registry, SamPredictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=ckpt).to(dev).eval()
        self.pred = SamPredictor(sam)

    @torch.inference_mode()
    def best_mask(self, img_bgr: np.ndarray):
        """Kareyi dolduran ana obje maskesi (bool, HxW) + skor + alan."""
        H, W = img_bgr.shape[:2]
        self.pred.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mx, my = int(0.04 * W), int(0.04 * H)
        box = np.array([mx, my, W - mx, H - my])
        cx, cy = W // 2, H // 2
        dx, dy = int(0.18 * W), int(0.18 * H)
        pts = np.array([[cx, cy], [cx-dx, cy], [cx+dx, cy], [cx, cy-dy], [cx, cy+dy]])
        lbl = np.ones(len(pts), int)
        masks, scores, _ = self.pred.predict(
            point_coords=pts, point_labels=lbl, box=box[None, :],
            multimask_output=True, hq_token_only=False)
        best, bs = None, -1e9
        for m, sc in zip(masks, scores):
            m = m.astype(bool); a = m.mean()
            if a < 0.05 or a > 0.985:
                continue
            ys, xs = np.where(m)
            d = np.hypot(xs.mean() - cx, ys.mean() - cy) / np.hypot(W, H)
            val = float(sc) + 0.3 * a - 1.0 * d
            if val > bs:
                bs, best = val, (m, float(sc), a)
        if best is None:
            j = int(np.argmax(scores)); m = masks[j].astype(bool)
            best = (m, float(scores[j]), m.mean())
        return best   # (mask_bool, score, area)

    def cutout(self, img_bgr: np.ndarray):
        """Maskeli kesim: arka plan siyah. Dönüş: (cut_bgr, mask, score, area)."""
        mask, score, area = self.best_mask(img_bgr)
        cut = img_bgr.copy()
        cut[~mask] = 0
        return cut, mask, score, area


def overlay(img_bgr, mask, color=(0, 0, 255), alpha=0.45):
    ov = img_bgr.copy()
    ov[mask] = (alpha * np.array(color) + (1 - alpha) * ov[mask]).astype(np.uint8)
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="offline_data/ref_extract_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ext = RefExtractor()
    for p in args.images:
        img = cv2.imread(p)
        if img is None:
            print(f"[HATA] okunamadı: {p}"); continue
        cut, mask, score, area = ext.cutout(img)
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(args.out, f"{name}_cut.png"), cut)
        cv2.imwrite(os.path.join(args.out, f"{name}_overlay.jpg"), overlay(img, mask))
        # şeffaf PNG (alpha = maske)
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[..., 3] = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, f"{name}_rgba.png"), bgra)
        print(f"[OK] {name}  score={score:.3f}  alan={area*100:.1f}%  -> {args.out}")


if __name__ == "__main__":
    main()
