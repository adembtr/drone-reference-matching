#!/usr/bin/env python3
"""
Görev 3 segmenter'ları:
  - HQSAMRef   : referans crop → kutu-prompt tek nesne maskesi (RGB+termal ORTAK)
  - CropFormerSeg : RGB kare → nesne başına tek bütün maske
  - SAM2Seg    : termal kare → şekil-temelli otomatik maske
Kare segmenter'i modaliteye göre seçilir; referans segmenter'i (HQ-SAM) ortak.
"""
import os
import sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import cv2
import torch
from src.task3_reference import paths as P

DEV = P.DEVICE if torch.cuda.is_available() else "cpu"


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------------- HQ-SAM: referans kesme (ORTAK) ----------------
class HQSAMRef:
    def __init__(self):
        from segment_anything_hq import sam_model_registry, SamPredictor
        sam = sam_model_registry[P.HQSAM_TYPE](checkpoint=P.HQSAM_CKPT).to(DEV).eval()
        self.pred = SamPredictor(sam)

    @torch.inference_mode()
    def best_mask(self, img_bgr):
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
        return best


# ---------------- CropFormer: RGB kare ----------------
class CropFormerSeg:
    def __init__(self):
        sys.path.insert(0, P.CROPFORMER_REPO)
        sys.path.insert(0, os.path.join(P.CROPFORMER_REPO, "demo_cropformer"))
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config
        from predictor import CropFormerPredictor
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(P.CROPFORMER_CFG)
        cfg.merge_from_list([
            "MODEL.WEIGHTS", P.CROPFORMER_CKPT,
            "ENTITY.CROP_STRIDE_RATIO", 1.0,
            "ENTITY.CROP_SAMPLE_NUM_TEST", 1,
            "INPUT.MIN_SIZE_TEST", P.CF_MIN_SIZE_TEST,
            "INPUT.MAX_SIZE_TEST", P.CF_MAX_SIZE_TEST,
        ])
        cfg.freeze()
        self.pred = CropFormerPredictor(cfg)

    @torch.inference_mode()
    def masks(self, img_bgr):
        out = self.pred(img_bgr)["instances"]
        scores = out.scores.cpu().numpy()
        pm = out.pred_masks.cpu().numpy()
        keep = scores >= P.CF_SCORE_THRESH
        return [m.astype(bool) for m in pm[keep]]


# ---------------- SAM2: termal kare ----------------
class SAM2Seg:
    def __init__(self):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam2 = build_sam2(P.SAM2_CFG, P.SAM2_CKPT, device=DEV, apply_postprocessing=False)
        self.gen = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=P.SAM2_POINTS_PER_SIDE,
            pred_iou_thresh=P.SAM2_PRED_IOU,
            stability_score_thresh=P.SAM2_STABILITY,
            min_mask_region_area=P.SAM2_MIN_AREA,
        )

    @torch.inference_mode()
    def masks(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            res = self.gen.generate(rgb)
        return [m["segmentation"].astype(bool) for m in res]


def make_frame_segmenter(modality: str):
    """Kare segmenter'ı modaliteye göre kur."""
    return CropFormerSeg() if modality == "rgb" else SAM2Seg()
