#!/usr/bin/env python3
"""
Referans bankası kurucu (oturum başında BİR kez çalışır).
referance/ klasöründeki her referans görselini:
  HQ-SAM ile kes → ViT-L renkli + gri MASKELİ embedding → ref_bank.npz
Modaliteden bağımsız: hem renkli hem gri saklanır (kare modalitesi hangisini kullanacağını seçer).

object_id: dosya adından çıkarılır (Referans_Nesne_04 → 4), yoksa sıra numarası.
"""
import os
import re
import glob
import numpy as np
import cv2
from src.task3_reference import paths as P
from src.task3_reference.segmenters import HQSAMRef, mask_bbox
from src.task3_reference.embedder import load_dino, embed_crop, to_gray3


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(p for p in glob.glob(folder + "/*")
                  if os.path.isfile(p) and cv2.imread(p) is not None)


def _obj_id(name, idx):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else idx + 1


def _load_resized(path, maxside):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = maxside / max(h, w) if max(h, w) > maxside else 1.0
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img


def build(reference_dir, out_path=None, save_crops=True):
    out_path = out_path or P.REF_BANK
    paths = _list_images(reference_dir)
    if not paths:
        print(f"[!] Referans yok: {reference_dir}"); return None
    print(f"[+] {len(paths)} referans | HQ-SAM + ViT-L yükleniyor...")
    hq = HQSAMRef()
    dino = load_dino()

    names, ids, embs_c, embs_g = [], [], [], []
    gray_crops, gray_masks = [], []   # termal frame-adaptif ton için gri ref crop + maske
    S = P.EMB_SIZE
    crop_dir = os.path.join(os.path.dirname(out_path), "ref_crops")
    if save_crops:
        os.makedirs(crop_dir, exist_ok=True)
    for i, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        img = _load_resized(path, P.MAXSIDE_SAM)
        mask, sc, area = hq.best_mask(img)
        bb = mask_bbox(mask)
        x1, y1, x2, y2 = bb
        crop = img[y1:y2, x1:x2]
        cm = mask[y1:y2, x1:x2]
        names.append(name)
        ids.append(_obj_id(name, i))
        embs_c.append(embed_crop(dino, crop, mask_bool=cm))
        embs_g.append(embed_crop(dino, to_gray3(crop), mask_bool=cm))
        # gri crop'u (SxS) + maskesini sakla → termalde her karenin tonuna göre yeniden tonlanır
        gc = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_crops.append(cv2.resize(gc, (S, S), interpolation=cv2.INTER_AREA))
        gray_masks.append(
            cv2.resize(cm.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST).astype(bool))
        if save_crops:
            cut = img.copy(); cut[~mask] = 0
            cv2.imwrite(os.path.join(crop_dir, f"{name}_cut.png"), cut[y1:y2, x1:x2])
        print(f"  {name} (id={ids[-1]}): HQ={sc:.3f} kaplama %{area*100:.1f}")

    bank = {
        "names": np.array(names),
        "ids": np.array(ids, dtype=np.int64),
        "color": np.stack(embs_c),   # [R, D]  renkli embedding (RGB oturum + termal 'direkt')
        "gray": np.stack(embs_g),    # [R, D]  düz gri embedding
        "gray_crop": np.stack(gray_crops).astype(np.uint8),  # [R, S, S] termal-ton için ham gri
        "gray_mask": np.stack(gray_masks),                   # [R, S, S] bool
    }
    np.savez(out_path, **bank)
    print(f"[+] {len(names)} referans (renkli+gri) → {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else P.REF_BANK
    build(ref_dir)
