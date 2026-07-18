#!/usr/bin/env python3
"""
Görev 3 — UÇTAN UCA OFFLINE TEST ÇALIŞTIRICISI.

Referans bankasını kurar (yoksa/--rebuild), bir kare klasörünü tek modalitede işler,
her kare için eşleşen referansları çizer (kutu + id + cosine) ve results.json'a yazar.
Yarışma orkestratörü olmadan Görev 3'ü baştan sona denemek için.

Kullanım (run from the repo root so the `src` package resolves):

  # RGB kareler (CropFormer):
  python -m src.task3_reference.run_offline \
      --refs   offline_data/referance/THYZ_2026_Ornek_Veri_1_Referans_Nesneler \
      --frames offline_data/referance/frame \
      --modality rgb --glob 'v1_*.jpg' --out logs/g3_rgb

  # Termal kareler (SAM2):
  python -m src.task3_reference.run_offline \
      --refs   offline_data/referance/THYZ_2026_Ornek_Veri_1_Referans_Nesneler \
      --frames offline_data/referance/frame \
      --modality termal --glob 'v2t*.jpg' --out logs/g3_termal

  # opsiyonel plan (kare aralığı → aktif referans):
      --schedule offline_data/g3_schedule.json
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import glob
import json
import argparse

import numpy as np
import cv2

from src.task3_reference import paths as P
from src.task3_reference.session import ReferenceSession, load_schedule

# eşleşen her nesneyi ayırt etmek için sabit renk paleti (BGR)
_COLORS = [(0, 0, 255), (0, 200, 0), (255, 100, 0), (0, 200, 255),
           (255, 0, 200), (0, 128, 255), (128, 0, 255), (0, 255, 128)]


def _list_frames(folder: str, pattern: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    return [p for p in paths if os.path.isfile(p)]


def _draw(frame_bgr, objs) -> np.ndarray:
    vis = frame_bgr.copy()
    for o in objs:
        col = _COLORS[o.object_id % len(_COLORS)]
        p1 = (int(o.top_left_x), int(o.top_left_y))
        p2 = (int(o.bottom_right_x), int(o.bottom_right_y))
        cv2.rectangle(vis, p1, p2, col, 3)
        label = f"id={o.object_id}"
        cv2.putText(vis, label, (p1[0], max(p1[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description="Görev 3 offline uçtan uca test")
    ap.add_argument("--refs", required=True, help="referans nesne görselleri klasörü")
    ap.add_argument("--frames", required=True, help="kare (frame) klasörü")
    ap.add_argument("--modality", required=True, choices=["rgb", "termal"],
                    help="oturum modalitesi (kareler bu modalitede kabul edilir)")
    ap.add_argument("--glob", default="*", help="kare dosya deseni (örn 'v1_*.jpg')")
    ap.add_argument("--bank", default=P.REF_BANK, help="ref_bank.npz yolu")
    ap.add_argument("--rebuild", action="store_true", help="banka varsa bile yeniden kur")
    ap.add_argument("--schedule", default=None, help="opsiyonel plan JSON (kare aralığı→ref)")
    ap.add_argument("--maxside", type=int, default=P.MAXSIDE, help="kare küçültme sınırı (VRAM)")
    ap.add_argument("--out", default="logs/g3_offline", help="çıktı klasörü")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- 1) Referans bankası (oturum başı bir kez) ---
    if args.rebuild or not os.path.exists(args.bank):
        print(f"[1] Referans bankası kuruluyor: {args.refs}")
        if ReferenceSession.build_bank(args.refs, out_path=args.bank) is None:
            print("[!] Banka kurulamadı (referans bulunamadı).")
            return 1
        # HQ-SAM + ViT-L bellekten düşsün → matcher'ın segmenter'ı için yer aç
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[1] Mevcut banka kullanılıyor: {args.bank}  (yeniden kurmak için --rebuild)")

    # --- 2) Oturum: modaliteye göre segmenter + embedder + banka ---
    print(f"[2] Oturum başlatılıyor (modalite={args.modality})...")
    sess = ReferenceSession(args.modality, bank_path=args.bank, maxside=args.maxside).load_bank()
    sess.set_schedule(load_schedule(args.schedule))
    print(f"    {len(sess.ref_ids)} referans: ids={sess.ref_ids}")

    # --- 3) Kareleri işle ---
    frames = _list_frames(args.frames, args.glob)
    if not frames:
        print(f"[!] Kare yok: {args.frames} / {args.glob}")
        return 1
    print(f"[3] {len(frames)} kare işleniyor ({args.modality})...")

    results = []
    n_hit = 0
    for idx, path in enumerate(frames):
        frame = cv2.imread(path)
        if frame is None:
            print(f"  [{idx}] okunamadı: {path}")
            continue
        objs = sess.process(frame, frame_idx=idx)
        scores = sess.matcher.last_scores            # {id: cosine}
        variants = getattr(sess.matcher, "last_variants", {})  # {id: 'color'|'gray'|'thermal'}
        vis = _draw(frame, objs)
        fn = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out, f"{fn}_match.jpg"), vis)
        # results.json'a skoru + kazanan varyantı DEBUG alanı olarak ekle (sunucuya gitmez)
        det = []
        for o in objs:
            d = o.to_json()
            d["score"] = round(float(scores.get(o.object_id, 0.0)), 4)
            d["variant"] = variants.get(o.object_id, "")
            det.append(d)
        results.append({
            "frame": os.path.basename(path),
            "frame_idx": idx,
            "threshold": sess.matcher.thresh,
            "detected_undefined_objects": det,
        })
        if objs:
            n_hit += 1
            hits = ", ".join(
                f"id={o.object_id}({scores.get(o.object_id,0):.3f},{variants.get(o.object_id,'')})"
                for o in objs)
            print(f"  [{idx}] {fn}: EŞLEŞTİ (eşik {sess.matcher.thresh}) → {hits}")
        else:
            print(f"  [{idx}] {fn}: eşleşme yok")

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Bitti: {n_hit}/{len(frames)} karede eşleşme.")
    print(f"[+] Görseller + results.json → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
