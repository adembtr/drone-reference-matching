#!/usr/bin/env python3
"""
Görev 3 — Referans Eşleştirici (streaming, kare kare).

Kullanım (orkestratörden):
    m = ReferenceMatcher(modality="rgb")     # oturum modalitesi
    m.load_bank("offline_data/ref_bank.npz") # oturum başı kurulan banka
    m.set_active([4])                        # sunucu: bu kare aralığında sadece id=4 aranıyor
    objs = m.match(frame_bgr)                # -> [UndefinedObject, ...]

Tasarım:
- Segmenter modaliteye göre (rgb→CropFormer, termal→SAM2) — TEK sefer yüklenir.
- Embedding TEK model (ViT-L); termalde gri, RGB'de renkli.
- Sadece AKTİF referanslar aranır (yarışma: aralık başına tek referans, çakışmaz).
- Eşik altı → gönderilmez (false-positive cezası). Aktif ref başına en iyi TEK aday.
"""
import numpy as np
import cv2
from src.task3_reference import paths as P
from src.task3_reference.embedder import load_dino, embed_crop, to_gray3, to_edge3, to_shape3
from src.task3_reference.segmenters import make_frame_segmenter, mask_bbox
from src.common.schema import UndefinedObject


class ReferenceMatcher:
    def __init__(self, modality: str, load_models: bool = True):
        assert modality in ("rgb", "termal")
        self.modality = modality
        self.gray = (modality == "termal")
        self.thresh = P.MATCH_THRESH_TERMAL if self.gray else P.MATCH_THRESH
        self.names = None
        self.ids = None
        self.bank_color = None  # [R, D] renkli embedding
        self.bank_gray = None   # [R, D] düz gri embedding
        self.gray_crops = None  # [R, S, S] termal-ton için ham gri crop
        self.gray_masks = None  # [R, S, S] bool
        self._has_crops = False
        self.active = None      # None = hepsi; yoksa aktif id kümesi
        self.last_scores = {}   # {object_id: cosine} — son match'te eşleşenlerin skoru
        self.last_variants = {} # {object_id: 'color'|'gray'|'thermal'} — kazanan varyant (debug)
        self.dino = None
        self.segmenter = None
        if load_models:
            self.dino = load_dino()
            self.segmenter = make_frame_segmenter(modality)

    def load_bank(self, bank_path=None):
        d = np.load(bank_path or P.REF_BANK, allow_pickle=True)
        self.names = list(d["names"])
        self.ids = list(int(x) for x in d["ids"])
        self.bank_color = d["color"]
        self.bank_gray = d["gray"] if "gray" in d.files else d["color"]
        # termal frame-adaptif ton için ham gri crop'lar (yeni banka formatı)
        self._has_crops = ("gray_crop" in d.files)
        if self._has_crops:
            self.gray_crops = d["gray_crop"]
            self.gray_masks = d["gray_mask"]
        self.bank_edge = None    # şekil varyantları — ilk match'te tembel hesap (dino gerekir)
        self.bank_shape = None
        return self

    def _ensure_shape_banks(self):
        """Referansların kenar (edge) ve siluet (shape) embedding'leri — bir kez."""
        if self.bank_edge is not None or not self._has_crops:
            return
        e, s = [], []
        for gc, mk in zip(self.gray_crops, self.gray_masks):
            g3 = cv2.cvtColor(gc, cv2.COLOR_GRAY2BGR)
            # şekil girdilerinde maske-ağırlıklama YOK: kontur/siluet bütün olarak embed edilir
            e.append(embed_crop(self.dino, to_edge3(g3, mk)))
            s.append(embed_crop(self.dino, to_shape3(mk)))
        self.bank_edge = np.stack(e)
        self.bank_shape = np.stack(s)

    def set_active(self, active_ids):
        """Sunucunun bu kare aralığı için istediği referans id'leri. None=hepsi."""
        self.active = set(active_ids) if active_ids is not None else None

    def _active_indices(self):
        if self.active is None:
            return list(range(len(self.ids)))
        return [i for i, rid in enumerate(self.ids) if rid in self.active]

    @staticmethod
    def _resize(img, maxside):
        h, w = img.shape[:2]
        s = maxside / max(h, w) if max(h, w) > maxside else 1.0
        if s < 1.0:
            return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA), s
        return img, 1.0

    @staticmethod
    def _frame_stats(proc):
        """Kare gri yoğunluğu ort/std (frame ölçeği) — termal-ton hedefi."""
        g = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(g.mean()), float(g.std() + 1e-6)

    def _thermal_ref_embs(self, idxs, fmean, fstd):
        """Aktif referansların gri crop'unu KARENİN tonuna (ort/std) eşleyip embed et.
        Referansı 'termale yakın ton'a çevirip arama (frame-adaptif)."""
        out = []
        for i in idxs:
            gc = self.gray_crops[i].astype(np.float32)
            mk = self.gray_masks[i]
            fg = gc[mk] if mk.any() else gc.reshape(-1)
            m, s = float(fg.mean()), float(fg.std() + 1e-6)
            toned = np.clip((gc - m) / s * fstd + fmean, 0, 255).astype(np.uint8)
            toned3 = cv2.cvtColor(toned, cv2.COLOR_GRAY2BGR)
            out.append(embed_crop(self.dino, toned3, mask_bool=mk))
        return np.stack(out)

    def match(self, frame_bgr, maxside=None) -> list[UndefinedObject]:
        """maxside: 8GB VRAM için kareyi küçült (segmentasyon belleği). bbox tam çözünürlüğe
        geri ölçeklenir. None → küçültme yok.

        RGB oturum: referans DİREKT renkli aranır (tek varyant).
        Termal oturum: her referans için max(renkli-direkt, düz-gri, frame-adaptif-termal-ton)."""
        idxs = self._active_indices()
        if not idxs or self.bank_color is None:
            return []
        proc, scale = (frame_bgr, 1.0) if maxside is None else self._resize(frame_bgr, maxside)
        masks = self.segmenter.masks(proc)
        if not masks:
            return []
        # adaylar: embedding + bbox (işlenen çözünürlükte)
        # termalde aday 3 girdiyle embed edilir: gri (doku) + kenar + siluet (saf şekil)
        inv = 1.0 / scale
        embs, embs_e, embs_s, boxes = [], [], [], []
        for m in masks:
            bb = mask_bbox(m)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            crop = proc[y1:y2, x1:x2]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            cmask = m[y1:y2, x1:x2]
            cin = to_gray3(crop) if self.gray else crop
            embs.append(embed_crop(self.dino, cin, mask_bool=cmask))
            if self.gray and P.USE_SHAPE_VARIANTS:
                embs_e.append(embed_crop(self.dino, to_edge3(crop, cmask)))
                embs_s.append(embed_crop(self.dino, to_shape3(cmask)))
            # bbox'ı tam çözünürlüğe geri ölçekle
            boxes.append((x1*inv, y1*inv, x2*inv, y2*inv))
        if not embs:
            return []
        E = np.stack(embs)                       # [N, D]

        # varyantlar: (aday_embs, ref_bank, isim). RGB: sadece renkli.
        # Termal: renkli + gri (doku) + kenar + siluet (saf şekil, DINOv3 girişinde).
        variants = [(E, self.bank_color[idxs], "color")]
        if self.gray:
            variants.append((E, self.bank_gray[idxs], "gray"))
            if P.USE_SHAPE_VARIANTS and self._has_crops:   # kapalı — elendi
                self._ensure_shape_banks()
                variants.append((np.stack(embs_e), self.bank_edge[idxs], "edge"))
                variants.append((np.stack(embs_s), self.bank_shape[idxs], "shape"))
            if P.USE_THERMAL_TONE and self._has_crops:   # kapalı (paths.USE_THERMAL_TONE)
                fmean, fstd = self._frame_stats(proc)
                variants.append((E, self._thermal_ref_embs(idxs, fmean, fstd), "thermal"))
        # her varyant için [N, A] benzerlik → element bazında MAX (en iyi varyantı seç)
        sims_stack = np.stack([Ev @ Bv.T for Ev, Bv, _ in variants], axis=0)  # [V, N, A]
        sims = sims_stack.max(axis=0)            # [N, A]
        win = sims_stack.argmax(axis=0)          # [N, A] hangi varyant kazandı

        out = []
        self.last_scores = {}
        self.last_variants = {}
        for col, i in enumerate(idxs):
            j = int(np.argmax(sims[:, col]))
            sc = float(sims[j, col])
            if sc >= self.thresh:
                x1, y1, x2, y2 = boxes[j]
                out.append(UndefinedObject(
                    object_id=self.ids[i],
                    top_left_x=float(x1), top_left_y=float(y1),
                    bottom_right_x=float(x2), bottom_right_y=float(y2)))
                self.last_scores[self.ids[i]] = sc
                self.last_variants[self.ids[i]] = variants[int(win[j, col])][2]
        return out
