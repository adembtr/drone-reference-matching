#!/usr/bin/env python3
"""SAM2 STREAMING sarmalayici — encode-once + buyuyen memory bank, WINDOW YOK.

Offline video_predictor'i NEDENSEL besler: her kare BIR KEZ encode edilir (images'a eklenir),
sadece o kareye propagate edilir; gecmis kareler memory bank'ten gelir (yeniden encode YOK).
Referans araligi basinda start_box (cipa), her kare track(), aralik bitince reset (memory sil).
Boylece sona dogru ~N encode birikir ve SAM daha iyi takip eder (O(n), O(n^2) DEGIL)."""
import os, tempfile, cv2, numpy as np, torch

_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Sam2Stream:
    def __init__(self, predictor):
        self.p = predictor
        self.S = predictor.image_size
        self.st = None
        self.n = 0
        self._tmp = tempfile.mkdtemp(prefix="sam2stream_")

    def _prep(self, bgr):
        """BGR kare -> SAM2 girdi tensoru [1,3,S,S] (resize+normalize), images ile ayni cihaz/tip."""
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.S, self.S)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - _MEAN) / _STD
        return t.unsqueeze(0)

    @torch.inference_mode()
    def start_box(self, frame_bgr, box_xyxy):
        """Yeni cipa: memory'yi sil, bu kareyi 0. kare yap, box ile baslat. Doner: bu karenin bbox'u."""
        self.reset()
        cv2.imwrite(os.path.join(self._tmp, "00000.jpg"), frame_bgr)
        # offload_video_to_cpu: images CPU'da tutulur (buyuyen memory GPU'yu sismesin)
        self.st = self.p.init_state(self._tmp, offload_video_to_cpu=True)
        _, _, masks = self.p.add_new_points_or_box(
            self.st, frame_idx=0, obj_id=1, box=np.asarray(box_xyxy, np.float32))
        self.n = 1
        return _bbox(masks)

    @torch.inference_mode()
    def track(self, frame_bgr):
        """Yeni kare ekle (BIR KEZ encode) ve sadece bu kareye propagate et. Doner: bbox ya da None."""
        if self.st is None:
            return None
        img = self._prep(frame_bgr).to(self.st["images"].device, self.st["images"].dtype)
        self.st["images"] = torch.cat([self.st["images"], img], dim=0)
        self.st["num_frames"] += 1
        idx = self.n
        self.n += 1
        out = None
        for f, ids, masks in self.p.propagate_in_video(
                self.st, start_frame_idx=idx, max_frame_num_to_track=1):
            out = _bbox(masks)
        return out

    def reset(self):
        """Memory bank + encode'lari sil (aralik bitti / yeni referans / olum)."""
        if self.st is not None:
            self.p.reset_state(self.st)
            self.st = None
        self.n = 0

    def close(self):
        import shutil
        self.reset(); shutil.rmtree(self._tmp, ignore_errors=True)


def _bbox(video_res_masks):
    """SAM2 cikti maskesi (1,1,H,W logits) -> (x1,y1,x2,y2) ya da None."""
    m = (video_res_masks[0, 0] > 0.0).cpu().numpy()
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
