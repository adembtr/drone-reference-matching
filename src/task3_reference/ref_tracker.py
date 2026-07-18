#!/usr/bin/env python3
"""G3 STREAMING referans takip — SAM2 kutusunu uretir (referans DOGRUDAN yollanmaz).

Nedensel (kare kare, gecmisle) SAM2 takibi = Sam2Stream (encode-once + buyuyen memory bank,
WINDOW YOK). Referans tespiti SADECE arka planda: SEED_N tespitte cipala; SAM ile uyusmayan AMA
birbiriyle TUTARLI REANCHOR_N tespit birikince cogunluga yeniden cipala; kenardan cikinca (bad_jump)
BIRAK; olumden sonra REACQ_N yeni tespitle yeniden yakala. Aktif referans degisince start() ->
memory sifirlanir (aralik bitti -> encode'lar silinir, sonraki referansta sifirdan).

Kullanim (orkestrator, aktif referans aralik icinde her kare):
    trk = RefTracker(sam2_cfg, sam2_ckpt)
    trk.start(object_id)
    box = trk.step(frame_bgr, det_box|None)   # -> (x1,y1,x2,y2) SAM2 kutusu ya da None
"""
import math, numpy as np, torch, cv2
from collections import deque

SEED_N     = 3    # kac tespitte cipalanir ("uc tespiti veririz")
REACQ_N    = 3    # olumden sonra yeniden yakalama icin gereken yeni tespit
# --- NURON 2026-07-16: HAREKET-TELAFILI MEDYAN reanchor (eski IoU-tabanli REANCHOR_N kaldirildi) ---
# Sorun: kare kayinca (nesne 4-15px/kare ilerliyor) eski kutular yenisiyle IoU catismiyordu ->
# "4 tutarli tespit" HIC tetiklenmiyordu; ayrica hizli nesnede eski medyan-tohum LAG yapiyordu.
# Cozum: ORB homografisiyle son WIN tespiti BUGUNE tasi -> MEDYAN hedef (outlier parca + lag ikisi de gider).
WIN          = 5    # hareket-telafili medyan penceresi (son kac DINOv3 tespiti)
TOL_REANCHOR = 55.0 # SAM merkezi telafili medyandan bu kadar (px) saparsa -> yeniden cipala
JUMP_FR  = 0.18   # merkez sicramasi (kare kosegeni orani) -> olum
AREA_HI, AREA_LO = 3.2, 0.28

_ORB = cv2.ORB_create(1500)
_BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _area(b): return max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1])
def _cen(b):  return ((b[0]+b[2])/2, (b[1]+b[3])/2)
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    it = max(0, ix2-ix1)*max(0, iy2-iy1); u = _area(a)+_area(b)-it
    return it/u if u > 0 else 0.0


def _homography(g0, g1):
    """g0->g1 (onceki->simdiki) 0.5x gri kareler arasi ORB+RANSAC homografisi. Az eslesme -> None."""
    k0, d0 = _ORB.detectAndCompute(g0, None); k1, d1 = _ORB.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12: return None
    m = _BF.match(d0, d1)
    if len(m) < 12: return None
    p0 = np.float32([k0[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    p1 = np.float32([k1[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
    return H


class RefTracker:
    def __init__(self, sam2_cfg, sam2_ckpt, device="cuda"):
        from sam2.build_sam import build_sam2_video_predictor
        from src.task3_reference.sam2_stream import Sam2Stream
        pred = build_sam2_video_predictor(sam2_cfg, sam2_ckpt, device=device)
        self.stream = Sam2Stream(pred)
        self.start(None)

    def start(self, object_id):
        """Yeni aktif referans — durum + SAM2 memory sifirla."""
        self.oid = object_id
        self.box = None          # o anki takip kutusu (SAM2)
        self.dets = []           # cipalama oncesi tespit kutulari
        self.dis_buf = []        # (eski, kullanilmiyor)
        self.state = "idle"      # idle -> active -> dead
        # hareket-telafili medyan tamponlari (HEP simdiki kare koordinatinda)
        self.cbuf = deque(maxlen=WIN)   # tespit merkezleri
        self.sbuf = deque(maxlen=WIN)   # tespit (w,h) boyutlari
        self.prev_gray = None
        self.stream.reset()

    def _target(self):
        """Tampondaki (telafili) tespit merkezlerinin MEDYANI + boyut medyani."""
        if not self.cbuf: return None, None
        c = np.median(np.array(self.cbuf), axis=0)
        wh = np.median(np.array(self.sbuf), axis=0) if self.sbuf else None
        return c, wh

    def _box_from(self, c, wh):
        w, h = wh
        return np.array([c[0]-w/2, c[1]-h/2, c[0]+w/2, c[1]+h/2], np.float32)

    def _near_edge(self, b, W, H, m=4):
        return b[0] <= m or b[1] <= m or b[2] >= W-m or b[3] >= H-m

    def _bad_jump(self, prev, cur, W, H):
        if cur is None or _area(cur) < 50: return True
        if prev is None: return False
        if math.dist(_cen(prev), _cen(cur)) > JUMP_FR*math.hypot(W, H): return True
        ap, ac = _area(prev), _area(cur)
        return ac > AREA_HI*ap or ac < AREA_LO*ap

    def _anchor(self, box, frame):
        """SAM2 memory'yi sifirla, bu kareyi cipa yap. Doner: cipa karesinin SAM2 kutusu."""
        bb = self.stream.start_box(frame, box)
        self.box = bb if bb is not None else tuple(box)
        self.dis_buf = []
        return self.box

    def step(self, frame_bgr, det_box):
        """Bir kare isle. det_box: referans tespiti ya da None. Doner: SAM2 kutusu ya da None.
        HAREKET-TELAFILI MEDYAN: son WIN tespiti ORB homografisiyle BUGUNE tasi -> medyan hedef;
        SEED/REACQ o medyandan tohumlanir (LAG yok), SAM medyandan saparsa yeniden cipalanir."""
        H, W = frame_bgr.shape[:2]
        # 1) onceki tespit merkezlerini BUGUNE tasi (kamera donme/ilerlemesini geri al)
        g = cv2.cvtColor(cv2.resize(frame_bgr, (W // 2, H // 2)), cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None and self.cbuf:
            Hm = _homography(self.prev_gray, g)
            if Hm is not None:
                S = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
                Hf = np.linalg.inv(S) @ Hm @ S          # 0.5x gri uzay -> tam cozunurluk
                a = np.array(list(self.cbuf), np.float32).reshape(-1, 1, 2)
                self.cbuf = deque([tuple(p) for p in cv2.perspectiveTransform(a, Hf).reshape(-1, 2)], maxlen=WIN)
        self.prev_gray = g
        # 2) yeni DINOv3 tespitini tampona ekle (merkez + boyut, simdiki koordinat)
        if det_box is not None:
            self.cbuf.append(tuple(_cen(det_box)))
            self.sbuf.append((det_box[2] - det_box[0], det_box[3] - det_box[1]))
        tc, twh = self._target()

        # 3) SEED / REACQ: telafili medyandan tohumla (guncel konum -> hizli nesnede LAG yok)
        if self.state in ("idle", "dead"):
            need = SEED_N if self.state == "idle" else REACQ_N
            if det_box is not None: self.dets.append(det_box)
            if len(self.dets) >= need and tc is not None and twh is not None:
                self.state = "active"
                return self._anchor(self._box_from(tc, twh), frame_bgr)
            return None

        # 4) active: streaming SAM2; SAM merkezi telafili medyandan saparsa -> YENIDEN cipala
        #    (outlier DINOv3 karesi (agiz/govde) medyanla elenir; hizli-hareket drift'i toparlanir)
        prev = self.box
        cur = self.stream.track(frame_bgr)
        if self._bad_jump(prev, cur, W, H):        # kenardan cikti/zipladi -> BIRAK
            self.state = "dead"; self.box = None; self.dets = []
            self.cbuf.clear(); self.sbuf.clear(); self.stream.reset()
            return None
        self.box = cur
        if tc is not None and twh is not None and math.dist(_cen(cur), tuple(tc)) > TOL_REANCHOR:
            return self._anchor(self._box_from(tc, twh), frame_bgr)
        return self.box

    def close(self):
        self.stream.close()
