#!/usr/bin/env python3
"""
Görev 3 — Referans Eşleme OTURUM ORKESTRATÖRÜ.

Tüm boru hattını (banka + segmenter + embedder + eşleştirici) tek bir nesnede toplar.
Hem offline test (`run_offline.py`) hem de yarışma orkestratörü (`client/main_loop.py`)
bu sınıfı çağırır — Görev 3 için TEK giriş noktası.

Akış (PLAN §5.2):
  Oturum başı  : referans crop'ları → HQ-SAM kes → ViT-L (renkli+gri) embedding → ref_bank.npz
  Oturum modalitesi: rgb  → kare segmenter CropFormer, renkli embedding, eşik 0.60
                     termal→ kare segmenter SAM2,       gri   embedding, eşik 0.40
  Her kare     : aktif referans(lar) belirle → segmentle → aday embedding → cosine
                 → eşik üstü en iyi aday = tespit → UndefinedObject

Referans ister RGB ister termal olsun HQ-SAM ile kesilir (cross-modal). Oturum modalitesi
hangi bankanın (renkli/gri) ve hangi segmenter'ın kullanılacağını belirler.

Kullanım:
    # oturum başı bir kez (ayrı süreç önerilir; VRAM):
    ReferenceSession.build_bank("offline_data/referance/..._Referans_Nesneler")

    sess = ReferenceSession("rgb").load_bank()          # modeller yüklenir
    sess.set_schedule([{"object_id": 4, "start": 100, "end": 300}])  # opsiyonel
    objs = sess.process(frame_bgr, frame_idx=137)        # -> [UndefinedObject, ...]
"""
import os
import numpy as np

from src.task3_reference import paths as P
from src.task3_reference.matcher import ReferenceMatcher
from src.task3_reference.ref_bank import build as _build_bank
from src.common.schema import UndefinedObject


class ReferenceSession:
    def __init__(self, modality: str, bank_path: str | None = None,
                 maxside: int | None = P.MAXSIDE, load_models: bool = True,
                 use_tracker: bool = True):
        """
        modality : "rgb" | "termal" (oturum boyunca sabit).
        bank_path: ref_bank.npz yolu (None → paths.REF_BANK).
        maxside  : kareyi segmentasyondan önce küçültme sınırı (8GB VRAM). None → küçültme yok.
        load_models: DINOv3 + kare segmenter'ı hemen yükle (test/dry-run için False).
        """
        assert modality in ("rgb", "termal"), f"geçersiz modalite: {modality}"
        self.modality = modality
        self.maxside = maxside
        self.bank_path = bank_path or P.REF_BANK
        self.matcher = ReferenceMatcher(modality, load_models=load_models)
        self.schedule: list[dict] = []   # [{object_id, start, end}, ...]
        self._bank_loaded = False
        # SAM2 takip: sunucuya ref DEGIL SAM2 kutusu gider; ref arka planda cipa besler
        self.use_tracker = use_tracker
        self._tracker = None
        self._active_oid = None

    # ---------------- Oturum başı: referans bankası ----------------
    @staticmethod
    def build_bank(reference_dir: str, out_path: str | None = None, save_crops: bool = True):
        """Referans klasöründen bankayı kur (HQ-SAM + ViT-L). Oturum başında BİR kez.
        VRAM için ayrı süreçte çalıştırmak ideal (HQ-SAM+CropFormer/SAM2 aynı anda OOM riski)."""
        return _build_bank(reference_dir, out_path=out_path, save_crops=save_crops)

    def load_bank(self, bank_path: str | None = None) -> "ReferenceSession":
        path = bank_path or self.bank_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Referans bankası yok: {path}\n"
                f"    Önce kur: ReferenceSession.build_bank(<referans_klasoru>)")
        self.matcher.load_bank(path)
        self.bank_path = path
        self._bank_loaded = True
        return self

    @property
    def ref_ids(self) -> list[int]:
        return list(self.matcher.ids) if self.matcher.ids else []

    @property
    def ref_names(self) -> list[str]:
        return list(self.matcher.names) if self.matcher.names else []

    # ---------------- Aktif referans planı (kare aralığı) ----------------
    def set_schedule(self, schedule: list[dict] | None):
        """Her referansın arandığı kare aralığı (yarışma: aralıklar çakışmaz, aralık başına
        tek referans). Biçim: [{"object_id": 4, "start": 100, "end": 300}, ...].
        None/boş → tüm referanslar her karede aktif (offline hızlı test)."""
        self.schedule = list(schedule) if schedule else []

    def active_ids(self, frame_idx: int | None) -> list[int] | None:
        """frame_idx için aktif referans id'leri. Plan yoksa None (=hepsi)."""
        if not self.schedule or frame_idx is None:
            return None
        return [s["object_id"] for s in self.schedule
                if int(s.get("start", 0)) <= frame_idx <= int(s.get("end", 10**9))]

    # ---------------- Kare işleme ----------------
    def process(self, frame_bgr: np.ndarray, frame_idx: int | None = None,
                active_ids="auto") -> list[UndefinedObject]:
        """Bir kareyi işle → eşleşen referanslar (UndefinedObject listesi).
        active_ids="auto" → plandan çöz; None → tüm referanslar; liste → o id'ler.
        Boş liste ([]) → bu karede aktif referans yok, arama yapılmaz (boş döner)."""
        if not self._bank_loaded:
            raise RuntimeError("Banka yüklenmedi: önce .load_bank() çağır.")
        if active_ids == "auto":
            active_ids = self.active_ids(frame_idx)
        self.matcher.set_active(active_ids)
        objs = self.matcher.match(frame_bgr, maxside=self.maxside)
        if not self.use_tracker:
            return objs                      # sade eslesme (offline hizli test)

        # --- SAM2 STREAMING takip: tek aktif referans, sunucuya SAM2 kutusu ---
        aid = (active_ids[0] if active_ids else (objs[0].object_id if objs else None))
        if aid is None:
            return []
        if self._tracker is None:
            from src.task3_reference.ref_tracker import RefTracker
            self._tracker = RefTracker(P.SAM2_CFG, P.SAM2_CKPT)
        if aid != self._active_oid:          # aktif referans degisti -> yeni takip
            self._tracker.start(aid); self._active_oid = aid
        det_box = next((( o.top_left_x, o.top_left_y, o.bottom_right_x, o.bottom_right_y)
                        for o in objs if o.object_id == aid), None)
        sam = self._tracker.step(frame_bgr, det_box)   # ref arka planda cipa; SAM2 kutusu doner
        if sam is None:
            return []
        x1, y1, x2, y2 = sam
        return [UndefinedObject(object_id=aid, top_left_x=float(x1), top_left_y=float(y1),
                                bottom_right_x=float(x2), bottom_right_y=float(y2))]


def load_schedule(path: str | None) -> list[dict]:
    """Opsiyonel plan JSON'unu oku: [{"object_id","start","end"}, ...]."""
    if not path:
        return []
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):          # {"schedule": [...]} sarmalını da kabul et
        data = data.get("schedule", [])
    return list(data)
