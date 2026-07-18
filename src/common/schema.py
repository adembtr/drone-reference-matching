#!/usr/bin/env python3
"""
Yarışma JSON şeması (teknik şartname Şekil 16-17).

Sunucudan GELEN kare (frame request cevabı) alanları:
    url, image_url, video_name, session, translation_x/y/z, gps_health_status
Bize GÖNDERİLECEK sonuç paketi alanları:
    id, user, frame, detected_objects[], detected_translations[], detected_undefined_objects[]
"""
from dataclasses import dataclass, field
from typing import List, Optional
import math


# ----------------------------- GELEN KARE -----------------------------
@dataclass
class FrameInfo:
    url: str
    image_url: str
    video_name: str
    session: str
    translation_x: Optional[float]
    translation_y: Optional[float]
    translation_z: Optional[float]
    gps_health_status: int          # 1 = GT sağlıklı (ilk 450), 0 = GPS yok (NaN)

    @staticmethod
    def from_json(d: dict) -> "FrameInfo":
        def num(v):
            if v is None:
                return None
            if isinstance(v, str) and v.strip().lower() in ("nan", "none", ""):
                return None
            try:
                f = float(v)
                return None if math.isnan(f) else f
            except (TypeError, ValueError):
                return None
        # health alanı iki isimle gelebilir (şartname: gps_health_status / health_status)
        health = d.get("gps_health_status", d.get("health_status", 0))
        try:
            health = int(health)
        except (TypeError, ValueError):
            health = 0
        return FrameInfo(
            url=d.get("url", ""),
            image_url=d.get("image_url", ""),
            video_name=d.get("video_name", ""),
            session=d.get("session", ""),
            translation_x=num(d.get("translation_x")),
            translation_y=num(d.get("translation_y")),
            translation_z=num(d.get("translation_z")),
            gps_health_status=health,
        )

    @property
    def has_gt(self) -> bool:
        return self.gps_health_status == 1 and self.translation_x is not None


# ----------------------------- SONUÇ PAKETİ -----------------------------
@dataclass
class DetectedObject:
    cls: int                    # 0=tasit 1=insan 2=uap 3=uai
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float
    landing_status: int = -1    # 0=uygun değil, 1=uygun, -1=iniş alanı değil
    motion_status: int = -1     # 0=hareketsiz, 1=hareketli, -1=taşıt değil

    def to_json(self) -> dict:
        return {
            "cls": str(self.cls),
            "landing_status": str(self.landing_status),
            "motion_status": str(self.motion_status),
            "top_left_x": self.top_left_x,
            "top_left_y": self.top_left_y,
            "bottom_right_x": self.bottom_right_x,
            "bottom_right_y": self.bottom_right_y,
        }


@dataclass
class DetectedTranslation:
    translation_x: float
    translation_y: float
    translation_z: float

    def to_json(self) -> dict:
        return {
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
            "translation_z": self.translation_z,
        }


@dataclass
class UndefinedObject:
    object_id: int
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float

    def to_json(self) -> dict:
        return {
            "object_id": self.object_id,
            "top_left_x": self.top_left_x,
            "top_left_y": self.top_left_y,
            "bottom_right_x": self.bottom_right_x,
            "bottom_right_y": self.bottom_right_y,
        }


@dataclass
class ResultPackage:
    """Bir kare için sunucuya gönderilecek tam paket."""
    frame_url: str
    user_url: str = ""
    pred_id: Optional[int] = None
    detected_objects: List[DetectedObject] = field(default_factory=list)
    detected_translations: List[DetectedTranslation] = field(default_factory=list)
    detected_undefined_objects: List[UndefinedObject] = field(default_factory=list)

    def to_json(self) -> dict:
        out = {
            "frame": self.frame_url,
            "detected_objects": [o.to_json() for o in self.detected_objects],
            "detected_translations": [t.to_json() for t in self.detected_translations],
            "detected_undefined_objects": [u.to_json() for u in self.detected_undefined_objects],
        }
        if self.pred_id is not None:
            out["id"] = self.pred_id
        if self.user_url:
            out["user"] = self.user_url
        return out


def empty_result(frame_url: str) -> dict:
    """Boş/pas sonuç (bozuk kare veya ilk 10 kare)."""
    return ResultPackage(frame_url=frame_url).to_json()
