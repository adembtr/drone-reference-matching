#!/usr/bin/env python3
"""Central configuration loader. All modules read paths from here.

`competition.yaml` holds private, machine/competition-specific settings
(server URL, credentials) and is intentionally NOT committed. If it is absent,
this module falls back to safe defaults so the offline pipeline still imports
and runs. Copy `config/competition.example.yaml` to `config/competition.yaml`
and fill it in only if you need the online client.
"""
import os

try:
    import yaml
except Exception:  # pyyaml optional for the pure-offline path
    yaml = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(ROOT, "config", "competition.yaml")

# Safe defaults used when config/competition.yaml is missing (public checkout).
_DEFAULTS = {
    "paths": {
        "models_root": "models",
        "reference_dir": "offline_data/reference",
        "log_dir": "logs",
    },
    "session": {"gt_frames": 450, "fps": 7.5},
}


def load(path: str = CONFIG_PATH) -> dict:
    """Load competition.yaml if present, otherwise return safe defaults."""
    if yaml is None or not os.path.exists(path):
        return _DEFAULTS
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or _DEFAULTS


CFG = load()


def model_root(modality: str) -> str:
    """Absolute path to models/{modality}_models. modality in {'rgb','termal'}."""
    assert modality in ("rgb", "termal"), f"invalid modality: {modality}"
    return os.path.join(ROOT, CFG["paths"]["models_root"], f"{modality}_models")


def abspath(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
