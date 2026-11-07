"""Config loading with YAML + CLI dotlist overrides."""
from __future__ import annotations
from typing import Any, List
import yaml


def _parse_scalar(v: str) -> Any:
    lv = v.lower()
    if lv in ("null", "none", "~"):
        return None
    if lv in ("true", "false"):
        return lv == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def load_config(path: str, overrides: List[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), _parse_scalar(v.strip()))
    return cfg
