"""Smoke-train each method for 1 epoch on tiny subsets (needs CIFAR-10 download)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import load_config
from src.training.trainer import train_one_run


def _run(method, tmpdir, adaptive_enabled=False):
    cfg = load_config("configs/smoke.yaml")
    cfg["method"] = method
    cfg["adaptive"]["enabled"] = adaptive_enabled
    cfg["data"]["root"] = os.environ.get("CIFAR_ROOT", "./data")
    out = os.path.join(tmpdir, method)
    r = train_one_run(cfg, out)
    assert os.path.exists(os.path.join(out, "metrics.csv"))
    assert os.path.exists(os.path.join(out, "checkpoint_last.pt"))
    return r


def test_smoke_supervised(tmp_path):
    _run("supervised", str(tmp_path))


def test_smoke_fixmatch(tmp_path):
    _run("fixmatch", str(tmp_path))


def test_smoke_adaptive(tmp_path):
    _run("adaptive_fixmatch", str(tmp_path), adaptive_enabled=True)
