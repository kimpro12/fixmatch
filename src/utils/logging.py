"""Minimal run logger: console + CSV rows."""
from __future__ import annotations
import csv
import os
from typing import Dict, List, Optional


class CsvLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._init = False

    def log(self, row: Dict) -> None:
        mode = "a" if self._init or os.path.exists(self.path) else "w"
        # ensure header exists
        need_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if mode == "a" and need_header:
            mode = "w"
        with open(self.path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if mode == "w":
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._init = True
