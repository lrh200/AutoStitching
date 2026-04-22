from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(slots=True)
class CameraPrior:
    image_name: str
    xyz: Tuple[float, float, float]


@dataclass(slots=True)
class TiePointPrior:
    image_a: str
    image_b: str
    score: float


def list_images(img_dir: str) -> List[Path]:
    root = Path(img_dir)
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def _tokenize_line(line: str) -> List[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return []
    if "," in line:
        return [tok.strip() for tok in line.split(",") if tok.strip()]
    return [tok for tok in re.split(r"\s+", line) if tok]


def load_camera_priors(params_file: str) -> Dict[str, CameraPrior]:
    """
    Flexible parser:
      - supports whitespace or csv
      - expects at least: image_name x y z
      - ignores extra columns
    """
    priors: Dict[str, CameraPrior] = {}
    with open(params_file, "r", encoding="utf-8") as f:
        for line in f:
            toks = _tokenize_line(line)
            if len(toks) < 4:
                continue
            name = Path(toks[0]).name
            try:
                x, y, z = float(toks[1]), float(toks[2]), float(toks[3])
            except ValueError:
                continue
            priors[name] = CameraPrior(image_name=name, xyz=(x, y, z))
    return priors


def load_tiepoint_priors(tp_file: str) -> Dict[Tuple[str, str], float]:
    """
    Flexible parser for Pix4D tiepoint co-visibility priors.
    Expected per row:
      image_a image_b score
    score can be tiepoint count, confidence, or overlap ratio.
    """
    priors: Dict[Tuple[str, str], float] = {}
    with open(tp_file, "r", encoding="utf-8") as f:
        for line in f:
            toks = _tokenize_line(line)
            if len(toks) < 3:
                continue
            a = Path(toks[0]).name
            b = Path(toks[1]).name
            try:
                score = float(toks[2])
            except ValueError:
                continue
            k = tuple(sorted((a, b)))
            priors[k] = max(priors.get(k, 0.0), score)
    return priors


def pair_key(name_a: str, name_b: str) -> Tuple[str, str]:
    return tuple(sorted((name_a, name_b)))
