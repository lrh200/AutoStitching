from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass(slots=True)
class TiePointObservation:
    point_id: int
    x: float
    y: float
    scale: float


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


def load_camera_priors_pix4d(params_file: str) -> Dict[str, CameraPrior]:
    """
    Parse Pix4D external camera parameter format:
      imageName X Y Z Omega Phi Kappa

    Robustness:
    - tolerate header lines
    - tolerate wrapped/broken lines by accumulating tokens
    """
    priors: Dict[str, CameraPrior] = {}
    current_name: str | None = None
    numeric_buf: List[float] = []

    with open(params_file, "r", encoding="utf-8") as f:
        for line in f:
            toks = _tokenize_line(line)
            if not toks:
                continue

            if len(toks) >= 7:
                # standard full row case
                name = Path(toks[0]).name
                try:
                    x, y, z = float(toks[1]), float(toks[2]), float(toks[3])
                except ValueError:
                    continue
                priors[name] = CameraPrior(image_name=name, xyz=(x, y, z))
                current_name = None
                numeric_buf.clear()
                continue

            # wrapped line case: image-name only line + numeric continuation lines
            if len(toks) == 1 and re.search(r"\.(jpg|jpeg|png|tif|tiff|bmp)$", toks[0], flags=re.IGNORECASE):
                current_name = Path(toks[0]).name
                numeric_buf.clear()
                continue

            if current_name is None:
                continue

            for t in toks:
                try:
                    numeric_buf.append(float(t))
                except ValueError:
                    pass
            if len(numeric_buf) >= 3:
                x, y, z = numeric_buf[0], numeric_buf[1], numeric_buf[2]
                priors[current_name] = CameraPrior(image_name=current_name, xyz=(x, y, z))
                current_name = None
                numeric_buf.clear()

    return priors


def parse_pix4d_tp_observations(tp_file: str) -> Dict[str, List[TiePointObservation]]:
    """
    Parse Pix4D tie-point observation file in repeating block form:
      image_name
      point_id x y scale
      point_id x y scale
      ...
      next_image_name
      ...

    Returns:
      {image_name: [TiePointObservation, ...]}
    """
    observations: Dict[str, List[TiePointObservation]] = {}
    current_image: str | None = None

    with open(tp_file, "r", encoding="utf-8") as f:
        for line in f:
            toks = _tokenize_line(line)
            if not toks:
                continue

            if len(toks) == 1:
                current_image = Path(toks[0]).name
                observations.setdefault(current_image, [])
                continue

            if len(toks) < 4 or current_image is None:
                continue

            try:
                point_id = int(float(toks[0]))
                x = float(toks[1])
                y = float(toks[2])
                scale = float(toks[3])
            except ValueError:
                continue

            observations[current_image].append(
                TiePointObservation(point_id=point_id, x=x, y=y, scale=scale)
            )

    return observations


def load_tiepoint_priors(tp_file: str) -> Dict[Tuple[str, str], float]:
    """
    Build pairwise co-visibility prior scores directly from Pix4D tie-point observations.
    score = number of shared point IDs between two images.
    """
    obs = parse_pix4d_tp_observations(tp_file)
    point_to_images: Dict[int, List[str]] = {}
    for img, arr in obs.items():
        for o in arr:
            point_to_images.setdefault(o.point_id, []).append(img)

    priors: Dict[Tuple[str, str], float] = {}
    for imgs in point_to_images.values():
        uniq = sorted(set(imgs))
        for i in range(len(uniq) - 1):
            for j in range(i + 1, len(uniq)):
                k = (uniq[i], uniq[j])
                priors[k] = priors.get(k, 0.0) + 1.0
    return priors


def build_tp_correspondences(
    tp_file: str,
) -> Dict[Tuple[str, str], Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
    """
    Build per-pair 2D-2D correspondences from shared point IDs.

    Returns:
      {
        (imgA, imgB): ([(xa,ya), ...], [(xb,yb), ...])
      }
    """
    obs = parse_pix4d_tp_observations(tp_file)
    by_img_point: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for img, arr in obs.items():
        m: Dict[int, Tuple[float, float]] = {}
        for o in arr:
            m[o.point_id] = (o.x, o.y)
        by_img_point[img] = m

    imgs = sorted(by_img_point.keys())
    pairs: Dict[Tuple[str, str], Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]] = {}
    for i in range(len(imgs) - 1):
        a = imgs[i]
        pa = by_img_point[a]
        for j in range(i + 1, len(imgs)):
            b = imgs[j]
            pb = by_img_point[b]
            shared = sorted(set(pa.keys()) & set(pb.keys()))
            if not shared:
                continue
            pts_a = [pa[pid] for pid in shared]
            pts_b = [pb[pid] for pid in shared]
            pairs[(a, b)] = (pts_a, pts_b)
    return pairs


def pair_key(name_a: str, name_b: str) -> Tuple[str, str]:
    return tuple(sorted((name_a, name_b)))
