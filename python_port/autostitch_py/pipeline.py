from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .io_utils import list_images, load_camera_priors, load_tiepoint_priors, pair_key
from .matcher import FeatureMatcher
from .topology import PriorBundle, TopologyEstimator


@dataclass(slots=True)
class PipelineConfig:
    params_f: str
    tp_f: str
    img_dir: str
    is_time_consecutive: bool = False
    output_dir: str = "./python_port/output"


class MosaicPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, object]:
        images = list_images(self.cfg.img_dir)
        if len(images) < 2:
            raise ValueError("Need at least two images")

        cam = load_camera_priors(self.cfg.params_f)
        tp = load_tiepoint_priors(self.cfg.tp_f)

        priors = PriorBundle(
            xyz_by_name={k: v.xyz for k, v in cam.items()},
            tp_score_by_pair=tp,
        )

        matcher = FeatureMatcher(images)
        estimator = TopologyEstimator([p.name for p in images], matcher, priors)
        topo = estimator.estimate(self.cfg.is_time_consecutive)

        models = self._initialize_models(images, topo.visit_order, topo.parents, matcher)
        pano_path = self._render_mosaic(images, topo.visit_order, models)

        report = {
            "num_images": len(images),
            "attempts": topo.attempts,
            "successful": topo.successful,
            "visit_order": topo.visit_order,
            "parents": topo.parents,
            "mosaic_path": str(pano_path),
        }
        with open(self.output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        np.savetxt(self.output_dir / "similarity_mat.txt", topo.similarity, fmt="%d")
        return report

    def _initialize_models(self, images: List[Path], order: List[int], parents: List[int], matcher: FeatureMatcher) -> Dict[int, np.ndarray]:
        models: Dict[int, np.ndarray] = {}
        root = order[0]
        models[root] = np.eye(3, dtype=np.float64)

        order_index = {node: idx for idx, node in enumerate(order)}
        for idx in range(1, len(order)):
            node = order[idx]
            parent = parents[idx]
            if parent < 0:
                models[node] = np.eye(3, dtype=np.float64)
                continue

            # parents store node-id in this implementation
            p_node = parent
            result = matcher.full_match(p_node, node)
            if not result.ok:
                models[node] = models.get(p_node, np.eye(3, dtype=np.float64)).copy()
                continue

            H, _ = cv2.findHomography(result.points_b, result.points_a, cv2.RANSAC, 3.0)
            if H is None:
                models[node] = models.get(p_node, np.eye(3, dtype=np.float64)).copy()
                continue
            models[node] = models[p_node] @ H
        return models

    def _render_mosaic(self, images: List[Path], order: List[int], models: Dict[int, np.ndarray]) -> Path:
        # estimate bounds
        corners_all = []
        wh: Dict[int, Tuple[int, int]] = {}
        for idx in order:
            img = cv2.imread(str(images[idx]))
            h, w = img.shape[:2]
            wh[idx] = (w, h)
            corners = np.array([[[0, 0]], [[w, 0]], [[0, h]], [[w, h]]], dtype=np.float32)
            wc = cv2.perspectiveTransform(corners, models[idx]).reshape(-1, 2)
            corners_all.append(wc)
        all_pts = np.vstack(corners_all)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)

        tx, ty = -min_xy[0], -min_xy[1]
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

        out_w = int(np.ceil(max_xy[0] - min_xy[0])) + 10
        out_h = int(np.ceil(max_xy[1] - min_xy[1])) + 10
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        mask = np.zeros((out_h, out_w), dtype=np.uint8)

        for idx in order:
            img = cv2.imread(str(images[idx]))
            H = T @ models[idx]
            warped = cv2.warpPerspective(img, H, (out_w, out_h))
            m = (warped.sum(axis=2) > 0).astype(np.uint8)
            # simple overwrite blending
            canvas[m > 0] = warped[m > 0]
            mask = np.maximum(mask, m)

        out_path = self.output_dir / "mosaic.png"
        cv2.imwrite(str(out_path), canvas)
        return out_path
