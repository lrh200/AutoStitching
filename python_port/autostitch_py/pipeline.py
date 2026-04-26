from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .io_utils import (
    build_tp_correspondences,
    list_images,
    load_camera_priors_pix4d,
    load_tiepoint_priors,
)
from .matcher import FeatureMatcher
from .optimizer import GlobalHomographyOptimizer, OptimizerConfig
from .topology import PriorBundle, TopologyEstimator


@dataclass(slots=True)
class PipelineConfig:
    params_f: str
    tp_f: str
    img_dir: str
    is_time_consecutive: bool = False
    output_dir: str = "./python_port/output"
    ba_iters: int = 30
    ba_sample_step: int = 3
    ba_lambda: float = 1e-3


class MosaicPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, object]:
        images = list_images(self.cfg.img_dir)
        if len(images) < 2:
            raise ValueError("Need at least two images")

        cam = load_camera_priors_pix4d(self.cfg.params_f)
        tp = load_tiepoint_priors(self.cfg.tp_f)
        tp_corr_raw = build_tp_correspondences(self.cfg.tp_f)
        tp_corr = {
            k: (np.asarray(v[0], dtype=np.float32), np.asarray(v[1], dtype=np.float32))
            for k, v in tp_corr_raw.items()
        }

        priors = PriorBundle(
            xyz_by_name={k: v.xyz for k, v in cam.items()},
            tp_score_by_pair=tp,
        )

        matcher = FeatureMatcher(images, tp_corr_by_pair=tp_corr)
        estimator = TopologyEstimator([p.name for p in images], matcher, priors)
        topo = estimator.estimate(self.cfg.is_time_consecutive)

        pair_matches = self._collect_pair_matches(topo.similarity, matcher)
        models = self._optimize_models_direct_8dof(
            topo.visit_order,
            topo.visit_order[0],
            pair_matches,
            iters=self.cfg.ba_iters,
            sample_step=self.cfg.ba_sample_step,
            lm_lambda=self.cfg.ba_lambda,
        )
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

    def _collect_pair_matches(
        self, similarity: np.ndarray, matcher: FeatureMatcher
    ) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
        n = similarity.shape[0]
        out: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for i in range(n - 1):
            for j in range(i + 1, n):
                if similarity[i, j] <= 0:
                    continue
                res = matcher.full_match(i, j)
                if not res.ok:
                    continue
                out[(i, j)] = (res.points_a.astype(np.float64), res.points_b.astype(np.float64))
        return out

    def _optimize_models_direct_8dof(
        self,
        order: List[int],
        root: int,
        pair_matches: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
        iters: int,
        sample_step: int,
        lm_lambda: float,
    ) -> Dict[int, np.ndarray]:
        optimizer = GlobalHomographyOptimizer(
            OptimizerConfig(max_iters=iters, sample_step=sample_step, lm_lambda=lm_lambda)
        )
        models = optimizer.optimize(order=order, root_img=root, pair_matches=pair_matches)
        for img in order:
            if img not in models:
                models[img] = np.eye(3, dtype=np.float64)
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
