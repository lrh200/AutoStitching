from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(slots=True)
class OptimizerConfig:
    max_iters: int = 30
    lm_lambda: float = 1e-3
    sample_step: int = 3
    eps: float = 1e-6


def _hvec_to_mat(h: np.ndarray) -> np.ndarray:
    return np.array(
        [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]],
        dtype=np.float64,
    )


def _mat_to_hvec(H: np.ndarray) -> np.ndarray:
    return np.array([H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]], dtype=np.float64)


def _project(h: np.ndarray, p: np.ndarray) -> np.ndarray:
    x, y = float(p[0]), float(p[1])
    w = h[6] * x + h[7] * y + 1.0
    u = (h[0] * x + h[1] * y + h[2]) / w
    v = (h[3] * x + h[4] * y + h[5]) / w
    return np.array([u, v], dtype=np.float64)


def _jacobian_numeric(h: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    base = _project(h, p)
    J = np.zeros((2, 8), dtype=np.float64)
    for k in range(8):
        hp = h.copy()
        hp[k] += eps
        J[:, k] = (_project(hp, p) - base) / eps
    return J


class GlobalHomographyOptimizer:
    """
    Ablation mode: remove linear initialization and directly optimize 8DoF homographies globally.
    Root image is fixed to identity.
    """

    def __init__(self, cfg: OptimizerConfig):
        self.cfg = cfg

    def optimize(
        self,
        order: List[int],
        root_img: int,
        pair_matches: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        # variables: all images except root
        vars_imgs = [img for img in order if img != root_img]
        var_pos = {img: i for i, img in enumerate(vars_imgs)}
        dim = 8 * len(vars_imgs)

        h_by_img: Dict[int, np.ndarray] = {img: np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64) for img in order}

        if dim == 0:
            return {root_img: np.eye(3, dtype=np.float64)}

        for _ in range(self.cfg.max_iters):
            AtA = np.zeros((dim, dim), dtype=np.float64)
            Atb = np.zeros((dim,), dtype=np.float64)
            res_sum = 0.0
            res_cnt = 0

            for (i, j), (pts_i, pts_j) in pair_matches.items():
                step = max(1, self.cfg.sample_step)
                for t in range(0, len(pts_i), step):
                    pi = pts_i[t]
                    pj = pts_j[t]

                    hi = h_by_img[i]
                    hj = h_by_img[j]
                    qi = _project(hi, pi)
                    qj = _project(hj, pj)
                    r = qi - qj  # 2-vector

                    res_sum += float(np.linalg.norm(r))
                    res_cnt += 1

                    Ji = _jacobian_numeric(hi, pi, self.cfg.eps)
                    Jj = -_jacobian_numeric(hj, pj, self.cfg.eps)

                    # accumulate normal equations with sparse blocks
                    if i != root_img:
                        bi = 8 * var_pos[i]
                        AtA[bi : bi + 8, bi : bi + 8] += Ji.T @ Ji
                        Atb[bi : bi + 8] += Ji.T @ (-r)
                    if j != root_img:
                        bj = 8 * var_pos[j]
                        AtA[bj : bj + 8, bj : bj + 8] += Jj.T @ Jj
                        Atb[bj : bj + 8] += Jj.T @ (-r)
                    if i != root_img and j != root_img:
                        bi = 8 * var_pos[i]
                        bj = 8 * var_pos[j]
                        cross = Ji.T @ Jj
                        AtA[bi : bi + 8, bj : bj + 8] += cross
                        AtA[bj : bj + 8, bi : bi + 8] += cross.T

            AtA += np.eye(dim, dtype=np.float64) * self.cfg.lm_lambda

            try:
                delta = np.linalg.solve(AtA, Atb)
            except np.linalg.LinAlgError:
                break

            mean_step = float(np.mean(np.abs(delta)))
            for img in vars_imgs:
                b = 8 * var_pos[img]
                h_by_img[img] += delta[b : b + 8]

            if mean_step < 1e-6:
                break

        models = {root_img: np.eye(3, dtype=np.float64)}
        for img in vars_imgs:
            models[img] = _hvec_to_mat(h_by_img[img])
        return models
