from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class MatchResult:
    ok: bool
    points_a: np.ndarray
    points_b: np.ndarray


@dataclass(slots=True)
class FeatureData:
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray


class FeatureMatcher:
    def __init__(
        self,
        image_paths: List[Path],
        tp_corr_by_pair: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.image_paths = image_paths
        self.image_names = [p.name for p in image_paths]
        self._img_cache: Dict[int, np.ndarray] = {}
        self._feat_cache: Dict[int, FeatureData] = {}
        self.detector = cv2.ORB_create(nfeatures=6000)
        self.tp_corr_by_pair = tp_corr_by_pair or {}

    def read_image(self, idx: int) -> np.ndarray:
        if idx not in self._img_cache:
            img = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
            self._img_cache[idx] = img
        return self._img_cache[idx]

    def features(self, idx: int) -> FeatureData:
        if idx in self._feat_cache:
            return self._feat_cache[idx]
        img = self.read_image(idx)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = self.detector.detectAndCompute(gray, None)
        if des is None:
            des = np.zeros((0, 32), dtype=np.uint8)
        data = FeatureData(kps, des)
        self._feat_cache[idx] = data
        return data

    def quick_similarity(self, idx_a: int, idx_b: int, ratio: float = 0.7) -> int:
        a = self.features(idx_a)
        b = self.features(idx_b)
        if len(a.keypoints) < 2 or len(b.keypoints) < 2:
            return 0
        knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(a.descriptors, b.descriptors, k=2)
        good = 0
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good += 1
        return good

    def full_match(self, idx_a: int, idx_b: int) -> MatchResult:
        key = tuple(sorted((self.image_names[idx_a], self.image_names[idx_b])))
        if key in self.tp_corr_by_pair:
            pts_a, pts_b = self.tp_corr_by_pair[key]
            if self.image_names[idx_a] != key[0]:
                pts_a, pts_b = pts_b, pts_a
            return self._verify_points(pts_a, pts_b, reproj_thresh=3.0, min_points=10)

        a = self.features(idx_a)
        b = self.features(idx_b)
        if len(a.keypoints) < 20 or len(b.keypoints) < 20:
            return MatchResult(False, np.empty((0, 2)), np.empty((0, 2)))

        knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(a.descriptors, b.descriptors, k=2)
        prelim = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.7 * n.distance:
                prelim.append(m)
        if len(prelim) < 10:
            return MatchResult(False, np.empty((0, 2)), np.empty((0, 2)))

        pts_a = np.float32([a.keypoints[m.queryIdx].pt for m in prelim])
        pts_b = np.float32([b.keypoints[m.trainIdx].pt for m in prelim])

        return self._verify_points(pts_a, pts_b, reproj_thresh=3.0, min_points=10)

    def _verify_points(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
        reproj_thresh: float,
        min_points: int,
    ) -> MatchResult:
        if len(pts_a) < min_points:
            return MatchResult(False, np.empty((0, 2)), np.empty((0, 2)))

        _, inlier = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.5, 0.99)
        if inlier is None:
            return MatchResult(False, np.empty((0, 2)), np.empty((0, 2)))
        inlier = inlier.ravel().astype(bool)
        pts_a = pts_a[inlier]
        pts_b = pts_b[inlier]
        if len(pts_a) < min_points:
            return MatchResult(False, np.empty((0, 2)), np.empty((0, 2)))

        H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 2.5)
        if H is None:
            return MatchResult(False, np.empty((0, 2)), np.empty((0, 2)))

        pts_b_h = cv2.perspectiveTransform(pts_b.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(pts_b_h - pts_a, axis=1)
        keep = err < reproj_thresh
        pts_a = pts_a[keep]
        pts_b = pts_b[keep]

        ok = len(pts_a) >= min_points
        return MatchResult(ok, pts_a, pts_b)
