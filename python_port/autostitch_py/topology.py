from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .io_utils import pair_key
from .matcher import FeatureMatcher


MAX_COST = 9999.0


@dataclass(slots=True)
class TopologyResult:
    similarity: np.ndarray
    attempts: int
    successful: int
    visit_order: List[int]
    parents: List[int]


@dataclass(slots=True)
class PriorBundle:
    xyz_by_name: Dict[str, Tuple[float, float, float]]
    tp_score_by_pair: Dict[Tuple[str, str], float]


class TopologyEstimator:
    def __init__(self, image_names: List[str], matcher: FeatureMatcher, priors: PriorBundle):
        self.names = image_names
        self.matcher = matcher
        self.priors = priors
        self.n = len(image_names)

    def _cost_from_similarity(self, s: float) -> float:
        if s <= 0:
            return -1.0
        return 6.0 / np.log(s + 50.0)

    def _guiding_cost(self, i: int, j: int) -> float:
        ni, nj = self.names[i], self.names[j]
        key = pair_key(ni, nj)
        tp = self.priors.tp_score_by_pair.get(key, 0.0)
        sim = self.matcher.quick_similarity(i, j) if tp <= 0 else 0.0

        # use tie-point co-visibility as first-class retrieval basis;
        # ORB quick score is fallback when tp prior is absent.
        fused = tp if tp > 0 else sim

        # optional camera prior penalty
        if ni in self.priors.xyz_by_name and nj in self.priors.xyz_by_name:
            pi = np.array(self.priors.xyz_by_name[ni])
            pj = np.array(self.priors.xyz_by_name[nj])
            dist = np.linalg.norm(pi - pj)
            fused = fused / (1.0 + 0.01 * dist)

        return self._cost_from_similarity(fused)

    def _build_guiding_table(self) -> np.ndarray:
        g = np.full((self.n, self.n), -1.0, dtype=np.float64)
        np.fill_diagonal(g, 0.0)
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                c = self._guiding_cost(i, j)
                g[i, j] = c
                g[j, i] = c
        return g

    def _extract_mst_edges(self, graph: np.ndarray) -> List[Tuple[int, int]]:
        g = graph.copy()
        g[g < 0] = MAX_COST
        n = g.shape[0]
        in_tree = np.zeros(n, dtype=bool)
        low = np.full(n, MAX_COST)
        adj = np.zeros(n, dtype=np.int32)
        in_tree[0] = True
        low[:] = g[0]
        edges: List[Tuple[int, int]] = []
        for _ in range(n - 1):
            j = -1
            best = MAX_COST
            for k in range(1, n):
                if (not in_tree[k]) and low[k] < best:
                    best = low[k]
                    j = k
            if j < 0:
                break
            in_tree[j] = True
            edges.append((int(adj[j]), int(j)))
            for k in range(1, n):
                if (not in_tree[k]) and g[j, k] < low[k]:
                    low[k] = g[j, k]
                    adj[k] = j
        return edges

    def _floyd_root(self, cost: np.ndarray) -> Tuple[int, np.ndarray]:
        dist = np.where(cost > 0, cost, MAX_COST).astype(np.float64)
        np.fill_diagonal(dist, 0.0)
        parent = np.full((self.n, self.n), -1, dtype=np.int32)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    parent[i, j] = i
                elif cost[i, j] > 0:
                    parent[i, j] = i

        for k in range(self.n):
            for i in range(self.n):
                cand = dist[i, k] + dist[k]
                better = cand < dist[i]
                dist[i, better] = cand[better]
                parent[i, better] = parent[k, better]

        sums = dist.sum(axis=1)
        root = int(np.argmin(sums))
        return root, parent[root]

    def _bfs_order(self, parent_row: np.ndarray, root: int) -> Tuple[List[int], List[int]]:
        parent = parent_row.copy()
        parent[root] = -1
        order = [root]
        parents = [-1]
        heads = [root]
        level_heads: List[int] = []
        while len(order) < self.n:
            for h in heads:
                for node in range(self.n):
                    if parent[node] == h and node not in order:
                        level_heads.append(node)
                        order.append(node)
                        parents.append(h)
            if not level_heads:
                remain = [k for k in range(self.n) if k not in order]
                for r in remain:
                    order.append(r)
                    parents.append(root)
                break
            heads = level_heads
            level_heads = []
        return order, parents

    def estimate(self, is_time_consecutive: bool) -> TopologyResult:
        similarity = np.zeros((self.n, self.n), dtype=np.int32)
        attempted = np.eye(self.n, dtype=np.uint8)
        attempts = 0
        successful = 0

        if is_time_consecutive:
            # main chain: adjacent pairs only
            for i in range(self.n - 1):
                j = i + 1
                result = self.matcher.full_match(i, j)
                attempts += 1
                attempted[i, j] = 1
                attempted[j, i] = 1
                if result.ok:
                    similarity[i, j] = len(result.points_a)
                    similarity[j, i] = len(result.points_a)
                    successful += 1
            root = self.n // 2
            parents = np.full(self.n, -1, dtype=np.int32)
            for i in range(self.n):
                if i == root:
                    continue
                parents[i] = i - 1 if i > root else i + 1
            order, parent_list = self._bfs_order(parents, root)
        else:
            guiding = self._build_guiding_table()
            max_iter = max(int(self.n * 0.2), 20)
            for _ in range(max_iter):
                built = 0
                for i, j in self._extract_mst_edges(guiding):
                    if attempted[i, j] != 0:
                        built += 1
                        continue
                    attempted[i, j] = 1
                    attempted[j, i] = 1
                    attempts += 1
                    result = self.matcher.full_match(i, j)
                    if result.ok:
                        successful += 1
                        cnt = len(result.points_a)
                        similarity[i, j] = cnt
                        similarity[j, i] = cnt
                        guiding[i, j] = 0.0
                        guiding[j, i] = 0.0
                        built += 1
                    else:
                        guiding[i, j] = 999.0
                        guiding[j, i] = 999.0
                        break
                if built >= self.n - 1:
                    break
            cost = np.full((self.n, self.n), -1.0, dtype=np.float64)
            for i in range(self.n):
                for j in range(self.n):
                    cost[i, j] = self._cost_from_similarity(float(similarity[i, j]))
            root, parent_row = self._floyd_root(cost)
            order, parent_list = self._bfs_order(parent_row, root)

        return TopologyResult(similarity=similarity, attempts=attempts, successful=successful, visit_order=order, parents=parent_list)
