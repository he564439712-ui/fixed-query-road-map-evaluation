"""Segmentation evaluation metrics.

Implements: IoU, Dice/F1, Precision, Recall, clDice, APLS.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def compute_iou(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> float:
    """Intersection over Union."""
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)
    tp = float(np.logical_and(pred_bool, gt_bool).sum())
    fp = float(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = float(np.logical_and(~pred_bool, gt_bool).sum())
    return tp / max(tp + fp + fn, 1.0)


def compute_dice(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> float:
    """Dice / F1 score."""
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)
    tp = float(np.logical_and(pred_bool, gt_bool).sum())
    fp = float(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = float(np.logical_and(~pred_bool, gt_bool).sum())
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)


def compute_precision(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> float:
    """Precision."""
    tp = float(np.logical_and(prediction, ground_truth).sum())
    fp = float(np.logical_and(prediction, ~ground_truth).sum())
    return tp / max(tp + fp, 1.0)


def compute_recall(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> float:
    """Recall."""
    tp = float(np.logical_and(prediction, ground_truth).sum())
    fn = float(np.logical_and(~prediction, ground_truth).sum())
    return tp / max(tp + fn, 1.0)


def compute_cldice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float, float]:
    """Centerline Dice (clDice) for topology-aware evaluation.

    clDice = 2 * T_prec * T_sens / (T_prec + T_sens)

    where:
      T_prec(pred, GT) = |pred_skel ∩ GT| / |pred_skel|
      T_sens(pred, GT) = |GT_skel ∩ pred| / |GT_skel|

    Returns (clDice, topology_precision, topology_sensitivity).
    """
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)

    pred_skel = skeletonize(pred_bool)
    gt_skel = skeletonize(gt_bool)

    # Topology precision: predicted centerline covered by the GT mask.
    t_prec = float(np.logical_and(pred_skel, gt_bool).sum())
    t_prec /= max(float(pred_skel.sum()), 1.0)

    # Topology sensitivity: GT centerline covered by the predicted mask.
    t_sens = float(np.logical_and(gt_skel, pred_bool).sum())
    t_sens /= max(float(gt_skel.sum()), 1.0)

    cldice = 2.0 * t_prec * t_sens / max(t_prec + t_sens, 1e-6)
    return float(cldice), float(t_prec), float(t_sens)


def _skeleton_graph(
    skeleton: np.ndarray,
) -> tuple[dict[tuple[int, int], dict[tuple[int, int], float]], list[tuple[int, int]]]:
    """Build a weighted road graph from a binary skeleton.

    Nodes are skeleton pixels whose 8-neighborhood degree != 2 (endpoints and
    junctions); for a closed loop an arbitrary pixel is used as a break point.
    Edges connect adjacent nodes along the skeleton path, weighted by the
    Euclidean path length. Returns (adj, nodes) with node = (x, y).
    """
    ys, xs = np.nonzero(skeleton)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    if not pixels:
        return {}, []

    def neighbors(p: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = p
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) == (0, 0):
                    continue
                q = (x + dx, y + dy)
                if q in pixels:
                    out.append(q)
        return out

    node_set = {p for p in pixels if len(neighbors(p)) != 2}
    if not node_set:
        node_set = {min(pixels)}  # closed loop: arbitrary break point
    nodes = list(node_set)

    adj: dict[tuple[int, int], dict[tuple[int, int], float]] = {p: {} for p in nodes}
    for p in nodes:
        for q in neighbors(p):
            prev, cur = p, q
            length = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            while cur not in node_set:
                nxt = [w for w in neighbors(cur) if w != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                length += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            if cur in node_set and cur != p:
                prev_len = adj[p].get(cur)
                if prev_len is None or length < prev_len:
                    adj[p][cur] = length
    return adj, nodes


def _dijkstra(
    adj: dict[tuple[int, int], dict[tuple[int, int], float]],
    source: tuple[int, int],
) -> dict[tuple[int, int], float]:
    """Shortest-path distances from source on the adjacency graph."""
    dist = {source: 0.0}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def compute_apls(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    max_path_length: float = 500.0,
    snap_radius: float = 8.0,
) -> float:
    """Compute an APLS-inspired local graph-path similarity score.

    Builds weighted road graphs from the GT and predicted skeletons, then for
    every GT node pair compares the GT shortest-path length against the
    shortest-path length on the predicted graph between the matched predicted
    nodes. Each GT node is matched to its nearest predicted node within
    snap_radius (unmatched -> unroutable). Unroutable predicted pairs score 0.

    Only GT paths no longer than ``max_path_length`` are evaluated. This is a
    lightweight raster-graph approximation, not the official SpaceNet APLS
    implementation. Returns a value in [0, 1] (1 = identical path lengths).
    """
    gt_bool = ground_truth.astype(bool)
    pred_bool = prediction.astype(bool)
    if not np.any(gt_bool):
        return 1.0 if not np.any(pred_bool) else 0.0

    gt_adj, gt_nodes = _skeleton_graph(skeletonize(gt_bool))
    pred_adj, pred_nodes = _skeleton_graph(skeletonize(pred_bool))
    if len(gt_nodes) < 2:
        return 1.0

    # Match each GT node to its nearest predicted node within snap_radius.
    match: dict[tuple[int, int], tuple[int, int] | None] = {}
    for u in gt_nodes:
        if u in pred_nodes:
            match[u] = u
            continue
        best, bd = None, snap_radius
        for p in pred_nodes:
            d = math.hypot(u[0] - p[0], u[1] - p[1])
            if d < bd:
                best, bd = p, d
        match[u] = best

    # Dijkstra from each unique matched predicted node (cached).
    pred_dists: dict[tuple[int, int], dict] = {}
    for u in gt_nodes:
        pu = match[u]
        if pu is not None and pu not in pred_dists:
            pred_dists[pu] = _dijkstra(pred_adj, pu)

    gt_dists: dict[tuple[int, int], dict] = {u: _dijkstra(gt_adj, u) for u in gt_nodes}

    total, count = 0.0, 0
    for i, u in enumerate(gt_nodes):
        for v in gt_nodes[i + 1 :]:
            d_gt = gt_dists[u].get(v)
            if d_gt is None or d_gt == float("inf"):
                continue  # disconnected GT pair imposes no constraint
            if d_gt > max_path_length:
                continue
            pu, pv = match[u], match[v]
            if pu is None or pv is None:
                d_pred = float("inf")
            else:
                d_pred = pred_dists[pu].get(pv, float("inf"))
            if d_pred == float("inf"):
                distortion = 1.0
            else:
                distortion = min(abs(d_gt - d_pred) / d_gt, 1.0)
            total += 1.0 - distortion
            count += 1
    return float(total / count) if count else 1.0


def segmentation_metrics(
    probability: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all segmentation metrics.

    Args:
        probability: (H, W) float probability map
        ground_truth: (H, W) binary mask
        threshold: binarization threshold

    Returns dict of metric_name → value.
    """
    pred_binary = probability >= threshold
    gt = ground_truth.astype(bool)

    return {
        "iou": compute_iou(pred_binary, gt),
        "dice": compute_dice(pred_binary, gt),
        "precision": compute_precision(pred_binary, gt),
        "recall": compute_recall(pred_binary, gt),
        "cldice": compute_cldice(pred_binary, gt)[0],
        "apls": compute_apls(pred_binary, gt),
    }
