"""AA-DC scoring (Adamic-Adar with a degree-calibrated gate and L3 rescue), used as a baseline and as FLEX's +AD channel.

The functions below are copied verbatim from the authors' reference implementation, aa_dc.py
(Copyright (c) 2026 Daisuke Hirota, MIT License; see third_party/AA-DC_LICENSE). Only the functions FLEX calls and their
helpers are kept; the CLI, data loading and OGB evaluation of the original file are omitted.
"""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as ssp
import torch
from tqdm import tqdm

# Defaults of the reference CLI (aa_dc.py argparse), used for every cell.
EXT_THRESHOLD = 0.5
EXT_PENALTY = 0.5
ANCHOR_SCALE = 1e-6
ANCHOR_Q = 0.9

def build_adj(num_nodes: int, edges: np.ndarray) -> List[Set[int]]:
    """Build an unweighted adjacency list (set-based) from edge array."""
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def build_weighted_adj(
    num_nodes: int, edges: np.ndarray, weights: np.ndarray
) -> ssp.csr_matrix:
    """Build a symmetric weighted adjacency matrix (CSR) from edges + weights."""
    row = edges[:, 0].astype(np.int64)
    col = edges[:, 1].astype(np.int64)
    row_full = np.concatenate([row, col])
    col_full = np.concatenate([col, row])
    w_full = np.concatenate([weights, weights]).astype(np.float32)
    A = ssp.csr_matrix((w_full, (row_full, col_full)), shape=(num_nodes, num_nodes))
    return A.tocsr()


def precompute_inv_log_deg(A: ssp.csr_matrix) -> np.ndarray:
    """Compute 1/log(degree) for AA weighting. Nodes with deg <= 1 get 0."""
    deg = np.array(A.sum(axis=1)).flatten().astype(np.float64)
    inv = np.zeros_like(deg, dtype=np.float64)
    mask = deg > 1.0
    inv[mask] = 1.0 / np.log(deg[mask])
    return inv.astype(np.float32)


def precompute_aa_matrix(A: ssp.csr_matrix, inv_log_deg: np.ndarray) -> ssp.csr_matrix:
    """Precompute A * diag(inv_log_deg) for vectorized AA scoring."""
    return A.multiply(inv_log_deg).tocsr()


def cn_external_ratio(u: int, v: int, adj: List[Set[int]]) -> float:
    """Fraction of each common neighbor's links that point outside the local set.

    Local set = N(u) | N(v) | {u, v}.
    High ext_ratio signals pseudo-cohesion: common neighbors are externally
    oriented despite forming an apparent local cluster.
    """
    cn_set = adj[u] & adj[v]
    k = len(cn_set)
    if k == 0:
        return 0.0

    local_set = adj[u] | adj[v] | {u, v}
    total = 0.0
    for w in cn_set:
        adj_w = adj[w]
        deg_w = len(adj_w)
        if deg_w == 0:
            continue
        external = len(adj_w - local_set)
        total += external / deg_w
    return total / k


def compute_gate_value(
    ext_ratio: float,
    gate_mode: str,
    ext_threshold: float,
    ext_penalty: float,
    avg_lcc: float = 0.0,
    gate_thresholds: Optional[Dict[str, float]] = None,
) -> float:
    """Compute gate value from ext_ratio.

    gate_mode='progressive' (primary):
        Data-driven 2-stage gating with calibrated thresholds.
        Stage 1: ext_ratio > thresh_1 -> basic external orientation (gate=0.5)
        Stage 2: + avg_lcc > thresh_1 -> cross-fiber bridge pattern (gate=0.25)

    gate_mode='threshold' (used internally during calibration):
        Binary gate: ext_penalty if ext_ratio > ext_threshold, else 1.0
    """
    if gate_mode == "threshold":
        return ext_penalty if ext_ratio > ext_threshold else 1.0

    if gate_mode == "progressive":
        if gate_thresholds is None:
            return ext_penalty if ext_ratio > ext_threshold else 1.0

        gate = 1.0

        # Stage 1: Basic external orientation
        if ext_ratio > gate_thresholds.get("ext_thresh_1", ext_threshold):
            gate = 0.5

        # Stage 2: Clear cross-fiber bridge pattern
        if ext_ratio > gate_thresholds.get(
            "ext_thresh_2", ext_threshold + 0.1
        ) and avg_lcc > gate_thresholds.get("lcc_thresh_1", 0.5):
            gate *= 0.5  # -> 0.25

        return gate

    raise ValueError(f"Unknown gate_mode: {gate_mode}")


def build_twohop_contrib_dict(u: int, A: ssp.csr_matrix) -> Dict[int, float]:
    """Build {node: weighted_contribution} for u's 2-hop neighborhood."""
    row_u = A.getrow(u)
    twohop = row_u.dot(A).tocsr()
    return {int(i): float(v) for i, v in zip(twohop.indices, twohop.data)}


def l3_score(
    v: int,
    A: ssp.csr_matrix,
    inv_log_deg: np.ndarray,
    twohop_dict: Dict[int, float],
) -> float:
    """Compute L3 score for edge (u, v) given u's precomputed 2-hop dict.

    L3 path: u -> N(u) -> 2hop(u) -> N(v) -> v.
    The score aggregates weighted contributions through 3-hop paths,
    penalized by 1/log(deg) at intermediate nodes.
    """
    row_v = A.getrow(v)
    nbrs = row_v.indices
    if nbrs.size == 0:
        return 0.0
    s = 0.0
    for t in nbrs:
        tv = twohop_dict.get(int(t), 0.0)
        if tv != 0.0:
            s += tv * float(inv_log_deg[int(t)])
    return float(s)


def compute_exact_lcc(A: ssp.csr_matrix) -> np.ndarray:
    """Compute exact LCC for all nodes via sparse matrix multiplication.

    LCC_i = triangles_i / (d_i * (d_i - 1) / 2)
    """
    A_bin = A.copy()
    A_bin.data = np.ones_like(A_bin.data, dtype=np.float32)
    deg = np.array(A_bin.sum(axis=1)).flatten()
    print("    Computing exact triangles via sparse matrix mult ...")

    triangles_2x = A_bin.dot(A_bin).multiply(A_bin).sum(axis=1).A1
    denom = deg * (deg - 1)
    lcc = np.zeros_like(deg, dtype=np.float32)
    mask = denom > 0
    lcc[mask] = (triangles_2x[mask] / denom[mask]).astype(np.float32)
    return lcc


def score_edges(
    edges: np.ndarray,
    A: ssp.csr_matrix,
    A_invlog: ssp.csr_matrix,
    inv_log_deg: np.ndarray,
    adj: List[Set[int]],
    use_gate: bool,
    gate_mode: str,
    ext_threshold: float,
    ext_penalty: float,
    lcc: Optional[np.ndarray],
    use_l3: bool,
    rescue_mode: str,
    anchor_scale: float,
    gate_thresholds: Optional[Dict[str, float]] = None,
    collect_l3: bool = False,
    show_progress: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, object],
]:
    """Score all edges with AA (+ optional gate) and L3 rescue.

    Self-loops are forced to score 0 and excluded from all computations.

    Returns:
        scores:    (E,) final scores
        aa_vals:   (E,) raw AA values
        gate_vals: (E,) gate values (None if not use_gate)
        ext_vals:  (E,) ext_ratio values (None if not use_gate)
        stats:     diagnostic counters
    """
    E = edges.shape[0]
    scores = np.zeros(E, dtype=np.float32)
    aa_vals = np.zeros(E, dtype=np.float32)
    gate_vals = np.full(E, np.nan, dtype=np.float32)
    ext_vals = np.full(E, np.nan, dtype=np.float32)

    aa0_idx: List[int] = []
    self_loop_count = 0

    # Batch AA computation
    print("  Batch-computing AA scores ...")
    us = edges[:, 0]
    vs = edges[:, 1]
    rows_u = A[us]
    rows_v = A_invlog[vs]
    aa_batch = np.array(rows_u.multiply(rows_v).sum(axis=1)).flatten()

    it = range(E)
    if show_progress:
        it = tqdm(it, desc="AA(+gate)", leave=False)

    for i in it:
        u = int(edges[i, 0])
        v = int(edges[i, 1])

        if u == v:
            self_loop_count += 1
            aa0_idx.append(i)
            if use_gate:
                gate_vals[i] = 0.0
                ext_vals[i] = 0.0
            continue

        aa = float(aa_batch[i])
        aa_vals[i] = aa

        if aa == 0.0:
            aa0_idx.append(i)
            continue

        if use_gate:
            ext = cn_external_ratio(u, v, adj)
            ext_vals[i] = ext

            avg_lcc_val = 0.0
            if lcc is not None:
                avg_lcc_val = (float(lcc[u]) + float(lcc[v])) / 2.0

            g = compute_gate_value(
                ext,
                gate_mode,
                ext_threshold,
                ext_penalty,
                avg_lcc=avg_lcc_val,
                gate_thresholds=gate_thresholds,
            )
            gate_vals[i] = g
            scores[i] = float(aa * g)
        else:
            scores[i] = aa

    stats: Dict[str, object] = {
        "E": E,
        "aa0": len(aa0_idx),
        "aa_pos": E - len(aa0_idx),
        "self_loops": self_loop_count,
        "l3_called_u": 0,
        "l3_scored_edges": 0,
        "l3_nonzero": 0,
    }

    if collect_l3:
        stats["l3_values"] = np.full(E, np.nan, dtype=np.float32)

    # L3 rescue for AA==0 edges (anchor mode: score = anchor_scale * L3)
    if use_l3 and len(aa0_idx) > 0:
        idx_by_u: Dict[int, List[int]] = {}
        for i in aa0_idx:
            u = int(edges[i, 0])
            v = int(edges[i, 1])
            if u == v:
                continue
            idx_by_u.setdefault(u, []).append(i)

        u_list = list(idx_by_u.keys())
        if show_progress:
            u_list = tqdm(u_list, desc="L3(twohop by u)", leave=False)

        for u in u_list:
            twohop = build_twohop_contrib_dict(u, A)
            stats["l3_called_u"] += 1

            for i in idx_by_u[u]:
                v = int(edges[i, 1])
                l3 = l3_score(v, A, inv_log_deg, twohop)
                if collect_l3:
                    stats["l3_values"][i] = float(l3)
                if l3 != 0.0:
                    stats["l3_nonzero"] += 1
                stats["l3_scored_edges"] += 1

                # Place pseudo-fragmentation edges near the AA>0 boundary
                scores[i] = float(anchor_scale * l3)

    return (
        scores,
        aa_vals,
        gate_vals if use_gate else None,
        ext_vals if use_gate else None,
        stats,
    )


def calibrate_gate_thresholds(
    pos_ext_ratio: np.ndarray,
    pos_avg_lcc: np.ndarray,
    neg_ext_ratio: np.ndarray,
    neg_avg_lcc: np.ndarray,
    *,
    pos_protection_q: float = 0.95,
    margin: float = 0.10,
    neg_quantiles: tuple = (0.33, 0.67),
) -> Dict[str, float]:
    """Calibrate progressive gate thresholds from validation data.

    Principle:
      1. Protect positive distribution: use pos_protection_q quantile + margin
      2. Progressive negative targeting: use quantiles of problematic negatives
    """
    pos_ext_high = float(np.quantile(pos_ext_ratio, pos_protection_q))
    pos_lcc_high = float(np.quantile(pos_avg_lcc, pos_protection_q))
    ext_thresh_1 = pos_ext_high + margin
    lcc_thresh_1 = pos_lcc_high + margin

    neg_above_base = neg_ext_ratio > ext_thresh_1
    n_neg_above = int(neg_above_base.sum())

    if n_neg_above > 0:
        neg_ext_filtered = neg_ext_ratio[neg_above_base]
        ext_thresh_2 = float(np.quantile(neg_ext_filtered, neg_quantiles[0]))
    else:
        ext_thresh_2 = ext_thresh_1 + 0.1

    pos_violate_1 = int((pos_ext_ratio > ext_thresh_1).sum())
    pos_violate_2 = int(
        ((pos_ext_ratio > ext_thresh_2) & (pos_avg_lcc > lcc_thresh_1)).sum()
    )
    neg_hit_1 = int((neg_ext_ratio > ext_thresh_1).sum())
    neg_hit_2 = int(
        ((neg_ext_ratio > ext_thresh_2) & (neg_avg_lcc > lcc_thresh_1)).sum()
    )

    return {
        "ext_thresh_1": ext_thresh_1,
        "ext_thresh_2": ext_thresh_2,
        "lcc_thresh_1": lcc_thresh_1,
        "pos_ext_q": pos_ext_high,
        "pos_lcc_q": pos_lcc_high,
        "margin": margin,
        "pos_protection_q": pos_protection_q,
        "pos_violate_1": pos_violate_1,
        "pos_violate_2": pos_violate_2,
        "neg_hit_1": neg_hit_1,
        "neg_hit_2": neg_hit_2,
        "n_pos": len(pos_ext_ratio),
        "n_neg": len(neg_ext_ratio),
        "n_neg_above_base": n_neg_above,
    }


def optimize_anchor_q(
    y_pos_base: np.ndarray,
    y_neg_base: np.ndarray,
    l3_pos_vals: np.ndarray,
    l3_neg_vals: np.ndarray,
    kth_aa: float,
    K: int = 50,
    q_grid: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Optimize anchor_q by maximizing Hits@K on the validation set.

    Sweeps anchor_q over q_grid with gate_floor fixed at 0. For each q,
    computes anchor_scale = kth@K / quantile(pos_L3_nz, q) and simulates
    the resulting Hits@K without re-running score_edges.
    """
    if q_grid is None:
        q_grid = np.arange(0.50, 0.99, 0.02)

    pos_l3_mask = (~np.isnan(l3_pos_vals)) & (l3_pos_vals > 0)
    neg_l3_mask = (~np.isnan(l3_neg_vals)) & (l3_neg_vals > 0)

    l3_pos_nz = l3_pos_vals[pos_l3_mask]

    if l3_pos_nz.size == 0:
        return {
            "best_q": 0.9,
            "best_hits": 0.0,
            "best_anchor_scale": 1e-6,
            "grid": [],
            "note": "fallback - no L3 data",
        }

    best_q = 0.9
    best_hits = -1.0
    best_scale = 1e-6
    grid_results = []

    y_pos_work = y_pos_base.copy()
    y_neg_work = y_neg_base.copy()

    # Zero out AA==0 positions before sweep
    pos_aa0_l3zero = (~np.isnan(l3_pos_vals)) & (l3_pos_vals == 0)
    neg_aa0_l3zero = (~np.isnan(l3_neg_vals)) & (l3_neg_vals == 0)
    y_pos_work[pos_l3_mask] = 0.0
    y_pos_work[pos_aa0_l3zero] = 0.0
    y_neg_work[neg_l3_mask] = 0.0
    y_neg_work[neg_aa0_l3zero] = 0.0

    for q in q_grid:
        Qpos = float(np.quantile(l3_pos_nz, q))
        if Qpos <= 0:
            continue
        scale = kth_aa / Qpos

        y_pos_work[pos_l3_mask] = l3_pos_vals[pos_l3_mask] * scale
        y_neg_work[neg_l3_mask] = l3_neg_vals[neg_l3_mask] * scale

        if y_neg_work.size >= K:
            kth = float(np.partition(y_neg_work, -K)[-K])
        else:
            kth = float(np.max(y_neg_work))

        hits = float(np.mean(y_pos_work > kth))
        neg_intrusion = (
            int((y_neg_work[neg_l3_mask] > kth).sum()) if neg_l3_mask.any() else 0
        )
        pos_rescued = (
            int((y_pos_work[pos_l3_mask] > kth).sum()) if pos_l3_mask.any() else 0
        )

        grid_results.append(
            {
                "q": float(q),
                "scale": float(scale),
                "hits": float(hits),
                "kth": float(kth),
                "neg_intrusion": neg_intrusion,
                "pos_rescued": pos_rescued,
            }
        )

        if hits > best_hits:
            best_hits = hits
            best_q = float(q)
            best_scale = float(scale)

        # Reset for next iteration
        y_pos_work[pos_l3_mask] = 0.0
        y_neg_work[neg_l3_mask] = 0.0

    return {
        "best_q": best_q,
        "best_hits": best_hits,
        "best_anchor_scale": best_scale,
        "grid": grid_results,
    }
