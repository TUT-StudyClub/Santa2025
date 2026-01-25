"""
exp023_penalty_gls_v2: SSA + Penalty Learning (GLS) with CORRECT SCALE & INTERSECTION CHECK & INFINITE LOOP

スケール設定を修正し、さらに「線分交差判定」を追加したSSA + GLSの実装。
無限ループとランダムリスタート、過去のsubmission活用を導入。
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numba import njit
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def resolve_config_path() -> str:
    config_name = "000"
    for arg in sys.argv[1:]:
        if arg.startswith("exp="):
            config_name = arg.split("=", 1)[1]
    return os.path.join("experiments", "exp023_penalty", "exp", f"{config_name}.yaml")


CONFIG_PATH = resolve_config_path()
if not os.path.exists(CONFIG_PATH):
    if os.path.exists(f"{os.path.basename(CONFIG_PATH)}"):
        CONFIG_PATH = f"{os.path.basename(CONFIG_PATH)}"

try:
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

# 正しい小さいスケール設定
TREE_CFG = CONFIG.get("tree_shape", {})
TRUNK_W = float(TREE_CFG.get("trunk_w", 0.15))
BASE_W = float(TREE_CFG.get("base_w", 0.7))
MID_W = float(TREE_CFG.get("mid_w", 0.4))
TOP_W = float(TREE_CFG.get("top_w", 0.25))
TIP_Y = float(TREE_CFG.get("tip_y", 0.8))
TIER_1_Y = float(TREE_CFG.get("tier_1_y", 0.5))
TIER_2_Y = float(TREE_CFG.get("tier_2_y", 0.25))
BASE_Y = float(TREE_CFG.get("base_y", 0.0))
TRUNK_BOTTOM_Y = float(TREE_CFG.get("trunk_bottom_y", -0.2))


# -----------------------------------------------------------------------------
# Geometry Utils (Numba) - REVISED
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def rotate_point(x: float, y: float, cos_a: float, sin_a: float) -> tuple[float, float]:
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


@njit(cache=True, fastmath=True)
def get_tree_vertices(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    vertices = np.empty((15, 2), dtype=np.float64)

    pts = np.array(
        [
            [0.0, TIP_Y],
            [TOP_W / 2.0, TIER_1_Y],
            [TOP_W / 4.0, TIER_1_Y],
            [MID_W / 2.0, TIER_2_Y],
            [MID_W / 4.0, TIER_2_Y],
            [BASE_W / 2.0, BASE_Y],
            [TRUNK_W / 2.0, BASE_Y],
            [TRUNK_W / 2.0, TRUNK_BOTTOM_Y],
            [-TRUNK_W / 2.0, TRUNK_BOTTOM_Y],
            [-TRUNK_W / 2.0, BASE_Y],
            [-BASE_W / 2.0, BASE_Y],
            [-MID_W / 4.0, TIER_2_Y],
            [-MID_W / 2.0, TIER_2_Y],
            [-TOP_W / 4.0, TIER_1_Y],
            [-TOP_W / 2.0, TIER_1_Y],
        ],
        dtype=np.float64,
    )

    for i in range(15):
        rx, ry = rotate_point(pts[i, 0], pts[i, 1], cos_a, sin_a)
        vertices[i, 0] = rx + cx
        vertices[i, 1] = ry + cy
    return vertices


@njit(cache=True, fastmath=True)
def polygon_bounds(vertices: np.ndarray) -> tuple[float, float, float, float]:
    min_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_x = vertices[0, 0]
    max_y = vertices[0, 1]
    for i in range(1, vertices.shape[0]):
        x = vertices[i, 0]
        y = vertices[i, 1]
        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_y = min(y, min_y)
        max_y = max(y, max_y)
    return min_x, min_y, max_x, max_y


@njit(cache=True, fastmath=True)
def point_in_polygon(px: float, py: float, vertices: np.ndarray) -> bool:
    n = vertices.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i, 0], vertices[i, 1]
        xj, yj = vertices[j, 0], vertices[j, 1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@njit(cache=True, fastmath=True)
def segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y) -> bool:
    """線分 (p1-p2) と (p3-p4) が交差しているか判定"""
    d1x, d1y = p2x - p1x, p2y - p1y
    d2x, d2y = p4x - p3x, p4y - p3y
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-10:
        return False
    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / det
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / det
    # 端点接触は許容するため、厳密不等号ではなく余裕を持たせる
    return 1e-8 < t < 1.0 - 1e-8 and 1e-8 < u < 1.0 - 1e-8


@njit(cache=True, fastmath=True)
def point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
    l2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    if l2 == 0:
        return (px - x1) ** 2 + (py - y1) ** 2
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


@njit(cache=True, fastmath=True)
def calculate_pairwise_overlap_matrix(all_vertices: list[np.ndarray]) -> np.ndarray:
    """
    修正版: 頂点包含だけでなく、辺の交差も検知してペナルティを与える
    """
    n = len(all_vertices)
    overlap_mat = np.zeros((n, n), dtype=np.float64)
    nv = 15  # 頂点数

    for i in range(n):
        min_x1, min_y1, max_x1, max_y1 = polygon_bounds(all_vertices[i])
        for j in range(i + 1, n):
            min_x2, min_y2, max_x2, max_y2 = polygon_bounds(all_vertices[j])

            # AABB check
            if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                continue

            verts_i = all_vertices[i]
            verts_j = all_vertices[j]

            depth = 0.0
            has_intersection = False

            # 1. 頂点が内部にあるかチェック (Point in Polygon)
            for k in range(nv):
                px, py = verts_i[k, 0], verts_i[k, 1]
                if point_in_polygon(px, py, verts_j):
                    min_dist_sq = 1e15
                    for m in range(nv):
                        m_next = (m + 1) % nv
                        d_sq = point_to_segment_dist_sq(
                            px, py, verts_j[m, 0], verts_j[m, 1], verts_j[m_next, 0], verts_j[m_next, 1]
                        )
                        min_dist_sq = min(d_sq, min_dist_sq)
                    depth += math.sqrt(min_dist_sq)

            for k in range(nv):
                px, py = verts_j[k, 0], verts_j[k, 1]
                if point_in_polygon(px, py, verts_i):
                    min_dist_sq = 1e15
                    for m in range(nv):
                        m_next = (m + 1) % nv
                        d_sq = point_to_segment_dist_sq(
                            px, py, verts_i[m, 0], verts_i[m, 1], verts_i[m_next, 0], verts_i[m_next, 1]
                        )
                        min_dist_sq = min(d_sq, min_dist_sq)
                    depth += math.sqrt(min_dist_sq)

            # 2. 辺が交差しているかチェック (Edge Intersection)
            if depth < 1e-9:
                for k in range(nv):
                    k_next = (k + 1) % nv
                    p1x, p1y = verts_i[k, 0], verts_i[k, 1]
                    p2x, p2y = verts_i[k_next, 0], verts_i[k_next, 1]

                    for m in range(nv):
                        m_next = (m + 1) % nv
                        p3x, p3y = verts_j[m, 0], verts_j[m, 1]
                        p4x, p4y = verts_j[m_next, 0], verts_j[m_next, 1]

                        if segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
                            has_intersection = True
                            break
                    if has_intersection:
                        break

                if has_intersection:
                    # 改善: 交差ペナルティをより適切な値に（重なり深さの計算と整合性を保つ）
                    depth = 0.05  # 0.1から0.05に調整（過度なペナルティを避ける）

            overlap_mat[i, j] = depth
            overlap_mat[j, i] = depth

    return overlap_mat


@njit(cache=True, fastmath=True)
def compute_bounding_box(all_vertices: list[np.ndarray]) -> tuple[float, float, float, float]:
    min_x = 1e15
    min_y = 1e15
    max_x = -1e15
    max_y = -1e15
    for verts in all_vertices:
        x1, y1, x2, y2 = polygon_bounds(verts)
        min_x = min(x1, min_x)
        min_y = min(y1, min_y)
        max_x = max(x2, max_x)
        max_y = max(y2, max_y)
    return min_x, min_y, max_x, max_y


@njit(cache=True, fastmath=True)
def get_side_length(all_vertices: list[np.ndarray]) -> float:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True, fastmath=True)
def evaluate_gls_score(
    xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, weights: np.ndarray, alpha: float, beta: float
) -> float:
    """
    GLS評価関数: Score = Side^2 + alpha * TotalOverlap + beta * WeightedOverlap
    """
    n = len(xs)
    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]

    # 基本スコア
    side = get_side_length(all_vertices)
    base_score = side * side

    # 重なり行列
    overlap_mat = calculate_pairwise_overlap_matrix(all_vertices)
    total_overlap = np.sum(overlap_mat) * 0.5  # 対称なので半分に

    if total_overlap < 1e-9:
        return base_score / n

    # 重み付きペナルティ
    # weights[i, j] * overlap_mat[i, j]
    weighted_penalty = np.sum(overlap_mat * weights) * 0.5

    # ペナルティ項 (重なりがある場合は非常に大きな値をベースにする)
    penalty = alpha * total_overlap + beta * weighted_penalty

    # 重なりがある解は、重なりがない解よりも必ず悪くなるように大きなオフセットを足す
    # ただしペナルティ学習中は「少しでも重なりが少ない方」を選ばせるため、オフセットは控えめに
    return (base_score / n) + 1000.0 + penalty


@njit(cache=True, fastmath=True)
def evaluate_group_score(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> float:
    """
    純粋なスコア評価（重なりがあれば 1e9）
    """
    n = len(xs)
    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]

    # 簡易重なりチェック (面積ベース)
    overlap_mat = calculate_pairwise_overlap_matrix(all_vertices)
    if np.sum(overlap_mat) > 1e-9:
        return 1e9

    side = get_side_length(all_vertices)
    return side * side / n


# -----------------------------------------------------------------------------
# SSA with GLS
# -----------------------------------------------------------------------------
def wrap_to_180(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def to_abs_solution(
    base_cx: float,
    base_cy: float,
    base_degs: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    doff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = base_cx + dx
    ys = base_cy + dy
    degs = (base_degs + doff) % 360.0
    return xs, ys, degs


def _normalize_candidate(
    dx: np.ndarray, dy: np.ndarray, doff: np.ndarray, pos_bound: float, ang_bound: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = dx - float(dx.mean())
    dy = dy - float(dy.mean())
    dx = np.clip(dx, -pos_bound, pos_bound)
    dy = np.clip(dy, -pos_bound, pos_bound)
    doff = np.clip(doff, -ang_bound, ang_bound)
    return dx, dy, doff


def _finalize_candidate_gls(
    base_cx, base_cy, base_degs, cand_dx, cand_dy, cand_doff, pos_bound, ang_bound, weights, alpha, beta
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dx, dy, doff = _normalize_candidate(cand_dx, cand_dy, cand_doff, pos_bound, ang_bound)
    xs, ys, degs = to_abs_solution(base_cx, base_cy, base_degs, dx, dy, doff)
    s = evaluate_gls_score(xs, ys, degs, weights, alpha, beta)
    return dx, dy, doff, s


def _ssa_initialize_population_gls(
    rng,
    base_cx,
    base_cy,
    base_degs,
    base_dx0,
    base_dy0,
    base_score,
    pop_size,
    init_pos_delta,
    init_ang_delta,
    init_tries,
    pos_bound,
    ang_bound,
    weights,
    alpha,
    beta,
):
    n = len(base_dx0)
    pop_dx = np.zeros((pop_size, n))
    pop_dy = np.zeros((pop_size, n))
    pop_doff = np.zeros((pop_size, n))
    fitness = np.zeros(pop_size)

    # 0番目はベースライン
    pop_dx[0] = base_dx0
    pop_dy[0] = base_dy0
    pop_doff[0] = 0.0
    # 初期スコアもGLS基準で計算
    xs, ys, degs = to_abs_solution(base_cx, base_cy, base_degs, base_dx0, base_dy0, np.zeros(n))
    fitness[0] = evaluate_gls_score(xs, ys, degs, weights, alpha, beta)

    for i in range(1, pop_size):
        cur_pos = float(init_pos_delta)
        cur_ang = float(init_ang_delta)
        best_local_dx = base_dx0.copy()
        best_local_dy = base_dy0.copy()
        best_local_doff = np.zeros(n)
        best_local_score = 1e15

        for _ in range(max(1, init_tries)):
            dx = base_dx0 + rng.uniform(-cur_pos, cur_pos, size=n)
            dy = base_dy0 + rng.uniform(-cur_pos, cur_pos, size=n)
            doff = rng.uniform(-cur_ang, cur_ang, size=n)
            dx, dy, doff, s = _finalize_candidate_gls(
                base_cx, base_cy, base_degs, dx, dy, doff, pos_bound, ang_bound, weights, alpha, beta
            )

            if s < best_local_score:
                best_local_dx, best_local_dy, best_local_doff, best_local_score = dx, dy, doff, s

            cur_pos *= 0.8
            cur_ang *= 0.8

        pop_dx[i] = best_local_dx
        pop_dy[i] = best_local_dy
        pop_doff[i] = best_local_doff
        fitness[i] = best_local_score

    order = np.argsort(fitness)
    return pop_dx[order], pop_dy[order], pop_doff[order], fitness[order]


def optimize_group_ssa_gls(  # noqa: PLR0915
    base_xs: np.ndarray,
    base_ys: np.ndarray,
    base_degs: np.ndarray,
    pop_size: int,
    n_iters: int,
    pd_ratio: float,
    sd_ratio: float,
    st: float,
    danger_beta: float,
    init_pos_delta: float,
    init_ang_delta: float,
    init_tries: int,
    pos_delta: float,
    ang_delta: float,
    pos_bound_scale: float,
    ang_bound: float,
    repair_steps: int,  # 未使用だが互換性のため残す
    seed: int,
    debug: bool,
    # GLS Params
    gls_alpha: float = 1000.0,
    gls_beta: float = 100.0,
    gls_interval: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(base_xs)

    base_cx = float(base_xs.mean())
    base_cy = float(base_ys.mean())
    base_dx0 = base_xs - base_cx
    base_dy0 = base_ys - base_cy

    # 探索境界
    xs0, ys0, degs0 = to_abs_solution(base_cx, base_cy, base_degs, base_dx0, base_dy0, np.zeros_like(base_degs))

    # ペナルティ重み行列の初期化
    weights = np.zeros((n, n), dtype=np.float64)

    # 境界推定用の純粋スコア
    pure_side = get_side_length([get_tree_vertices(xs0[i], ys0[i], degs0[i]) for i in range(n)])
    pos_bound = float(max(1e-6, pure_side * pos_bound_scale))

    # GLS基準で初期化
    base_gls_score = evaluate_gls_score(xs0, ys0, degs0, weights, gls_alpha, gls_beta)
    pop_dx, pop_dy, pop_doff, fitness = _ssa_initialize_population_gls(
        rng,
        base_cx,
        base_cy,
        base_degs,
        base_dx0,
        base_dy0,
        base_gls_score,
        pop_size,
        init_pos_delta,
        init_ang_delta,
        init_tries,
        pos_bound,
        ang_bound,
        weights,
        gls_alpha,
        gls_beta,
    )

    # Global Best (Validな中で最小のもの)
    global_best_score = 1e15
    global_best_xs = xs0.copy()
    global_best_ys = ys0.copy()
    global_best_degs = degs0.copy()

    # 初期解のチェック (Validなら保存)
    init_overlap = np.sum(
        calculate_pairwise_overlap_matrix([get_tree_vertices(xs0[i], ys0[i], degs0[i]) for i in range(n)])
    )
    if init_overlap < 1e-9:
        t_side = get_side_length([get_tree_vertices(xs0[i], ys0[i], degs0[i]) for i in range(n)])
        global_best_score = t_side * t_side / n

    best_dx = pop_dx[0].copy()
    best_dy = pop_dy[0].copy()
    best_doff = pop_doff[0].copy()
    best_fit = fitness[0]

    pd_count = max(1, int(round(pop_size * pd_ratio)))
    sd_count = max(1, int(round(pop_size * sd_ratio)))

    for it in range(n_iters):
        # --- GLS Weight Update (Penalty Learning) ---
        if it > 0 and it % gls_interval == 0:
            # 現在のベスト解における重なりを取得
            bx, by, bd = to_abs_solution(base_cx, base_cy, base_degs, best_dx, best_dy, best_doff)
            verts = [get_tree_vertices(bx[i], by[i], bd[i]) for i in range(n)]
            overlaps = calculate_pairwise_overlap_matrix(verts)

            # 重なっているペアの重みを加算
            weights += (overlaps > 1e-6).astype(np.float64)

            # 全個体のフィットネスを新重みで再計算
            for i in range(pop_size):
                tx, ty, td = to_abs_solution(base_cx, base_cy, base_degs, pop_dx[i], pop_dy[i], pop_doff[i])
                fitness[i] = evaluate_gls_score(tx, ty, td, weights, gls_alpha, gls_beta)

            # ベスト情報の更新
            best_idx = np.argmin(fitness)
            best_fit = fitness[best_idx]
            best_dx = pop_dx[best_idx].copy()
            best_dy = pop_dy[best_idx].copy()
            best_doff = pop_doff[best_idx].copy()

        # --- SSA Optimization Steps ---
        order = np.argsort(fitness)
        pop_dx = pop_dx[order]
        pop_dy = pop_dy[order]
        pop_doff = pop_doff[order]
        fitness = fitness[order]

        cur_best_dx = pop_dx[0].copy()
        cur_best_dy = pop_dy[0].copy()
        cur_best_doff = pop_doff[0].copy()
        cur_best_fit = fitness[0]

        worst_dx = pop_dx[-1]
        worst_dy = pop_dy[-1]
        worst_doff = pop_doff[-1]
        worst_fit = fitness[-1]

        if cur_best_fit < best_fit:
            best_fit = cur_best_fit
            best_dx, best_dy, best_doff = cur_best_dx, cur_best_dy, cur_best_doff

        # --- Valid Check & Save ---
        # 探索解が Valid (重なりなし) か確認し、過去最高なら保存
        tx, ty, td = to_abs_solution(base_cx, base_cy, base_degs, cur_best_dx, cur_best_dy, cur_best_doff)
        t_verts = [get_tree_vertices(tx[i], ty[i], td[i]) for i in range(n)]

        # 高速化のため、まずGLSスコアから推測（GLSスコアが非常に高いなら重なりがあるはず）
        # ただし厳密性を期すためチェックする
        t_overlap_mat = calculate_pairwise_overlap_matrix(t_verts)
        t_overlap = np.sum(t_overlap_mat)

        if t_overlap < 1e-9:
            t_side = get_side_length(t_verts)
            t_score = t_side * t_side / n
            if t_score < global_best_score:
                global_best_score = t_score
                global_best_xs[:] = tx[:]
                global_best_ys[:] = ty[:]
                global_best_degs[:] = td[:]

        progress = it / max(1, n_iters)
        mix = 0.6 * (1.0 - progress)
        noise_pos = float(pos_delta) * (1.0 - progress)
        noise_ang = float(ang_delta) * (1.0 - progress)

        new_dx = pop_dx.copy()
        new_dy = pop_dy.copy()
        new_doff = pop_doff.copy()
        new_fit = fitness.copy()

        # SSA Logic with GLS finalizer
        # Producers
        for i in range(pd_count):
            r2 = rng.random()
            if r2 < st:
                shrink = math.exp(-(i + 1) / (rng.random() * max(1.0, float(n_iters)) + 1e-12))
                cand_dx = pop_dx[i] * shrink
                cand_dy = pop_dy[i] * shrink
                cand_doff = pop_doff[i] * shrink
            else:
                cand_dx = pop_dx[i] + rng.normal(0.0, max(1e-12, noise_pos), size=n)
                cand_dy = pop_dy[i] + rng.normal(0.0, max(1e-12, noise_pos), size=n)
                cand_doff = pop_doff[i] + rng.normal(0.0, max(1e-12, noise_ang), size=n)

            cand_dx = pop_dx[i] + mix * (cand_dx - pop_dx[i])
            cand_dy = pop_dy[i] + mix * (cand_dy - pop_dy[i])
            cand_doff = pop_doff[i] + mix * (cand_doff - pop_doff[i])

            new_dx[i], new_dy[i], new_doff[i], new_fit[i] = _finalize_candidate_gls(
                base_cx,
                base_cy,
                base_degs,
                cand_dx,
                cand_dy,
                cand_doff,
                pos_bound,
                ang_bound,
                weights,
                gls_alpha,
                gls_beta,
            )

        # Scroungers
        half = pop_size // 2
        signs = np.array([-1.0, 1.0])
        for i in range(pd_count, pop_size):
            if i > half:
                q = rng.normal()
                denom = (i + 1) ** 2
                cand_dx = q * np.exp((worst_dx - pop_dx[i]) / denom)
                cand_dy = q * np.exp((worst_dy - pop_dy[i]) / denom)
                cand_doff = q * np.exp((worst_doff - pop_doff[i]) / denom)
            else:
                a = rng.choice(signs, size=n)
                cand_dx = cur_best_dx + np.abs(pop_dx[i] - cur_best_dx) * a
                a = rng.choice(signs, size=n)
                cand_dy = cur_best_dy + np.abs(pop_dy[i] - cur_best_dy) * a
                a = rng.choice(signs, size=n)
                cand_doff = cur_best_doff + np.abs(pop_doff[i] - cur_best_doff) * a

            cand_dx = pop_dx[i] + mix * (cand_dx - pop_dx[i]) + rng.normal(0.0, max(1e-12, noise_pos * 0.1), size=n)
            cand_dy = pop_dy[i] + mix * (cand_dy - pop_dy[i]) + rng.normal(0.0, max(1e-12, noise_pos * 0.1), size=n)
            cand_doff = (
                pop_doff[i] + mix * (cand_doff - pop_doff[i]) + rng.normal(0.0, max(1e-12, noise_ang * 0.1), size=n)
            )

            new_dx[i], new_dy[i], new_doff[i], new_fit[i] = _finalize_candidate_gls(
                base_cx,
                base_cy,
                base_degs,
                cand_dx,
                cand_dy,
                cand_doff,
                pos_bound,
                ang_bound,
                weights,
                gls_alpha,
                gls_beta,
            )

        # Aware
        danger_idx = rng.choice(pop_size, size=min(sd_count, pop_size), replace=False)
        for i in danger_idx:
            if i == 0:
                continue
            k = rng.uniform(-1.0, 1.0, size=n)
            if new_fit[i] > cur_best_fit:
                cand_dx = cur_best_dx + danger_beta * np.abs(new_dx[i] - cur_best_dx) * k
                cand_dy = cur_best_dy + danger_beta * np.abs(new_dy[i] - cur_best_dy) * k
                cand_doff = cur_best_doff + danger_beta * np.abs(new_doff[i] - cur_best_doff) * k
            else:
                denom = abs(new_fit[i] - worst_fit) + 1e-12
                cand_dx = new_dx[i] + k * (np.abs(new_dx[i] - worst_dx) / denom)
                cand_dy = new_dy[i] + k * (np.abs(new_dy[i] - worst_dy) / denom)
                cand_doff = new_doff[i] + k * (np.abs(new_doff[i] - worst_doff) / denom)

            cand_dx = new_dx[i] + (mix * 0.5) * (cand_dx - new_dx[i])
            cand_dy = new_dy[i] + (mix * 0.5) * (cand_dy - new_dy[i])
            cand_doff = new_doff[i] + (mix * 0.5) * (cand_doff - new_doff[i])

            new_dx[i], new_dy[i], new_doff[i], new_fit[i] = _finalize_candidate_gls(
                base_cx,
                base_cy,
                base_degs,
                cand_dx,
                cand_dy,
                cand_doff,
                pos_bound,
                ang_bound,
                weights,
                gls_alpha,
                gls_beta,
            )

        pop_dx, pop_dy, pop_doff, fitness = new_dx, new_dy, new_doff, new_fit

        if debug and it % 100 == 0:
            print(f"  iter={it:4d} best_GLS={best_fit:.4f} valid_best={global_best_score:.4f}")

    return global_best_xs, global_best_ys, global_best_degs, global_best_score


# -----------------------------------------------------------------------------
# IO / Scoring
# -----------------------------------------------------------------------------
def load_submission_data(filepath: str, fallback_path: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_path = filepath
    if not os.path.exists(target_path):
        if fallback_path is None or not os.path.exists(fallback_path):
            return np.array([]), np.array([]), np.array([])
        target_path = fallback_path

    df = pd.read_csv(target_path)
    all_xs, all_ys, all_degs = [], [], []
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")
        for _, row in group.iterrows():
            x = float(row["x"].replace("s", "")) if isinstance(row["x"], str) else float(row["x"])
            y = float(row["y"].replace("s", "")) if isinstance(row["y"], str) else float(row["y"])
            deg = float(row["deg"].replace("s", "")) if isinstance(row["deg"], str) else float(row["deg"])
            all_xs.append(x)
            all_ys.append(y)
            all_degs.append(deg)
    return np.array(all_xs), np.array(all_ys), np.array(all_degs)


def save_submission(filepath: str, all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray) -> None:
    rows = []
    idx = 0
    for n in range(1, 201):
        for t in range(n):
            rows.append(
                {
                    "id": f"{n:03d}_{t}",
                    "x": f"s{all_xs[idx]}",
                    "y": f"s{all_ys[idx]}",
                    "deg": f"s{all_degs[idx]}",
                }
            )
            idx += 1
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(rows).to_csv(filepath, index=False)


def calculate_total_score(all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray) -> float:
    total = 0.0
    for n in range(1, 201):
        start = n * (n - 1) // 2
        xs = all_xs[start : start + n]
        ys = all_ys[start : start + n]
        degs = all_degs[start : start + n]
        total += float(evaluate_group_score(xs, ys, degs))
    return total


def find_best_submission(
    submissions_dir: str = "submissions",
) -> tuple[str, float, np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
    """
    submissionsディレクトリ内のすべてのCSVファイルをスキャンして、最もスコアが高いものを返す
    """
    submissions_path = Path(submissions_dir)
    if not submissions_path.exists():
        return None

    csv_files = list(submissions_path.glob("*.csv"))
    if not csv_files:
        return None

    best_filepath = None
    best_score = float("inf")
    best_xs = None
    best_ys = None
    best_degs = None

    print(f"\n過去のsubmissionファイルをスキャン中... ({len(csv_files)}ファイル)")
    for csv_file in csv_files:
        try:
            xs, ys, degs = load_submission_data(str(csv_file))
            if len(xs) == 0:
                continue
            score = calculate_total_score(xs, ys, degs)

            # スコアが改善されたかチェック（1e15以上のペナルティスコアは除外）
            if score < 1e14 and score < best_score:
                best_score = score
                best_filepath = str(csv_file)
                best_xs = xs
                best_ys = ys
                best_degs = degs
            print(f"  {csv_file.name}: {score:.6f}")
        except Exception as e:
            print(f"  {csv_file.name}: 読み込みエラー ({e})")
            continue

    if best_filepath is not None:
        print(f"\n最良submission: {best_filepath} (スコア: {best_score:.6f})")
        return best_filepath, best_score, best_xs, best_ys, best_degs

    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("SSA + Penalty Learning (GLS) - Correct Scale & Intersection Check")

    paths_cfg = CONFIG.get("paths", {})
    baseline_path = paths_cfg.get("baseline", "submissions/submission.csv")
    fallback_path = paths_cfg.get("baseline_fallback")
    output_path = paths_cfg.get("output", "submissions/submission_ssa_gls.csv")

    all_xs, all_ys, all_degs = load_submission_data(baseline_path, fallback_path)
    baseline_total = calculate_total_score(all_xs, all_ys, all_degs)
    print(f"ベースライン合計スコア: {baseline_total:.6f}")

    opt_cfg = CONFIG.get("optimization", {})
    n_min = int(opt_cfg.get("n_min", 1))
    n_max = int(opt_cfg.get("n_max", 200))
    target_groups = opt_cfg.get("target_groups", [])
    target_top_k = int(opt_cfg.get("target_top_k", 0))
    target_range = opt_cfg.get("target_range")

    ssa_cfg = opt_cfg.get("ssa", {})
    pop_size = int(ssa_cfg.get("pop_size", 24))
    n_iters = int(ssa_cfg.get("n_iters", 1000))
    pd_ratio = float(ssa_cfg.get("pd_ratio", 0.2))
    sd_ratio = float(ssa_cfg.get("sd_ratio", 0.1))
    st = float(ssa_cfg.get("st", 0.8))
    danger_beta = float(ssa_cfg.get("danger_beta", 1.0))

    init_pos_delta = float(ssa_cfg.get("init_pos_delta", 0.05))
    init_ang_delta = float(ssa_cfg.get("init_ang_delta", 8.0))
    init_tries = int(ssa_cfg.get("init_tries", 30))

    pos_delta = float(ssa_cfg.get("pos_delta", 0.03))
    ang_delta = float(ssa_cfg.get("ang_delta", 6.0))
    pos_bound_scale = float(ssa_cfg.get("pos_bound_scale", 2.0))
    ang_bound = float(ssa_cfg.get("ang_bound", 45.0))
    repair_steps = int(ssa_cfg.get("repair_steps", 12))
    seed_base = int(ssa_cfg.get("seed_base", 2042))
    debug = bool(ssa_cfg.get("debug", False))

    gls_alpha = float(ssa_cfg.get("gls_alpha", 1000.0))
    gls_beta = float(ssa_cfg.get("gls_beta", 100.0))
    gls_interval = int(ssa_cfg.get("gls_interval", 50))

    range_min = n_min
    range_max = n_max
    if isinstance(target_range, list | tuple) and len(target_range) == 2:
        range_min = max(n_min, int(target_range[0]))
        range_max = min(n_max, int(target_range[1]))

    target_group_set = None
    if target_groups:
        target_groups = sorted({int(n) for n in target_groups if range_min <= int(n) <= range_max})
        target_group_set = set(target_groups)
        print(f"対象グループ数: {len(target_groups)}")
    elif target_top_k > 0:
        scores = []
        for n in range(range_min, range_max + 1):
            if n < 2:
                continue
            start = n * (n - 1) // 2
            xs = all_xs[start : start + n]
            ys = all_ys[start : start + n]
            degs = all_degs[start : start + n]
            s = float(evaluate_group_score(xs, ys, degs))
            scores.append((s, n))
        scores.sort(reverse=True, key=lambda x: x[0])
        k = min(target_top_k, len(scores))
        target_groups = sorted([n for _, n in scores[:k]])
        target_group_set = set(target_groups)
        print(f"対象グループ数: {len(target_groups)} (top_k={k}, 範囲={range_min}-{range_max})")

    # 無限ループ設定
    loop_cfg = opt_cfg.get("infinite_loop", {})
    enable_infinite_loop = bool(loop_cfg.get("enable", True))  # デフォルト有効に変更
    max_time_seconds = float(loop_cfg.get("max_time_seconds", 3600.0))
    max_attempts = int(loop_cfg.get("max_attempts", 10))
    min_improvement_threshold = float(loop_cfg.get("min_improvement_threshold", 1e-6))
    random_restart_prob = float(loop_cfg.get("random_restart_prob", 0.3))

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    best_ever_score = baseline_total
    best_ever_xs = new_xs.copy()
    best_ever_ys = new_ys.copy()
    best_ever_degs = new_degs.copy()

    attempt = 0
    start_time = time.time()
    rng_main = np.random.default_rng(seed_base + 99999)

    while True:
        attempt += 1
        print(f"\n{'=' * 60}")
        print(f"試行 {attempt}/{max_attempts if enable_infinite_loop else 1}")
        if enable_infinite_loop:
            elapsed = time.time() - start_time
            print(f"経過時間: {elapsed:.1f}秒 / {max_time_seconds:.1f}秒")

        improved_groups = 0
        total_improved = 0.0

        # 初期解リセット
        if enable_infinite_loop and attempt > 1:
            if rng_main.random() < random_restart_prob:
                print("  ランダムリスタート: ベースラインから再開")
                current_xs = all_xs.copy()
                current_ys = all_ys.copy()
                current_degs = all_degs.copy()
            else:
                current_xs = best_ever_xs.copy()
                current_ys = best_ever_ys.copy()
                current_degs = best_ever_degs.copy()
        else:
            current_xs = new_xs.copy()
            current_ys = new_ys.copy()
            current_degs = new_degs.copy()

        for n in tqdm(range(range_min, range_max + 1), desc="最適化"):
            if target_group_set is not None and n not in target_group_set:
                continue
            if n < 2:
                continue

            start = n * (n - 1) // 2
            xs = current_xs[start : start + n].copy()
            ys = current_ys[start : start + n].copy()
            degs = current_degs[start : start + n].copy()

            orig_score = float(evaluate_group_score(xs, ys, degs))

            seed = seed_base + n * 1000 + attempt * 10000

            opt_xs, opt_ys, opt_degs, opt_score = optimize_group_ssa_gls(
                xs,
                ys,
                degs,
                pop_size,
                n_iters,
                pd_ratio,
                sd_ratio,
                st,
                danger_beta,
                init_pos_delta,
                init_ang_delta,
                init_tries,
                pos_delta,
                ang_delta,
                pos_bound_scale,
                ang_bound,
                repair_steps,
                seed,
                debug,
                gls_alpha,
                gls_beta,
                gls_interval,
            )

            if opt_score < orig_score - 1e-9:
                improvement = orig_score - opt_score
                improved_groups += 1
                total_improved += improvement
                current_xs[start : start + n] = opt_xs
                current_ys[start : start + n] = opt_ys
                current_degs[start : start + n] = opt_degs
                print(f"  グループ{n:03d}: {orig_score:.6f} -> {opt_score:.6f} (改善 {improvement:.6f})")

        final_score = calculate_total_score(current_xs, current_ys, current_degs)
        print(f"\n試行 {attempt} 結果:")
        print(f"  最適化後: {final_score:.6f}")
        print(f"  総改善量: {baseline_total - final_score:+.6f}")
        print(f"  改善グループ数: {improved_groups}")

        # 最良解更新
        if final_score < best_ever_score - min_improvement_threshold:
            improvement_amount = best_ever_score - final_score
            best_ever_score = final_score
            best_ever_xs = current_xs.copy()
            best_ever_ys = current_ys.copy()
            best_ever_degs = current_degs.copy()
            new_xs = current_xs.copy()
            new_ys = current_ys.copy()
            new_degs = current_degs.copy()
            print(f"  ✓ 最良解を更新: {improvement_amount:.6f} 改善")
        else:
            print(f"  × 改善なし（現在の最良: {best_ever_score:.6f}）")

        if not enable_infinite_loop:
            break
        if time.time() - start_time >= max_time_seconds:
            break
        if attempt >= max_attempts:
            break

    # 最終結果
    final_score = best_ever_score
    final_xs = best_ever_xs
    final_ys = best_ever_ys
    final_degs = best_ever_degs

    print(f"\n{'=' * 60}\n最終結果\n  最適化後: {final_score:.6f}")

    baseline_improved = final_score < baseline_total - 1e-9

    # 過去のsubmissionと比較・採用
    if os.path.exists(output_path):
        ref_xs, ref_ys, ref_degs = load_submission_data(output_path)
        submission_score = calculate_total_score(ref_xs, ref_ys, ref_degs)
        print(f"  既存提出スコア: {submission_score:.6f}")
    else:
        submission_score = None

    if not baseline_improved or (submission_score is not None and final_score >= submission_score - 1e-9):
        best_sub = find_best_submission(os.path.dirname(output_path))
        if best_sub and best_sub[1] < final_score - 1e-9:
            print(f"\n過去の最良submissionを使用: {best_sub[0]} (スコア: {best_sub[1]:.6f})")
            final_xs, final_ys, final_degs = best_sub[2], best_sub[3], best_sub[4]
            final_score = best_sub[1]

    if final_xs is not None and final_ys is not None and final_degs is not None:
        save_submission(output_path, final_xs, final_ys, final_degs)
        print(f"\nsubmissionを更新しました: {output_path} (スコア: {final_score:.6f})")
    else:
        print("\n有効な解が見つからなかったため、submissionは更新されませんでした。")
