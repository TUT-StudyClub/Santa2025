"""
exp021_ssa: Sparrow Search Algorithm (SSA / スズメ探索) による局所探索

各グループの (x, y, deg) を SSA で探索して、スコア（side^2 / n）を改善する。
ベースラインからの摂動を前提に、重なりが出た個体は「最良個体への線形補間」で修復する。
"""

from __future__ import annotations

import math
import os
import sys

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
    return os.path.join("experiments", "exp021_ssa", "exp", f"{config_name}.yaml")


CONFIG_PATH = resolve_config_path()
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

TREE_CFG = CONFIG["tree_shape"]
TRUNK_W = float(TREE_CFG["trunk_w"])
BASE_W = float(TREE_CFG["base_w"])
MID_W = float(TREE_CFG["mid_w"])
TOP_W = float(TREE_CFG["top_w"])
TIP_Y = float(TREE_CFG["tip_y"])
TIER_1_Y = float(TREE_CFG["tier_1_y"])
TIER_2_Y = float(TREE_CFG["tier_2_y"])
BASE_Y = float(TREE_CFG["base_y"])
TRUNK_BOTTOM_Y = float(TREE_CFG["trunk_bottom_y"])


# -----------------------------------------------------------------------------
# Geometry Utils (Numba)
# -----------------------------------------------------------------------------
@njit
def rotate_point(x: float, y: float, cos_a: float, sin_a: float) -> tuple[float, float]:
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


@njit
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


@njit
def polygon_bounds(vertices: np.ndarray) -> tuple[float, float, float, float]:
    min_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_x = vertices[0, 0]
    max_y = vertices[0, 1]
    for i in range(1, vertices.shape[0]):
        x = vertices[i, 0]
        y = vertices[i, 1]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y


@njit
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


@njit
def point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float) -> bool:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    cross = wx * vy - wy * vx
    if abs(cross) > eps:
        return False
    dot = wx * vx + wy * vy
    if dot < -eps:
        return False
    sq_len = vx * vx + vy * vy
    if dot - sq_len > eps:
        return False
    return True


@njit
def point_in_polygon_strict(px: float, py: float, vertices: np.ndarray) -> bool:
    # 境界上の点を「外側」として扱う（touch を overlap と誤判定しないため）
    eps = 1e-10
    n = vertices.shape[0]
    for i in range(n):
        j = (i + 1) % n
        if point_on_segment(px, py, vertices[i, 0], vertices[i, 1], vertices[j, 0], vertices[j, 1], eps):
            return False
    return point_in_polygon(px, py, vertices)


@njit
def segments_intersect_strict(  # noqa: PLR0913
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    p3x: float,
    p3y: float,
    p4x: float,
    p4y: float,
) -> bool:
    d1x = p2x - p1x
    d1y = p2y - p1y
    d2x = p4x - p3x
    d2y = p4y - p3y
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-10:  # noqa: PLR2004
        # 平行（ほぼ平行）な場合は「面積のある重なり」判定が難しいため、ここでは交差なし扱い
        return False
    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / det
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / det
    # 数値誤差で「端点接触」が交差扱いになりやすいので、少し大きめに取る
    eps = 1e-8
    return eps < t < 1.0 - eps and eps < u < 1.0 - eps


@njit
def polygons_overlap_area(verts1: np.ndarray, verts2: np.ndarray) -> bool:
    min_x1, min_y1, max_x1, max_y1 = polygon_bounds(verts1)
    min_x2, min_y2, max_x2, max_y2 = polygon_bounds(verts2)

    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False

    for i in range(verts1.shape[0]):
        if point_in_polygon_strict(verts1[i, 0], verts1[i, 1], verts2):
            return True
    for i in range(verts2.shape[0]):
        if point_in_polygon_strict(verts2[i, 0], verts2[i, 1], verts1):
            return True

    n1 = verts1.shape[0]
    n2 = verts2.shape[0]
    for i in range(n1):
        j = (i + 1) % n1
        p1x, p1y = verts1[i, 0], verts1[i, 1]
        p2x, p2y = verts1[j, 0], verts1[j, 1]
        for k in range(n2):
            m = (k + 1) % n2
            p3x, p3y = verts2[k, 0], verts2[k, 1]
            p4x, p4y = verts2[m, 0], verts2[m, 1]
            if segments_intersect_strict(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
                return True
    return False


@njit
def compute_bounding_box(all_vertices: list[np.ndarray]) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for verts in all_vertices:
        x1, y1, x2, y2 = polygon_bounds(verts)
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    return min_x, min_y, max_x, max_y


@njit
def get_side_length(all_vertices: list[np.ndarray]) -> float:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)


@njit
def evaluate_group_score(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> float:
    """
    重なりがあれば大きな罰点を返す。
    """
    n = len(xs)
    if n <= 1:
        verts = [get_tree_vertices(xs[0], ys[0], degs[0])]
        side = get_side_length(verts)
        return side * side

    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_area(all_vertices[i], all_vertices[j]):
                return 1e9
    side = get_side_length(all_vertices)
    return side * side / n


# -----------------------------------------------------------------------------
# SSA
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


def is_feasible(score: float) -> bool:
    return score < 1e8


def repair_towards_best(
    base_cx: float,
    base_cy: float,
    base_degs: np.ndarray,
    cand_dx: np.ndarray,
    cand_dy: np.ndarray,
    cand_doff: np.ndarray,
    best_dx: np.ndarray,
    best_dy: np.ndarray,
    best_doff: np.ndarray,
    pos_bound: float,
    ang_bound: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    best（可行）と cand（不可行の可能性あり）を線形補間して、可行解を探す。
    """
    dx = np.clip(cand_dx, -pos_bound, pos_bound)
    dy = np.clip(cand_dy, -pos_bound, pos_bound)
    doff = np.clip(cand_doff, -ang_bound, ang_bound)
    xs, ys, degs = to_abs_solution(base_cx, base_cy, base_degs, dx, dy, doff)
    score = float(evaluate_group_score(xs, ys, degs))
    if is_feasible(score):
        return dx, dy, doff, score

    lo = 0.0
    hi = 1.0
    best_score = None
    for _ in range(max(1, steps)):
        mid = (lo + hi) * 0.5
        mdx = best_dx + (dx - best_dx) * mid
        mdy = best_dy + (dy - best_dy) * mid
        mdoff = best_doff + (doff - best_doff) * mid
        mdx = np.clip(mdx, -pos_bound, pos_bound)
        mdy = np.clip(mdy, -pos_bound, pos_bound)
        mdoff = np.clip(mdoff, -ang_bound, ang_bound)
        xs, ys, degs = to_abs_solution(base_cx, base_cy, base_degs, mdx, mdy, mdoff)
        s = float(evaluate_group_score(xs, ys, degs))
        if is_feasible(s):
            lo = mid
            best_score = s
        else:
            hi = mid

    if best_score is None:
        xs, ys, degs = to_abs_solution(base_cx, base_cy, base_degs, best_dx, best_dy, best_doff)
        s = float(evaluate_group_score(xs, ys, degs))
        return best_dx.copy(), best_dy.copy(), best_doff.copy(), s

    fx = best_dx + (dx - best_dx) * lo
    fy = best_dy + (dy - best_dy) * lo
    fd = best_doff + (doff - best_doff) * lo
    return fx, fy, fd, float(best_score)


def _normalize_candidate(
    dx: np.ndarray,
    dy: np.ndarray,
    doff: np.ndarray,
    *,
    pos_bound: float,
    ang_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = dx - float(dx.mean())
    dy = dy - float(dy.mean())
    dx = np.clip(dx, -pos_bound, pos_bound)
    dy = np.clip(dy, -pos_bound, pos_bound)
    doff = np.clip(doff, -ang_bound, ang_bound)
    return dx, dy, doff


def _finalize_candidate(
    *,
    base_cx: float,
    base_cy: float,
    base_degs: np.ndarray,
    cand_dx: np.ndarray,
    cand_dy: np.ndarray,
    cand_doff: np.ndarray,
    best_dx: np.ndarray,
    best_dy: np.ndarray,
    best_doff: np.ndarray,
    pos_bound: float,
    ang_bound: float,
    repair_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cand_dx, cand_dy, cand_doff = _normalize_candidate(
        cand_dx,
        cand_dy,
        cand_doff,
        pos_bound=pos_bound,
        ang_bound=ang_bound,
    )
    return repair_towards_best(
        base_cx,
        base_cy,
        base_degs,
        cand_dx,
        cand_dy,
        cand_doff,
        best_dx,
        best_dy,
        best_doff,
        pos_bound,
        ang_bound,
        repair_steps,
    )


def _ssa_initialize_population(
    *,
    rng: np.random.Generator,
    base_cx: float,
    base_cy: float,
    base_degs: np.ndarray,
    base_dx0: np.ndarray,
    base_dy0: np.ndarray,
    base_score: float,
    pop_size: int,
    init_pos_delta: float,
    init_ang_delta: float,
    init_tries: int,
    pos_bound: float,
    ang_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(base_dx0)
    pop_dx = np.zeros((pop_size, n), dtype=np.float64)
    pop_dy = np.zeros((pop_size, n), dtype=np.float64)
    pop_doff = np.zeros((pop_size, n), dtype=np.float64)
    fitness = np.zeros(pop_size, dtype=np.float64)

    pop_dx[0] = base_dx0
    pop_dy[0] = base_dy0
    pop_doff[0] = 0.0
    fitness[0] = base_score

    for i in range(1, pop_size):
        cur_pos = float(init_pos_delta)
        cur_ang = float(init_ang_delta)

        best_local_dx = base_dx0.copy()
        best_local_dy = base_dy0.copy()
        best_local_doff = np.zeros(n, dtype=np.float64)
        best_local_score = float(base_score)

        for _ in range(max(1, init_tries)):
            dx = base_dx0 + rng.uniform(-cur_pos, cur_pos, size=n)
            dy = base_dy0 + rng.uniform(-cur_pos, cur_pos, size=n)
            doff = rng.uniform(-cur_ang, cur_ang, size=n)
            dx, dy, doff = _normalize_candidate(dx, dy, doff, pos_bound=pos_bound, ang_bound=ang_bound)

            xs, ys, degs = to_abs_solution(base_cx, base_cy, base_degs, dx, dy, doff)
            s = float(evaluate_group_score(xs, ys, degs))
            if is_feasible(s) and s < best_local_score:
                best_local_dx, best_local_dy, best_local_doff, best_local_score = dx, dy, doff, s
                break

            cur_pos *= 0.7
            cur_ang *= 0.7

        pop_dx[i] = best_local_dx
        pop_dy[i] = best_local_dy
        pop_doff[i] = best_local_doff
        fitness[i] = best_local_score

    order = np.argsort(fitness)
    return pop_dx[order], pop_dy[order], pop_doff[order], fitness[order]


def _ssa_update_producers(
    *,
    rng: np.random.Generator,
    pop_dx: np.ndarray,
    pop_dy: np.ndarray,
    pop_doff: np.ndarray,
    new_dx: np.ndarray,
    new_dy: np.ndarray,
    new_doff: np.ndarray,
    new_fit: np.ndarray,
    pd_count: int,
    st: float,
    mix: float,
    noise_pos: float,
    noise_ang: float,
    base_cx: float,
    base_cy: float,
    base_degs: np.ndarray,
    best_dx: np.ndarray,
    best_dy: np.ndarray,
    best_doff: np.ndarray,
    pos_bound: float,
    ang_bound: float,
    repair_steps: int,
    n_iters: int,
) -> None:
    n = pop_dx.shape[1]
    for i in range(pd_count):
        r2 = float(rng.random())
        if r2 < st:
            shrink = math.exp(-(i + 1) / (float(rng.random()) * max(1.0, float(n_iters)) + 1e-12))
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

        cand_dx, cand_dy, cand_doff, s = _finalize_candidate(
            base_cx=base_cx,
            base_cy=base_cy,
            base_degs=base_degs,
            cand_dx=cand_dx,
            cand_dy=cand_dy,
            cand_doff=cand_doff,
            best_dx=best_dx,
            best_dy=best_dy,
            best_doff=best_doff,
            pos_bound=pos_bound,
            ang_bound=ang_bound,
            repair_steps=repair_steps,
        )
        new_dx[i] = cand_dx
        new_dy[i] = cand_dy
        new_doff[i] = cand_doff
        new_fit[i] = s


def _ssa_update_scroungers(
    *,
    rng: np.random.Generator,
    pop_dx: np.ndarray,
    pop_dy: np.ndarray,
    pop_doff: np.ndarray,
    new_dx: np.ndarray,
    new_dy: np.ndarray,
    new_doff: np.ndarray,
    new_fit: np.ndarray,
    pd_count: int,
    mix: float,
    noise_pos: float,
    noise_ang: float,
    base_cx: float,
    base_cy: float,
    base_degs: np.ndarray,
    best_dx: np.ndarray,
    best_dy: np.ndarray,
    best_doff: np.ndarray,
    worst_dx: np.ndarray,
    worst_dy: np.ndarray,
    worst_doff: np.ndarray,
    pos_bound: float,
    ang_bound: float,
    repair_steps: int,
) -> None:
    pop_size = pop_dx.shape[0]
    n = pop_dx.shape[1]
    half = pop_size // 2
    signs = np.array([-1.0, 1.0], dtype=np.float64)

    for i in range(pd_count, pop_size):
        if i > half:
            q = float(rng.normal())
            denom = float((i + 1) ** 2)
            cand_dx = q * np.exp((worst_dx - pop_dx[i]) / denom)
            cand_dy = q * np.exp((worst_dy - pop_dy[i]) / denom)
            cand_doff = q * np.exp((worst_doff - pop_doff[i]) / denom)
        else:
            a = rng.choice(signs, size=n)
            cand_dx = best_dx + np.abs(pop_dx[i] - best_dx) * a
            a = rng.choice(signs, size=n)
            cand_dy = best_dy + np.abs(pop_dy[i] - best_dy) * a
            a = rng.choice(signs, size=n)
            cand_doff = best_doff + np.abs(pop_doff[i] - best_doff) * a

        cand_dx = pop_dx[i] + mix * (cand_dx - pop_dx[i]) + rng.normal(0.0, max(1e-12, noise_pos * 0.1), size=n)
        cand_dy = pop_dy[i] + mix * (cand_dy - pop_dy[i]) + rng.normal(0.0, max(1e-12, noise_pos * 0.1), size=n)
        cand_doff = pop_doff[i] + mix * (cand_doff - pop_doff[i]) + rng.normal(0.0, max(1e-12, noise_ang * 0.1), size=n)

        cand_dx, cand_dy, cand_doff, s = _finalize_candidate(
            base_cx=base_cx,
            base_cy=base_cy,
            base_degs=base_degs,
            cand_dx=cand_dx,
            cand_dy=cand_dy,
            cand_doff=cand_doff,
            best_dx=best_dx,
            best_dy=best_dy,
            best_doff=best_doff,
            pos_bound=pos_bound,
            ang_bound=ang_bound,
            repair_steps=repair_steps,
        )
        new_dx[i] = cand_dx
        new_dy[i] = cand_dy
        new_doff[i] = cand_doff
        new_fit[i] = s


def _ssa_update_aware(
    *,
    rng: np.random.Generator,
    new_dx: np.ndarray,
    new_dy: np.ndarray,
    new_doff: np.ndarray,
    new_fit: np.ndarray,
    cur_best_fit: float,
    worst_fit: float,
    best_dx: np.ndarray,
    best_dy: np.ndarray,
    best_doff: np.ndarray,
    worst_dx: np.ndarray,
    worst_dy: np.ndarray,
    worst_doff: np.ndarray,
    sd_count: int,
    mix: float,
    danger_beta: float,
    base_cx: float,
    base_cy: float,
    base_degs: np.ndarray,
    pos_bound: float,
    ang_bound: float,
    repair_steps: int,
) -> None:
    pop_size = new_dx.shape[0]
    n = new_dx.shape[1]
    danger_idx = rng.choice(pop_size, size=min(sd_count, pop_size), replace=False)

    for i in danger_idx:
        if i == 0:
            continue
        k = rng.uniform(-1.0, 1.0, size=n)
        if new_fit[i] > cur_best_fit:
            cand_dx = best_dx + danger_beta * np.abs(new_dx[i] - best_dx) * k
            cand_dy = best_dy + danger_beta * np.abs(new_dy[i] - best_dy) * k
            cand_doff = best_doff + danger_beta * np.abs(new_doff[i] - best_doff) * k
        else:
            denom = abs(float(new_fit[i]) - float(worst_fit)) + 1e-12
            cand_dx = new_dx[i] + k * (np.abs(new_dx[i] - worst_dx) / denom)
            cand_dy = new_dy[i] + k * (np.abs(new_dy[i] - worst_dy) / denom)
            cand_doff = new_doff[i] + k * (np.abs(new_doff[i] - worst_doff) / denom)

        cand_dx = new_dx[i] + (mix * 0.5) * (cand_dx - new_dx[i])
        cand_dy = new_dy[i] + (mix * 0.5) * (cand_dy - new_dy[i])
        cand_doff = new_doff[i] + (mix * 0.5) * (cand_doff - new_doff[i])

        cand_dx, cand_dy, cand_doff, s = _finalize_candidate(
            base_cx=base_cx,
            base_cy=base_cy,
            base_degs=base_degs,
            cand_dx=cand_dx,
            cand_dy=cand_dy,
            cand_doff=cand_doff,
            best_dx=best_dx,
            best_dy=best_dy,
            best_doff=best_doff,
            pos_bound=pos_bound,
            ang_bound=ang_bound,
            repair_steps=repair_steps,
        )
        new_dx[i] = cand_dx
        new_dy[i] = cand_dy
        new_doff[i] = cand_doff
        new_fit[i] = s


def optimize_group_ssa(
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
    repair_steps: int,
    seed: int,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(base_xs)

    base_cx = float(base_xs.mean())
    base_cy = float(base_ys.mean())
    base_dx0 = base_xs - base_cx
    base_dy0 = base_ys - base_cy

    # 探索境界
    xs0, ys0, degs0 = to_abs_solution(base_cx, base_cy, base_degs, base_dx0, base_dy0, np.zeros_like(base_degs))
    base_score = float(evaluate_group_score(xs0, ys0, degs0))
    base_side = math.sqrt(base_score * n) if n > 0 else 1.0
    pos_bound = float(max(1e-6, base_side * pos_bound_scale))

    pop_dx, pop_dy, pop_doff, fitness = _ssa_initialize_population(
        rng=rng,
        base_cx=base_cx,
        base_cy=base_cy,
        base_degs=base_degs,
        base_dx0=base_dx0,
        base_dy0=base_dy0,
        base_score=base_score,
        pop_size=pop_size,
        init_pos_delta=init_pos_delta,
        init_ang_delta=init_ang_delta,
        init_tries=init_tries,
        pos_bound=pos_bound,
        ang_bound=ang_bound,
    )

    best_dx = pop_dx[0].copy()
    best_dy = pop_dy[0].copy()
    best_doff = pop_doff[0].copy()
    best_fit = float(fitness[0])

    pd_count = max(1, int(round(pop_size * pd_ratio)))
    sd_count = max(1, int(round(pop_size * sd_ratio)))

    for it in range(max(1, n_iters)):
        order = np.argsort(fitness)
        pop_dx = pop_dx[order]
        pop_dy = pop_dy[order]
        pop_doff = pop_doff[order]
        fitness = fitness[order]

        cur_best_dx = pop_dx[0].copy()
        cur_best_dy = pop_dy[0].copy()
        cur_best_doff = pop_doff[0].copy()
        cur_best_fit = float(fitness[0])

        worst_dx = pop_dx[-1]
        worst_dy = pop_dy[-1]
        worst_doff = pop_doff[-1]
        worst_fit = float(fitness[-1])

        if cur_best_fit < best_fit:
            best_fit = cur_best_fit
            best_dx, best_dy, best_doff = cur_best_dx, cur_best_dy, cur_best_doff

        progress = it / max(1, n_iters)
        mix = 0.6 * (1.0 - progress)
        noise_pos = float(pos_delta) * (1.0 - progress)
        noise_ang = float(ang_delta) * (1.0 - progress)

        new_dx = pop_dx.copy()
        new_dy = pop_dy.copy()
        new_doff = pop_doff.copy()
        new_fit = fitness.copy()

        _ssa_update_producers(
            rng=rng,
            pop_dx=pop_dx,
            pop_dy=pop_dy,
            pop_doff=pop_doff,
            new_dx=new_dx,
            new_dy=new_dy,
            new_doff=new_doff,
            new_fit=new_fit,
            pd_count=pd_count,
            st=st,
            mix=mix,
            noise_pos=noise_pos,
            noise_ang=noise_ang,
            base_cx=base_cx,
            base_cy=base_cy,
            base_degs=base_degs,
            best_dx=cur_best_dx,
            best_dy=cur_best_dy,
            best_doff=cur_best_doff,
            pos_bound=pos_bound,
            ang_bound=ang_bound,
            repair_steps=repair_steps,
            n_iters=n_iters,
        )
        _ssa_update_scroungers(
            rng=rng,
            pop_dx=pop_dx,
            pop_dy=pop_dy,
            pop_doff=pop_doff,
            new_dx=new_dx,
            new_dy=new_dy,
            new_doff=new_doff,
            new_fit=new_fit,
            pd_count=pd_count,
            mix=mix,
            noise_pos=noise_pos,
            noise_ang=noise_ang,
            base_cx=base_cx,
            base_cy=base_cy,
            base_degs=base_degs,
            best_dx=cur_best_dx,
            best_dy=cur_best_dy,
            best_doff=cur_best_doff,
            worst_dx=worst_dx,
            worst_dy=worst_dy,
            worst_doff=worst_doff,
            pos_bound=pos_bound,
            ang_bound=ang_bound,
            repair_steps=repair_steps,
        )
        _ssa_update_aware(
            rng=rng,
            new_dx=new_dx,
            new_dy=new_dy,
            new_doff=new_doff,
            new_fit=new_fit,
            cur_best_fit=cur_best_fit,
            worst_fit=worst_fit,
            best_dx=cur_best_dx,
            best_dy=cur_best_dy,
            best_doff=cur_best_doff,
            worst_dx=worst_dx,
            worst_dy=worst_dy,
            worst_doff=worst_doff,
            sd_count=sd_count,
            mix=mix,
            danger_beta=danger_beta,
            base_cx=base_cx,
            base_cy=base_cy,
            base_degs=base_degs,
            pos_bound=pos_bound,
            ang_bound=ang_bound,
            repair_steps=repair_steps,
        )

        pop_dx, pop_dy, pop_doff, fitness = new_dx, new_dy, new_doff, new_fit

        if debug and it % 100 == 0:
            print(f"  iter={it:4d} best={best_fit:.6f} cur_best={cur_best_fit:.6f} base={base_score:.6f}")

    best_xs, best_ys, best_degs2 = to_abs_solution(base_cx, base_cy, base_degs, best_dx, best_dy, best_doff)
    best_score2 = float(evaluate_group_score(best_xs, best_ys, best_degs2))
    return best_xs, best_ys, best_degs2, best_score2


# -----------------------------------------------------------------------------
# IO / Scoring
# -----------------------------------------------------------------------------
def load_submission_data(filepath: str, fallback_path: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_path = filepath
    if not os.path.exists(target_path):
        if fallback_path is None or not os.path.exists(fallback_path):
            raise FileNotFoundError(f"submission not found: {filepath}")
        target_path = fallback_path

    df = pd.read_csv(target_path)
    all_xs, all_ys, all_degs = [], [], []
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")
        for _, row in group.iterrows():
            x = float(row["x"][1:]) if isinstance(row["x"], str) else float(row["x"])
            y = float(row["y"][1:]) if isinstance(row["y"], str) else float(row["y"])
            deg = float(row["deg"][1:]) if isinstance(row["deg"], str) else float(row["deg"])
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("SSA局所探索 (exp021_ssa)")
    print(f"設定: {CONFIG_PATH}")

    paths_cfg = CONFIG["paths"]
    baseline_path = paths_cfg.get("baseline", "submissions/submission.csv")
    fallback_path = paths_cfg.get("baseline_fallback")
    output_path = paths_cfg.get("output", "submissions/submission.csv")

    all_xs, all_ys, all_degs = load_submission_data(baseline_path, fallback_path)
    baseline_total = calculate_total_score(all_xs, all_ys, all_degs)
    print(f"ベースライン合計スコア: {baseline_total:.6f}")

    opt_cfg = CONFIG["optimization"]
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

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    improved_groups = 0
    total_improved = 0.0

    for n in tqdm(range(n_min, n_max + 1), desc="最適化"):
        if target_group_set is not None and n not in target_group_set:
            continue
        if n < 2:
            continue

        start = n * (n - 1) // 2
        xs = new_xs[start : start + n].copy()
        ys = new_ys[start : start + n].copy()
        degs = new_degs[start : start + n].copy()

        orig_score = float(evaluate_group_score(xs, ys, degs))
        seed = seed_base + n * 1000
        opt_xs, opt_ys, opt_degs, opt_score = optimize_group_ssa(
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
        )

        if opt_score < orig_score - 1e-9:
            improvement = orig_score - opt_score
            improved_groups += 1
            total_improved += improvement
            new_xs[start : start + n] = opt_xs
            new_ys[start : start + n] = opt_ys
            new_degs[start : start + n] = opt_degs
            print(f"  グループ{n:03d}: {orig_score:.6f} -> {opt_score:.6f} (改善 {improvement:.6f})")

    final_score = calculate_total_score(new_xs, new_ys, new_degs)
    print("\n最終結果")
    print(f"  最適化後: {final_score:.6f}")
    print(f"  総改善量: {baseline_total - final_score:+.6f}")
    print(f"  改善グループ数: {improved_groups}")

    baseline_improved = final_score < baseline_total - 1e-9
    if os.path.exists(output_path):
        ref_xs, ref_ys, ref_degs = load_submission_data(output_path)
        ref_score = calculate_total_score(ref_xs, ref_ys, ref_degs)
        print(f"  既存提出スコア: {ref_score:.6f}")
        if final_score < ref_score - 1e-9:
            save_submission(output_path, new_xs, new_ys, new_degs)
            print(f"submissionを更新しました: {output_path}")
        else:
            print("submissionより改善なしのため上書きしません")
    elif baseline_improved:
        save_submission(output_path, new_xs, new_ys, new_degs)
        print(f"submissionを作成しました: {output_path}")
    else:
        print("ベースラインから改善なしのため提出ファイルを更新しません")
