"""
exp018_direct_optimize: ベースラインの各グループを直接最適化

アプローチ:
1. ベースラインの各グループを読み込む
2. 各グループに対して、個別の木の位置と角度を微調整
3. 勾配降下的なアプローチで局所最適解を探索
4. 特に小さいグループ（スコアが悪い）に集中
"""

import math
import os
import sys
import time
from multiprocessing import Pool, cpu_count

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
        elif arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
    return os.path.join("experiments", "exp018_direct_optimize", "exp", f"{config_name}.yaml")


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
@njit(cache=True)
def rotate_point(x: float, y: float, cos_a: float, sin_a: float) -> tuple[float, float]:
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


@njit(cache=True)
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


@njit(cache=True)
def polygon_bounds(vertices: np.ndarray) -> tuple[float, float, float, float]:
    min_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_x = vertices[0, 0]
    max_y = vertices[0, 1]
    for i in range(1, vertices.shape[0]):
        x = vertices[i, 0]
        y = vertices[i, 1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return min_x, min_y, max_x, max_y


@njit(cache=True)
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


@njit(cache=True)
def segments_intersect(
    p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float, p4x: float, p4y: float
) -> bool:
    d1x = p2x - p1x
    d1y = p2y - p1y
    d2x = p4x - p3x
    d2y = p4y - p3y
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-10:
        return False
    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / det
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / det
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


@njit(cache=True)
def polygons_overlap(verts1: np.ndarray, verts2: np.ndarray) -> bool:
    min_x1, min_y1, max_x1, max_y1 = polygon_bounds(verts1)
    min_x2, min_y2, max_x2, max_y2 = polygon_bounds(verts2)

    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False

    for i in range(verts1.shape[0]):
        if point_in_polygon(verts1[i, 0], verts1[i, 1], verts2):
            return True
    for i in range(verts2.shape[0]):
        if point_in_polygon(verts2[i, 0], verts2[i, 1], verts1):
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
            if segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
                return True
    return False


@njit(cache=True)
def has_any_overlap(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> bool:
    n = len(xs)
    vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(vertices[i], vertices[j]):
                return True
    return False


@njit(cache=True)
def count_overlaps(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> int:
    n = len(xs)
    vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(vertices[i], vertices[j]):
                count += 1
    return count


@njit(cache=True)
def compute_score(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> float:
    """スコア計算: max(width, height)^2 / n"""
    n = len(xs)
    if n == 0:
        return 0.0

    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for i in range(n):
        verts = get_tree_vertices(xs[i], ys[i], degs[i])
        for j in range(15):
            x, y = verts[j, 0], verts[j, 1]
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    side = max(max_x - min_x, max_y - min_y)
    return side * side / n


def rotate_group(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    angle_deg: float,
    center_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if center_mode == "centroid":
        cx = float(xs.mean())
        cy = float(ys.mean())
    else:
        verts = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(len(xs))]
        min_x = min(v[:, 0].min() for v in verts)
        max_x = max(v[:, 0].max() for v in verts)
        min_y = min(v[:, 1].min() for v in verts)
        max_y = max(v[:, 1].max() for v in verts)
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)

    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    dx = xs - cx
    dy = ys - cy
    rx = cx + cos_a * dx - sin_a * dy
    ry = cy + sin_a * dx + cos_a * dy
    rdeg = (degs + angle_deg) % 360.0
    return rx, ry, rdeg


def search_best_rotation(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    angle_min = float(cfg.get("angle_min", 0.0))
    angle_max = float(cfg.get("angle_max", 355.0))
    angle_step = float(cfg.get("angle_step", 5.0))
    center_mode = str(cfg.get("center", "bounds"))

    best_xs = xs
    best_ys = ys
    best_degs = degs
    best_score = compute_score(xs, ys, degs)

    angle = angle_min
    while angle <= angle_max + 1e-9:
        rx, ry, rdeg = rotate_group(xs, ys, degs, angle, center_mode)
        score = compute_score(rx, ry, rdeg)
        if score < best_score - 1e-12:
            best_score = score
            best_xs, best_ys, best_degs = rx, ry, rdeg
        angle += angle_step

    return best_xs, best_ys, best_degs, best_score


def try_jitter_group(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    pos_std: float,
    ang_std: float,
    seed: int,
    max_tries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        jx = xs + rng.normal(0.0, pos_std, size=len(xs))
        jy = ys + rng.normal(0.0, pos_std, size=len(ys))
        jd = (degs + rng.normal(0.0, ang_std, size=len(degs))) % 360.0
        if not has_any_overlap(jx, jy, jd):
            return jx, jy, jd
    return xs, ys, degs


def build_random_group(
    n: int,
    base_side: float,
    seed: int,
    max_restarts: int,
    attempts_per_tree: int,
    expand_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)

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
    max_radius = float(np.sqrt((pts[:, 0] ** 2 + pts[:, 1] ** 2).max()))

    side = max(base_side, 4.0 * max_radius * math.sqrt(n))
    best_score = math.inf
    best_xs = None
    best_ys = None
    best_degs = None

    for _ in range(max_restarts):
        xs = np.empty(n, dtype=np.float64)
        ys = np.empty(n, dtype=np.float64)
        degs = np.empty(n, dtype=np.float64)
        placed = 0

        for i in range(n):
            placed_ok = False
            for _ in range(attempts_per_tree):
                x = rng.uniform(-0.5 * side, 0.5 * side)
                y = rng.uniform(-0.5 * side, 0.5 * side)
                deg = rng.uniform(0.0, 360.0)
                xs[i] = x
                ys[i] = y
                degs[i] = deg
                if placed == 0 or not has_any_overlap(xs[: i + 1], ys[: i + 1], degs[: i + 1]):
                    placed_ok = True
                    placed += 1
                    break
            if not placed_ok:
                break

        if placed == n:
            score = compute_score(xs, ys, degs)
            if score < best_score:
                best_score = score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_degs = degs.copy()
        side *= expand_factor

    if best_xs is None:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), math.inf
    return best_xs, best_ys, best_degs, best_score


@njit(cache=True)
def optimize_group_sa(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    n_iters: int,
    T_max: float,
    T_min: float,
    pos_delta: float,
    ang_delta: float,
    seed: int,
    allow_overlap: bool,
    overlap_penalty_min: float,
    overlap_penalty_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    グループ内の木を個別に微調整するSA
    """
    np.random.seed(seed)
    n = len(xs)

    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()
    best_score = compute_score(xs, ys, degs)

    cur_xs = xs.copy()
    cur_ys = ys.copy()
    cur_degs = degs.copy()
    cur_score = best_score
    cur_overlap = 0
    if allow_overlap:
        cur_overlap = count_overlaps(cur_xs, cur_ys, cur_degs)

    T_factor = -math.log(T_max / T_min)
    overlap_factor = 0.0
    if allow_overlap and overlap_penalty_min > 0.0 and overlap_penalty_max > 0.0:
        overlap_factor = math.log(overlap_penalty_max / overlap_penalty_min)

    for it in range(n_iters):
        progress = it / n_iters
        T = T_max * math.exp(T_factor * progress)
        decay = 1.0 - 0.8 * progress
        penalty = overlap_penalty_min
        if allow_overlap and overlap_penalty_min > 0.0:
            penalty = overlap_penalty_min * math.exp(overlap_factor * progress)

        # ランダムに木を選択
        tree_idx = np.random.randint(0, n)

        # 位置と角度を微調整
        old_x = cur_xs[tree_idx]
        old_y = cur_ys[tree_idx]
        old_deg = cur_degs[tree_idx]

        dx = (np.random.random() * 2.0 - 1.0) * pos_delta * decay
        dy = (np.random.random() * 2.0 - 1.0) * pos_delta * decay
        ddeg = (np.random.random() * 2.0 - 1.0) * ang_delta * decay

        cur_xs[tree_idx] += dx
        cur_ys[tree_idx] += dy
        cur_degs[tree_idx] = (cur_degs[tree_idx] + ddeg) % 360.0

        # オーバーラップチェック
        if not allow_overlap:
            if has_any_overlap(cur_xs, cur_ys, cur_degs):
                cur_xs[tree_idx] = old_x
                cur_ys[tree_idx] = old_y
                cur_degs[tree_idx] = old_deg
                continue

        new_score = compute_score(cur_xs, cur_ys, cur_degs)
        if allow_overlap:
            new_overlap = count_overlaps(cur_xs, cur_ys, cur_degs)
            new_obj = new_score + penalty * new_overlap
            cur_obj = cur_score + penalty * cur_overlap
            delta = new_obj - cur_obj
        else:
            new_overlap = 0
            delta = new_score - cur_score

        accept = False
        if delta < 0:
            accept = True
        elif T > 1e-10 and np.random.random() < math.exp(-delta / T):
            accept = True

        if accept:
            cur_score = new_score
            cur_overlap = new_overlap
            if new_score < best_score:
                best_score = new_score
                best_xs[:] = cur_xs[:]
                best_ys[:] = cur_ys[:]
                best_degs[:] = cur_degs[:]
        else:
            cur_xs[tree_idx] = old_x
            cur_ys[tree_idx] = old_y
            cur_degs[tree_idx] = old_deg

    return best_xs, best_ys, best_degs, best_score


@njit(cache=True)
def optimize_group_coordinate_descent(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    n_passes: int,
    step_size: float,
    step_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    座標降下法による最適化
    各木を順番に、各方向に少しずつ動かしてスコアが改善するか確認
    """
    n = len(xs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()
    best_score = compute_score(xs, ys, degs)

    cur_step = step_size

    for p in range(n_passes):
        improved = False

        for i in range(n):
            # X方向
            for direction in [-1.0, 1.0]:
                test_xs = best_xs.copy()
                test_xs[i] += direction * cur_step

                if not has_any_overlap(test_xs, best_ys, best_degs):
                    score = compute_score(test_xs, best_ys, best_degs)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_xs[i] = test_xs[i]
                        improved = True

            # Y方向
            for direction in [-1.0, 1.0]:
                test_ys = best_ys.copy()
                test_ys[i] += direction * cur_step

                if not has_any_overlap(best_xs, test_ys, best_degs):
                    score = compute_score(best_xs, test_ys, best_degs)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_ys[i] = test_ys[i]
                        improved = True

            # 角度
            for direction in [-1.0, 1.0]:
                test_degs = best_degs.copy()
                test_degs[i] = (test_degs[i] + direction * cur_step * 10.0) % 360.0

                if not has_any_overlap(best_xs, best_ys, test_degs):
                    score = compute_score(best_xs, best_ys, test_degs)
                    if score < best_score - 1e-9:
                        best_score = score
                        best_degs[i] = test_degs[i]
                        improved = True

        if not improved:
            cur_step *= step_decay
            if cur_step < 1e-6:
                break

    return best_xs, best_ys, best_degs, best_score


@njit(cache=True)
def shrink_group(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    min_scale: float,
    n_iters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """グループ全体を縮小"""
    n = len(xs)
    if n <= 1:
        return xs, ys, compute_score(xs, ys, degs)

    # 中心を計算
    cx = 0.0
    cy = 0.0
    for i in range(n):
        cx += xs[i]
        cy += ys[i]
    cx /= n
    cy /= n

    # 二分探索で最小スケールを見つける
    low = min_scale
    high = 1.0
    best_scale = 1.0

    for _ in range(n_iters):
        mid = (low + high) * 0.5
        test_xs = np.empty(n, dtype=np.float64)
        test_ys = np.empty(n, dtype=np.float64)
        for i in range(n):
            test_xs[i] = cx + mid * (xs[i] - cx)
            test_ys[i] = cy + mid * (ys[i] - cy)

        if not has_any_overlap(test_xs, test_ys, degs):
            best_scale = mid
            high = mid
        else:
            low = mid

    # 最良スケールを適用
    new_xs = np.empty(n, dtype=np.float64)
    new_ys = np.empty(n, dtype=np.float64)
    for i in range(n):
        new_xs[i] = cx + best_scale * (xs[i] - cx)
        new_ys[i] = cy + best_scale * (ys[i] - cy)

    score = compute_score(new_xs, new_ys, degs)
    return new_xs, new_ys, score


def shrink_group_aniso(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    min_scale_x: float,
    min_scale_y: float,
    n_iters: int,
    center_mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """グループ全体を異方的に縮小"""
    n = len(xs)
    if n <= 1:
        return xs, ys, compute_score(xs, ys, degs)

    if center_mode == "centroid":
        cx = float(xs.mean())
        cy = float(ys.mean())
    else:
        verts = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
        min_x = min(v[:, 0].min() for v in verts)
        max_x = max(v[:, 0].max() for v in verts)
        min_y = min(v[:, 1].min() for v in verts)
        max_y = max(v[:, 1].max() for v in verts)
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)

    scale_x = 1.0
    scale_y = 1.0

    for axis in ("x", "y"):
        if axis == "x":
            low = min_scale_x
        else:
            low = min_scale_y
        high = 1.0
        best_scale = 1.0

        for _ in range(n_iters):
            mid = 0.5 * (low + high)
            if axis == "x":
                test_xs = cx + mid * (xs - cx)
                test_ys = cy + scale_y * (ys - cy)
            else:
                test_xs = cx + scale_x * (xs - cx)
                test_ys = cy + mid * (ys - cy)

            if not has_any_overlap(test_xs, test_ys, degs):
                best_scale = mid
                high = mid
            else:
                low = mid

        if axis == "x":
            scale_x = best_scale
        else:
            scale_y = best_scale

    new_xs = cx + scale_x * (xs - cx)
    new_ys = cy + scale_y * (ys - cy)
    score = compute_score(new_xs, new_ys, degs)
    return new_xs, new_ys, score


def group_bounds(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for i in range(len(xs)):
        verts = get_tree_vertices(xs[i], ys[i], degs[i])
        min_x = min(min_x, float(verts[:, 0].min()))
        max_x = max(max_x, float(verts[:, 0].max()))
        min_y = min(min_y, float(verts[:, 1].min()))
        max_y = max(max_y, float(verts[:, 1].max()))
    return min_x, min_y, max_x, max_y


def lns_repack(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(xs)
    if n < 2:
        return xs, ys, degs, compute_score(xs, ys, degs)

    rng = np.random.default_rng(seed)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()
    best_score = compute_score(best_xs, best_ys, best_degs)

    n_iters = int(cfg.get("n_iters", 200))
    attempts = int(cfg.get("attempts", 100))
    k_min = int(cfg.get("k_min", 2))
    k_max = int(cfg.get("k_max", 5))

    for it in range(n_iters):
        k = int(rng.integers(k_min, min(k_max, n) + 1))
        idx = rng.choice(n, size=k, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[idx] = False

        fx = best_xs[mask]
        fy = best_ys[mask]
        fd = best_degs[mask]

        min_x, min_y, max_x, max_y = group_bounds(best_xs, best_ys, best_degs)

        improved = False
        for _ in range(attempts):
            rx = rng.uniform(min_x, max_x, size=k)
            ry = rng.uniform(min_y, max_y, size=k)
            rd = rng.uniform(0.0, 360.0, size=k)

            cand_xs = best_xs.copy()
            cand_ys = best_ys.copy()
            cand_degs = best_degs.copy()
            cand_xs[idx] = rx
            cand_ys[idx] = ry
            cand_degs[idx] = rd

            if has_any_overlap(cand_xs, cand_ys, cand_degs):
                continue

            score = compute_score(cand_xs, cand_ys, cand_degs)
            if score < best_score - 1e-12:
                best_xs = cand_xs
                best_ys = cand_ys
                best_degs = cand_degs
                best_score = score
                improved = True
                break

        if improved and it % 10 == 0:
            pass

    return best_xs, best_ys, best_degs, best_score


def optimize_single_group(args):
    """単一グループの最適化（並列処理用）"""
    n, xs, ys, degs, cfg, seed = args

    orig_score = compute_score(xs, ys, degs)
    best_xs = xs
    best_ys = ys
    best_degs = degs
    best_score = orig_score

    rebuild_cfg = cfg.get("rebuild", {})
    if rebuild_cfg.get("enabled", False) and n >= int(rebuild_cfg.get("n_min", 1)) and n <= int(
        rebuild_cfg.get("n_max", 30)
    ):
        base_side = math.sqrt(orig_score * n)
        rx, ry, rdeg, rscore = build_random_group(
            n,
            base_side,
            seed + 999,
            max_restarts=int(rebuild_cfg.get("max_restarts", 20)),
            attempts_per_tree=int(rebuild_cfg.get("attempts_per_tree", 200)),
            expand_factor=float(rebuild_cfg.get("expand_factor", 1.05)),
        )
        if rscore < best_score - 1e-12:
            best_xs, best_ys, best_degs, best_score = rx, ry, rdeg, rscore

    rot_cfg = cfg.get("global_rotation", {})
    if rot_cfg.get("enabled", False):
        best_xs, best_ys, best_degs, best_score = search_best_rotation(best_xs, best_ys, best_degs, rot_cfg)

    ms_cfg = cfg.get("multistart", {})
    restarts = int(ms_cfg.get("restarts", 1))
    pos_std = float(ms_cfg.get("pos_std", 0.0))
    ang_std = float(ms_cfg.get("ang_std", 0.0))
    max_jitter_tries = int(ms_cfg.get("max_jitter_tries", 5))

    for r in range(restarts):
        xs = best_xs.copy()
        ys = best_ys.copy()
        degs = best_degs.copy()

        if pos_std > 0.0 or ang_std > 0.0:
            xs, ys, degs = try_jitter_group(
                xs,
                ys,
                degs,
                pos_std,
                ang_std,
                seed + 1000 * (r + 1),
                max_jitter_tries,
            )

        # 1. SA最適化
        sa_cfg = cfg.get("sa", {})
        if sa_cfg.get("enabled", True):
            xs, ys, degs, _ = optimize_group_sa(
                xs,
                ys,
                degs,
                n_iters=int(sa_cfg.get("n_iters", 10000) * n),  # n に比例
                T_max=float(sa_cfg.get("T_max", 0.1)),
                T_min=float(sa_cfg.get("T_min", 0.0001)),
                pos_delta=float(sa_cfg.get("pos_delta", 0.05)),
                ang_delta=float(sa_cfg.get("ang_delta", 5.0)),
                seed=seed + 10 * (r + 1),
                allow_overlap=bool(sa_cfg.get("allow_overlap", False)),
                overlap_penalty_min=float(sa_cfg.get("overlap_penalty_min", 0.0)),
                overlap_penalty_max=float(sa_cfg.get("overlap_penalty_max", 0.0)),
            )

        # 2. 座標降下
        cd_cfg = cfg.get("coordinate_descent", {})
        if cd_cfg.get("enabled", True):
            xs, ys, degs, _ = optimize_group_coordinate_descent(
                xs,
                ys,
                degs,
                n_passes=int(cd_cfg.get("n_passes", 50)),
                step_size=float(cd_cfg.get("step_size", 0.01)),
                step_decay=float(cd_cfg.get("step_decay", 0.8)),
            )

        # 3. 異方的縮小
        shrink_aniso_cfg = cfg.get("shrink_aniso", {})
        if shrink_aniso_cfg.get("enabled", False):
            xs, ys, _ = shrink_group_aniso(
                xs,
                ys,
                degs,
                min_scale_x=float(shrink_aniso_cfg.get("min_scale_x", 0.9)),
                min_scale_y=float(shrink_aniso_cfg.get("min_scale_y", 0.9)),
                n_iters=int(shrink_aniso_cfg.get("n_iters", 30)),
                center_mode=str(shrink_aniso_cfg.get("center", "bounds")),
            )

        # 4. 縮小
        shrink_cfg = cfg.get("shrink", {})
        if shrink_cfg.get("enabled", True):
            xs, ys, _ = shrink_group(
                xs,
                ys,
                degs,
                min_scale=float(shrink_cfg.get("min_scale", 0.9)),
                n_iters=int(shrink_cfg.get("n_iters", 30)),
            )

        # 5. LNS再配置
        lns_cfg = cfg.get("lns", {})
        if lns_cfg.get("enabled", False):
            xs, ys, degs, _ = lns_repack(xs, ys, degs, lns_cfg, seed + 9999)

        final_score = compute_score(xs, ys, degs)
        if final_score < best_score - 1e-12:
            best_score = final_score
            best_xs = xs
            best_ys = ys
            best_degs = degs

    if rot_cfg.get("post_enabled", False):
        best_xs, best_ys, best_degs, best_score = search_best_rotation(best_xs, best_ys, best_degs, rot_cfg)

    return n, best_xs, best_ys, best_degs, orig_score, best_score


def load_baseline(filepath: str) -> tuple[dict, float]:
    """ベースラインをグループごとに読み込む"""
    df = pd.read_csv(filepath)
    groups = {}
    total_score = 0.0

    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")

        xs = []
        ys = []
        degs = []

        for _, row in group.iterrows():
            x = float(row["x"][1:]) if isinstance(row["x"], str) else float(row["x"])
            y = float(row["y"][1:]) if isinstance(row["y"], str) else float(row["y"])
            deg = float(row["deg"][1:]) if isinstance(row["deg"], str) else float(row["deg"])
            xs.append(x)
            ys.append(y)
            degs.append(deg)

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        degs = np.array(degs, dtype=np.float64)

        groups[n] = (xs, ys, degs)
        total_score += compute_score(xs, ys, degs)

    return groups, total_score


def save_submission(filepath: str, groups: dict) -> None:
    """結果を保存"""
    rows = []
    for n in range(1, 201):
        xs, ys, degs = groups[n]
        for t in range(n):
            rows.append(
                {
                    "id": f"{n:03d}_{t}",
                    "x": f"s{xs[t]}",
                    "y": f"s{ys[t]}",
                    "deg": f"s{degs[t]}",
                }
            )
    pd.DataFrame(rows).to_csv(filepath, index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("exp018_direct_optimize: Direct Group Optimization")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nLoading baseline: {baseline_path}")

    groups, baseline_total = load_baseline(baseline_path)
    print(f"Baseline total score: {baseline_total:.6f}")

    # 最適化対象のグループを選択
    opt_cfg = CONFIG.get("optimization", {})
    n_min = int(opt_cfg.get("n_min", 2))
    n_max = int(opt_cfg.get("n_max", 200))
    seed_base = int(opt_cfg.get("seed", 42))

    # タスクを準備
    tasks = []
    for n in range(n_min, n_max + 1):
        xs, ys, degs = groups[n]
        tasks.append((n, xs.copy(), ys.copy(), degs.copy(), opt_cfg, seed_base + n))

    # 並列処理
    print(f"\nOptimizing groups {n_min} to {n_max}...")
    num_workers = min(cpu_count(), len(tasks))
    t0 = time.time()

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(optimize_single_group, tasks),
            total=len(tasks),
            desc="Optimizing",
        ):
            results.append(result)

    print(f"Optimization completed in {time.time() - t0:.1f}s")

    # 結果を統合
    improved_groups = {}
    total_improvement = 0.0
    improved_count = 0

    for n, xs, ys, degs, orig_score, final_score in results:
        improvement = orig_score - final_score
        if improvement > 1e-9:
            improved_groups[n] = (xs, ys, degs)
            total_improvement += improvement
            improved_count += 1
            if improvement > 0.001:
                print(f"  Group {n}: {orig_score:.6f} -> {final_score:.6f} ({improvement:+.6f})")

    print(f"\nImproved {improved_count} groups")
    print(f"Total improvement: {total_improvement:.6f}")

    # 結果をマージ
    for n, (xs, ys, degs) in improved_groups.items():
        groups[n] = (xs, ys, degs)

    # 最終スコア計算
    final_total = 0.0
    for n in range(1, 201):
        xs, ys, degs = groups[n]
        final_total += compute_score(xs, ys, degs)

    print("=" * 80)
    print(f"  Baseline total:    {baseline_total:.6f}")
    print(f"  Final total:       {final_total:.6f}")
    print(f"  Total improvement: {baseline_total - final_total:+.6f}")
    print("=" * 80)

    # 保存
    if final_total < baseline_total:
        out_path = CONFIG["paths"]["output"]
        save_submission(out_path, groups)
        print(f"Saved to {out_path}")
    else:
        print("No improvement - keeping baseline")
