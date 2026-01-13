"""
exp018_gap_fill: ギャップ中心吸引 + スライド最適化

配置内の最大ギャップを検出し、近傍の木を中心方向に寄せて隙間を埋める。
必要に応じて方向スライドで全体を再調整する。
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


def resolve_config_path() -> str:
    config_name: str | None = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith("exp="):
            config_name = arg.split("=", 1)[1]
            break
        if arg.startswith("--exp="):
            config_name = arg.split("=", 1)[1]
            break
        if arg in ("--exp", "-e") and i + 1 < len(args):
            config_name = args[i + 1]
            break

    if config_name is None:
        config_name = os.getenv("EXP") or os.getenv("exp") or os.getenv("CFG") or os.getenv("cfg")

    if not config_name:
        config_name = "000"

    return os.path.join("experiments", "exp018_gap_fill", "exp", f"{config_name}.yaml")


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
    min_x, min_y = vertices[0, 0], vertices[0, 1]
    max_x, max_y = vertices[0, 0], vertices[0, 1]
    for i in range(1, vertices.shape[0]):
        x, y = vertices[i, 0], vertices[i, 1]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
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
def segments_intersect(  # noqa: PLR0913
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    p3x: float,
    p3y: float,
    p4x: float,
    p4y: float,
) -> bool:
    d1x, d1y = p2x - p1x, p2y - p1y
    d2x, d2y = p4x - p3x, p4y - p3y
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-10:
        return False
    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / det
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / det
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


@njit(cache=True, fastmath=True)
def polygons_overlap(verts1: np.ndarray, verts2: np.ndarray) -> bool:
    if (
        np.max(verts1[:, 0]) < np.min(verts2[:, 0])
        or np.max(verts2[:, 0]) < np.min(verts1[:, 0])
        or np.max(verts1[:, 1]) < np.min(verts2[:, 1])
        or np.max(verts2[:, 1]) < np.min(verts1[:, 1])
    ):
        return False

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
    n1, n2 = verts1.shape[0], verts2.shape[0]
    for i in range(n1):
        j = (i + 1) % n1
        for k in range(n2):
            m = (k + 1) % n2
            if segments_intersect(
                verts1[i, 0],
                verts1[i, 1],
                verts1[j, 0],
                verts1[j, 1],
                verts2[k, 0],
                verts2[k, 1],
                verts2[m, 0],
                verts2[m, 1],
            ):
                return True
    return False


@njit(cache=True, fastmath=True)
def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    c1 = vx * wx + vy * wy
    if c1 <= 0.0:
        return math.hypot(px - ax, py - ay)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - bx, py - by)
    b = c1 / c2
    proj_x = ax + b * vx
    proj_y = ay + b * vy
    return math.hypot(px - proj_x, py - proj_y)


@njit(cache=True, fastmath=True)
def point_polygon_distance(px: float, py: float, vertices: np.ndarray) -> float:
    if point_in_polygon(px, py, vertices):
        return 0.0
    min_d = 1e18
    n = vertices.shape[0]
    for i in range(n):
        j = (i + 1) % n
        d = point_segment_distance(px, py, vertices[i, 0], vertices[i, 1], vertices[j, 0], vertices[j, 1])
        min_d = min(min_d, d)
    return min_d


@njit(cache=True, fastmath=True)
def min_distance_to_polygons(px: float, py: float, polygons: np.ndarray) -> float:
    min_d = 1e18
    for i in range(polygons.shape[0]):
        d = point_polygon_distance(px, py, polygons[i])
        if d < min_d:
            min_d = d
            if min_d <= 0.0:
                return 0.0
    return min_d


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


def build_vertices(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> list[np.ndarray]:
    return [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(len(xs))]


def compute_bounds(polygons: list[np.ndarray]) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for verts in polygons:
        min_x = min(min_x, float(verts[:, 0].min()))
        max_x = max(max_x, float(verts[:, 0].max()))
        min_y = min(min_y, float(verts[:, 1].min()))
        max_y = max(max_y, float(verts[:, 1].max()))
    return min_x, min_y, max_x, max_y


def calculate_score(polygons: list[np.ndarray]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = compute_bounds(polygons)
    side = max(max_x - min_x, max_y - min_y)
    score = (side * side) / len(polygons)
    return score, side


def has_overlap_for_tree(tree_idx: int, new_verts: np.ndarray, polygons: list[np.ndarray]) -> bool:
    for j, verts in enumerate(polygons):
        if j == tree_idx:
            continue
        if polygons_overlap(new_verts, verts):
            return True
    return False


def compute_gap_center(
    polygons: list[np.ndarray],
    grid_n: int,
    margin_ratio: float,
) -> tuple[float, float, float]:
    min_x, min_y, max_x, max_y = compute_bounds(polygons)
    side = max(max_x - min_x, max_y - min_y)
    margin = side * margin_ratio
    width = max_x - min_x
    height = max_y - min_y
    if width <= 2.0 * margin or height <= 2.0 * margin:
        margin = 0.0

    xs = np.linspace(min_x + margin, max_x - margin, max(1, grid_n))
    ys = np.linspace(min_y + margin, max_y - margin, max(1, grid_n))

    polygons_arr = np.stack(polygons)
    best_d = -1.0
    best_x = xs[0]
    best_y = ys[0]
    for x in xs:
        for y in ys:
            d = min_distance_to_polygons(x, y, polygons_arr)
            if d > best_d:
                best_d = d
                best_x = x
                best_y = y
    return best_x, best_y, best_d


def collect_boundary_indices(
    polygons: list[np.ndarray],
    boundary_margin: float,
    boundary_k: int,
) -> list[int]:
    if boundary_margin <= 0.0 and boundary_k <= 0:
        return []

    min_x, min_y, max_x, max_y = compute_bounds(polygons)
    distances: list[tuple[float, int]] = []
    for i, verts in enumerate(polygons):
        pmin_x = float(verts[:, 0].min())
        pmax_x = float(verts[:, 0].max())
        pmin_y = float(verts[:, 1].min())
        pmax_y = float(verts[:, 1].max())
        dist = min(pmin_x - min_x, max_x - pmax_x, pmin_y - min_y, max_y - pmax_y)
        distances.append((dist, i))

    indices: set[int] = set()
    if boundary_margin > 0.0:
        for dist, idx in distances:
            if dist <= boundary_margin:
                indices.add(idx)
    if boundary_k > 0:
        distances.sort(key=lambda x: x[0])
        for _, idx in distances[: min(boundary_k, len(distances))]:
            indices.add(idx)

    return sorted(indices)


def compute_max_feasible_step(
    old_x: float,
    old_y: float,
    old_deg: float,
    polygons: list[np.ndarray],
    tree_idx: int,
    dir_x: float,
    dir_y: float,
    max_step: float,
    min_step: float,
    search_iters: int,
) -> float:
    low = 0.0
    high = max_step
    for _ in range(search_iters):
        mid = (low + high) * 0.5
        new_x = old_x + dir_x * mid
        new_y = old_y + dir_y * mid
        new_verts = get_tree_vertices(new_x, new_y, old_deg)
        if not has_overlap_for_tree(tree_idx, new_verts, polygons):
            low = mid
        else:
            high = mid
    return low if low >= min_step else 0.0


def evaluate_attract_candidates(  # noqa: PLR0913
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    polygons: list[np.ndarray],
    tree_idx: int,
    old_x: float,
    old_y: float,
    old_deg: float,
    old_verts: np.ndarray,
    dir_x: float,
    dir_y: float,
    max_feasible: float,
    min_step: float,
    candidate_fracs: list[float],
    angle_candidates: list[float],
    current_score: float,
    current_gap: float,
    grid_n: int,
    margin_ratio: float,
    score_tol: float,
    eps: float,
    gap_eps: float,
) -> tuple[bool, float, float, float, float, float, np.ndarray]:
    best_score = current_score
    best_gap = current_gap
    best_x = old_x
    best_y = old_y
    best_deg = old_deg
    best_verts = old_verts
    moved = False

    for frac in candidate_fracs:
        step = max_feasible * frac
        if step < min_step:
            continue
        new_x = old_x + dir_x * step
        new_y = old_y + dir_y * step

        for ddeg in angle_candidates:
            new_deg = (old_deg + ddeg) % 360.0
            new_verts = get_tree_vertices(new_x, new_y, new_deg)
            if has_overlap_for_tree(tree_idx, new_verts, polygons):
                continue

            polygons[tree_idx] = new_verts
            xs[tree_idx] = new_x
            ys[tree_idx] = new_y
            degs[tree_idx] = new_deg

            new_score, _ = calculate_score(polygons)
            if new_score <= current_score + score_tol + eps:
                _, _, new_gap = compute_gap_center(polygons, grid_n, margin_ratio)
                if new_score < best_score - eps or (
                    abs(new_score - best_score) <= eps and new_gap < best_gap - gap_eps
                ):
                    best_score = new_score
                    best_gap = new_gap
                    best_x = new_x
                    best_y = new_y
                    best_deg = new_deg
                    best_verts = new_verts.copy()
                    moved = True

            polygons[tree_idx] = old_verts
            xs[tree_idx] = old_x
            ys[tree_idx] = old_y
            degs[tree_idx] = old_deg

    return moved, best_score, best_gap, best_x, best_y, best_deg, best_verts


def try_attract_tree(  # noqa: PLR0913
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    polygons: list[np.ndarray],
    tree_idx: int,
    target_x: float,
    target_y: float,
    current_score: float,
    current_gap: float,
    max_step: float,
    min_step: float,
    search_iters: int,
    candidate_fracs: list[float],
    angle_candidates: list[float],
    grid_n: int,
    margin_ratio: float,
    score_tol: float,
    eps: float,
    gap_eps: float,
) -> tuple[bool, float, float, float, float, float, np.ndarray]:
    old_x, old_y, old_deg = xs[tree_idx], ys[tree_idx], degs[tree_idx]
    old_verts = polygons[tree_idx]

    dx = target_x - old_x
    dy = target_y - old_y
    dist = math.hypot(dx, dy)
    if dist < min_step:
        return False, current_score, current_gap, old_x, old_y, old_deg, old_verts

    dir_x = dx / dist
    dir_y = dy / dist
    max_step = min(max_step, dist)

    max_feasible = compute_max_feasible_step(
        old_x,
        old_y,
        old_deg,
        polygons,
        tree_idx,
        dir_x,
        dir_y,
        max_step,
        min_step,
        search_iters,
    )
    if max_feasible <= 0.0:
        return False, current_score, current_gap, old_x, old_y, old_deg, old_verts

    return evaluate_attract_candidates(
        xs,
        ys,
        degs,
        polygons,
        tree_idx,
        old_x,
        old_y,
        old_deg,
        old_verts,
        dir_x,
        dir_y,
        max_feasible,
        min_step,
        candidate_fracs,
        angle_candidates,
        current_score,
        current_gap,
        grid_n,
        margin_ratio,
        score_tol,
        eps,
        gap_eps,
    )


def gap_fill_group(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    iterations: int,
    grid_n: int,
    margin_ratio: float,
    min_gap: float,
    candidate_k: int,
    boundary_margin_ratio: float,
    boundary_k: int,
    max_step_ratio: float,
    min_step: float,
    search_iters: int,
    candidate_fracs: list[float],
    angle_candidates: list[float],
    score_tol: float,
    eps: float,
    gap_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    polygons = build_vertices(xs, ys, degs)
    current_score, current_side = calculate_score(polygons)
    gap_x, gap_y, gap_score = compute_gap_center(polygons, grid_n, margin_ratio)
    moves = 0

    for _ in range(iterations):
        if gap_score < min_gap:
            break
        max_step = current_side * max_step_ratio
        dists = (xs - gap_x) ** 2 + (ys - gap_y) ** 2
        order = np.argsort(dists)
        top_k = min(candidate_k, len(xs))
        if top_k <= 0:
            break

        boundary_margin = current_side * boundary_margin_ratio if boundary_margin_ratio > 0.0 else 0.0
        boundary_indices = collect_boundary_indices(polygons, boundary_margin, boundary_k)
        candidate_indices = list(order[:top_k])
        if boundary_indices:
            candidate_set = set(candidate_indices)
            for idx in boundary_indices:
                if idx not in candidate_set:
                    candidate_indices.append(idx)
                    candidate_set.add(idx)

        best_move = None
        best_score = current_score
        best_gap = gap_score

        for idx in candidate_indices:
            moved, new_score, new_gap, new_x, new_y, new_deg, new_verts = try_attract_tree(
                xs,
                ys,
                degs,
                polygons,
                int(idx),
                gap_x,
                gap_y,
                current_score,
                gap_score,
                max_step,
                min_step,
                search_iters,
                candidate_fracs,
                angle_candidates,
                grid_n,
                margin_ratio,
                score_tol,
                eps,
                gap_eps,
            )
            if not moved:
                continue
            if new_score < best_score - eps or (abs(new_score - best_score) <= eps and new_gap < best_gap - gap_eps):
                best_score = new_score
                best_gap = new_gap
                best_move = (int(idx), new_x, new_y, new_deg, new_verts)

        if best_move is None:
            break

        tree_idx, best_x, best_y, best_deg, best_verts = best_move
        xs[tree_idx] = best_x
        ys[tree_idx] = best_y
        degs[tree_idx] = best_deg
        polygons[tree_idx] = best_verts
        current_score = best_score
        gap_x, gap_y, gap_score = compute_gap_center(polygons, grid_n, margin_ratio)
        moves += 1

    return xs, ys, degs, current_score, gap_score, moves


def try_slide_tree(  # noqa: PLR0913
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    polygons: list[np.ndarray],
    tree_idx: int,
    dir_x: float,
    dir_y: float,
    current_score: float,
    max_step: float,
    min_step: float,
    search_iters: int,
    candidate_fracs: list[float],
    eps: float,
) -> tuple[bool, float]:
    old_x, old_y = xs[tree_idx], ys[tree_idx]
    old_verts = polygons[tree_idx]

    low = 0.0
    high = max_step
    for _ in range(search_iters):
        mid = (low + high) * 0.5
        new_x = old_x - dir_x * mid
        new_y = old_y - dir_y * mid
        new_verts = get_tree_vertices(new_x, new_y, degs[tree_idx])
        if not has_overlap_for_tree(tree_idx, new_verts, polygons):
            low = mid
        else:
            high = mid

    max_feasible = low
    if max_feasible < min_step:
        return False, current_score

    best_score = current_score
    best_x = old_x
    best_y = old_y
    best_verts = old_verts

    for frac in candidate_fracs:
        step = max_feasible * frac
        if step < min_step:
            continue
        new_x = old_x - dir_x * step
        new_y = old_y - dir_y * step
        new_verts = get_tree_vertices(new_x, new_y, degs[tree_idx])
        if has_overlap_for_tree(tree_idx, new_verts, polygons):
            continue

        polygons[tree_idx] = new_verts
        xs[tree_idx] = new_x
        ys[tree_idx] = new_y
        new_score, _ = calculate_score(polygons)

        if new_score < best_score - eps:
            best_score = new_score
            best_x = new_x
            best_y = new_y
            best_verts = new_verts.copy()

        polygons[tree_idx] = old_verts
        xs[tree_idx] = old_x
        ys[tree_idx] = old_y

    if best_score < current_score - eps:
        xs[tree_idx] = best_x
        ys[tree_idx] = best_y
        polygons[tree_idx] = best_verts
        return True, best_score

    return False, current_score


def slide_group(  # noqa: PLR0913
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    directions: list[tuple[float, float]],
    sweeps: int,
    max_step_ratio: float,
    min_step: float,
    search_iters: int,
    candidate_fracs: list[float],
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    polygons = build_vertices(xs, ys, degs)
    current_score, current_side = calculate_score(polygons)

    for _ in range(sweeps):
        improved = False
        current_score, current_side = calculate_score(polygons)
        max_step = current_side * max_step_ratio

        for dir_x, dir_y in directions:
            proj = xs * dir_x + ys * dir_y
            order = np.argsort(proj)
            for idx in order:
                moved, new_score = try_slide_tree(
                    xs,
                    ys,
                    degs,
                    polygons,
                    int(idx),
                    dir_x,
                    dir_y,
                    current_score,
                    max_step,
                    min_step,
                    search_iters,
                    candidate_fracs,
                    eps,
                )
                if moved:
                    improved = True
                    current_score = new_score
        if not improved:
            break

    return xs, ys, degs, current_score


def calculate_total_score(all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray) -> float:
    total = 0.0
    for n in range(1, 201):
        start = n * (n - 1) // 2
        polygons = [get_tree_vertices(all_xs[start + i], all_ys[start + i], all_degs[start + i]) for i in range(n)]
        score, _ = calculate_score(polygons)
        total += score
    return total


if __name__ == "__main__":
    print("ギャップ吸引スライド最適化 (exp018_gap_fill)")
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

    gap_cfg = opt_cfg.get("gap_fill", {})
    iterations = int(gap_cfg.get("iterations", 4))
    grid_n = int(gap_cfg.get("grid_n", 25))
    margin_ratio = float(gap_cfg.get("grid_margin_ratio", 0.06))
    min_gap = float(gap_cfg.get("min_gap", 0.02))
    candidate_k = int(gap_cfg.get("candidate_k", 8))
    boundary_margin_ratio = float(gap_cfg.get("boundary_margin_ratio", 0.0))
    boundary_k = int(gap_cfg.get("boundary_k", 0))
    max_step_ratio = float(gap_cfg.get("max_step_ratio", 0.6))
    min_step = float(gap_cfg.get("min_step", 1e-4))
    search_iters = int(gap_cfg.get("search_iters", 16))
    candidate_fracs = [float(v) for v in gap_cfg.get("candidate_fracs", [1.0, 0.7, 0.4, 0.2])]
    angle_candidates = [float(v) for v in gap_cfg.get("angle_candidates", [0.0, 10.0, -10.0])]
    score_tol = float(gap_cfg.get("score_tolerance", 0.0))
    gap_eps = float(gap_cfg.get("gap_eps", 1e-4))
    log_details = bool(gap_cfg.get("log_details", False))

    slide_cfg = opt_cfg.get("post_slide", {})
    slide_enabled = bool(slide_cfg.get("enabled", True))
    slide_sweeps = int(slide_cfg.get("sweeps", 2))
    slide_max_step_ratio = float(slide_cfg.get("max_step_ratio", 0.6))
    slide_min_step = float(slide_cfg.get("min_step", 1e-4))
    slide_search_iters = int(slide_cfg.get("search_iters", 16))
    slide_candidate_fracs = [float(v) for v in slide_cfg.get("candidate_fracs", [1.0, 0.75, 0.5, 0.25])]
    slide_dirs_deg = [float(v) for v in slide_cfg.get("directions_deg", [0.0, 90.0, 180.0, 270.0, 45.0, 135.0])]

    directions = []
    for angle in slide_dirs_deg:
        rad = math.radians(angle)
        directions.append((math.cos(rad), math.sin(rad)))

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
            start_idx = n * (n - 1) // 2
            polygons = [
                get_tree_vertices(all_xs[start_idx + i], all_ys[start_idx + i], all_degs[start_idx + i])
                for i in range(n)
            ]
            score, _ = calculate_score(polygons)
            scores.append((score, n))
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
    eps = 1e-9

    for n in tqdm(range(n_min, n_max + 1), desc="最適化"):
        if target_group_set is not None and n not in target_group_set:
            continue
        if n < 2:
            continue

        start_idx = n * (n - 1) // 2
        xs = new_xs[start_idx : start_idx + n].copy()
        ys = new_ys[start_idx : start_idx + n].copy()
        degs = new_degs[start_idx : start_idx + n].copy()

        polygons = build_vertices(xs, ys, degs)
        orig_score, _ = calculate_score(polygons)
        init_gap = None
        if log_details:
            _, _, init_gap = compute_gap_center(polygons, grid_n, margin_ratio)

        xs, ys, degs, mid_score, gap_score, moves = gap_fill_group(
            xs,
            ys,
            degs,
            iterations,
            grid_n,
            margin_ratio,
            min_gap,
            candidate_k,
            boundary_margin_ratio,
            boundary_k,
            max_step_ratio,
            min_step,
            search_iters,
            candidate_fracs,
            angle_candidates,
            score_tol,
            eps,
            gap_eps,
        )
        if log_details and init_gap is not None:
            reason = " (min_gap未満)" if init_gap < min_gap else ""
            print(f"  グループ {n}: 初期ギャップ {init_gap:.4f} -> 最終ギャップ {gap_score:.4f}, 移動 {moves}{reason}")

        new_score = mid_score
        if slide_enabled:
            xs, ys, degs, new_score = slide_group(
                xs,
                ys,
                degs,
                directions,
                slide_sweeps,
                slide_max_step_ratio,
                slide_min_step,
                slide_search_iters,
                slide_candidate_fracs,
                eps,
            )

        if new_score < orig_score - eps:
            improved = orig_score - new_score
            new_xs[start_idx : start_idx + n] = xs
            new_ys[start_idx : start_idx + n] = ys
            new_degs[start_idx : start_idx + n] = degs
            improved_groups += 1
            total_improved += improved
            print(
                f"  グループ {n}: {orig_score:.6f} -> {new_score:.6f} "
                f"(改善 {improved:.6f}, ギャップ {gap_score:.4f}, 移動 {moves})"
            )

    final_score = calculate_total_score(new_xs, new_ys, new_degs)
    print("\n最終結果")
    print(f"  最適化後: {final_score:.6f}")
    print(f"  総改善量: {baseline_total - final_score:+.6f}")
    print(f"  改善グループ数: {improved_groups}")

    submission_score = None
    if os.path.exists(output_path):
        ref_xs, ref_ys, ref_degs = load_submission_data(output_path, fallback_path)
        submission_score = calculate_total_score(ref_xs, ref_ys, ref_degs)
        print(f"  既存提出スコア: {submission_score:.6f}")
    else:
        print("  既存提出ファイル: なし")

    better_than_baseline = final_score < baseline_total - eps
    better_than_submission = submission_score is None or final_score < submission_score - eps

    if better_than_baseline and better_than_submission:
        save_submission(output_path, new_xs, new_ys, new_degs)
        print(f"提出ファイルを更新しました: {output_path}")
    elif not better_than_baseline:
        print("ベースラインから改善なしのため提出ファイルを更新しません")
    elif submission_score is not None:
        print("提出ファイルより改善なしのため上書きしません")
