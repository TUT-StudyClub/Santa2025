"""
exp021: Guided SA with Squeeze

配置全体を強制的に縮小（Squeeze）させた状態で物理演算を行い、重なりを解消することで密度を限界まで高める手法。
「緩める→締める」という圧力の変化を与えることで、膠着した局所解（Local Optima）から脱出し、さらなるスコア改善を狙う。
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Iterable

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

    return os.path.join("experiments", "exp020_symmetric_slide", "exp", f"{config_name}.yaml")


CONFIG_PATH = resolve_config_path()
# 簡易フォールバック: ファイルがなければカレントディレクトリを探す
if not os.path.exists(CONFIG_PATH):
    if os.path.exists(f"{os.path.basename(CONFIG_PATH)}"):
        CONFIG_PATH = f"{os.path.basename(CONFIG_PATH)}"

try:
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

# 定数読み込み
TREE_CFG = CONFIG.get("tree_shape", {})
TRUNK_W = float(TREE_CFG.get("trunk_w", 20.0))
BASE_W = float(TREE_CFG.get("base_w", 50.0))
MID_W = float(TREE_CFG.get("mid_w", 40.0))
TOP_W = float(TREE_CFG.get("top_w", 30.0))
TIP_Y = float(TREE_CFG.get("tip_y", 100.0))
TIER_1_Y = float(TREE_CFG.get("tier_1_y", 70.0))
TIER_2_Y = float(TREE_CFG.get("tier_2_y", 40.0))
BASE_Y = float(TREE_CFG.get("base_y", 10.0))
TRUNK_BOTTOM_Y = float(TREE_CFG.get("trunk_bottom_y", 0.0))

# --- [重要] スケール推定 ---
# 基準サイズ(TRUNK_W=20.0)に対する現在の比率の二乗（面積比）を計算
# これをSAの温度調整に使う
SCALE_REF = 20.0
TEMP_SCALE_FACTOR = (TRUNK_W / SCALE_REF) ** 2
TEMP_SCALE_FACTOR = min(TEMP_SCALE_FACTOR, 1.0)  # 大きくなる分には緩和しない
TEMP_SCALE_FACTOR = max(TEMP_SCALE_FACTOR, 1e-6)  # ゼロ除算防止

print(f"Tree Scale Factor for SA: {TEMP_SCALE_FACTOR:.8f} (Base Trunk: {SCALE_REF}, Curr: {TRUNK_W})")


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


@njit(cache=True)
def has_any_overlap(all_vertices: list[np.ndarray]) -> bool:
    n = len(all_vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(all_vertices[i], all_vertices[j]):
                return True
    return False


def load_submission_data(filepath: str, fallback_path: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_path = filepath
    if not os.path.exists(target_path):
        if fallback_path is None or not os.path.exists(fallback_path):
            # ダミー
            return np.array([]), np.array([]), np.array([])
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
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


def compute_center_from_polygons(polygons: list[np.ndarray]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = compute_bounds(polygons)
    return (min_x + max_x) * 0.5, (min_y + max_y) * 0.5


def compute_center_from_centers(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    return float(xs.mean()), float(ys.mean())


def build_symmetric_pairs(xs: np.ndarray, ys: np.ndarray, center_x: float, center_y: float) -> list[tuple[int, int]]:
    angles = np.arctan2(ys - center_y, xs - center_x)
    order = np.argsort(angles)
    half = len(xs) // 2
    pairs = [(int(order[i]), int(order[i + half])) for i in range(half)]
    return pairs


def apply_symmetry(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    pairs: Iterable[tuple[int, int]],
    cx: float,
    cy: float,
) -> None:
    for base_idx, mirror_idx in pairs:
        bx, by, bdeg = xs[base_idx], ys[base_idx], degs[base_idx]
        xs[mirror_idx] = 2.0 * cx - bx
        ys[mirror_idx] = 2.0 * cy - by
        degs[mirror_idx] = (bdeg + 180.0) % 360.0


def has_overlap_for_pair(
    base_idx: int,
    mirror_idx: int,
    new_verts_base: np.ndarray,
    new_verts_mirror: np.ndarray,
    polygons: list[np.ndarray],
) -> bool:
    if polygons_overlap(new_verts_base, new_verts_mirror):
        return True
    for j, verts in enumerate(polygons):
        if j in (base_idx, mirror_idx):
            continue
        if polygons_overlap(new_verts_base, verts) or polygons_overlap(new_verts_mirror, verts):
            return True
    return False


def spread_if_overlap(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    cx: float,
    cy: float,
    spread_factor: float,
    max_spread: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], bool]:
    spread = 1.0
    polygons = build_vertices(xs, ys, degs)
    while has_any_overlap(polygons) and spread < max_spread:
        spread *= spread_factor
        xs = cx + (xs - cx) * spread
        ys = cy + (ys - cy) * spread
        polygons = build_vertices(xs, ys, degs)
    return xs, ys, degs, polygons, not has_any_overlap(polygons)


def jitter_pairs(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    polygons: list[np.ndarray],
    pairs: list[tuple[int, int]],
    cx: float,
    cy: float,
    jitter_radius: float,
    jitter_deg: float,
    per_tree_tries: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    if jitter_radius <= 0.0 and jitter_deg <= 0.0:
        return xs, ys, degs, polygons

    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    for pair_idx in indices:
        base_idx, mirror_idx = pairs[int(pair_idx)]
        old_x, old_y, old_deg = xs[base_idx], ys[base_idx], degs[base_idx]
        old_verts_base = polygons[base_idx]
        old_verts_mirror = polygons[mirror_idx]
        moved = False

        for _ in range(per_tree_tries):
            dx = rng.uniform(-jitter_radius, jitter_radius) if jitter_radius > 0.0 else 0.0
            dy = rng.uniform(-jitter_radius, jitter_radius) if jitter_radius > 0.0 else 0.0
            ddeg = rng.uniform(-jitter_deg, jitter_deg) if jitter_deg > 0.0 else 0.0

            new_x = old_x + dx
            new_y = old_y + dy
            new_deg = (old_deg + ddeg) % 360.0
            new_x_m = 2.0 * cx - new_x
            new_y_m = 2.0 * cy - new_y
            new_deg_m = (new_deg + 180.0) % 360.0

            new_verts_base = get_tree_vertices(new_x, new_y, new_deg)
            new_verts_mirror = get_tree_vertices(new_x_m, new_y_m, new_deg_m)

            if has_overlap_for_pair(base_idx, mirror_idx, new_verts_base, new_verts_mirror, polygons):
                continue

            xs[base_idx] = new_x
            ys[base_idx] = new_y
            degs[base_idx] = new_deg
            xs[mirror_idx] = new_x_m
            ys[mirror_idx] = new_y_m
            degs[mirror_idx] = new_deg_m
            polygons[base_idx] = new_verts_base
            polygons[mirror_idx] = new_verts_mirror
            moved = True
            break

        if not moved:
            polygons[base_idx] = old_verts_base
            polygons[mirror_idx] = old_verts_mirror
            xs[base_idx] = old_x
            ys[base_idx] = old_y
            degs[base_idx] = old_deg
            xs[mirror_idx] = 2.0 * cx - old_x
            ys[mirror_idx] = 2.0 * cy - old_y
            degs[mirror_idx] = (old_deg + 180.0) % 360.0

    return xs, ys, degs, polygons


def compute_pair_max_feasible_step(  # noqa: PLR0913
    old_x: float,
    old_y: float,
    old_deg: float,
    polygons: list[np.ndarray],
    base_idx: int,
    mirror_idx: int,
    cx: float,
    cy: float,
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
        new_x = old_x - dir_x * mid
        new_y = old_y - dir_y * mid
        new_x_m = 2.0 * cx - new_x
        new_y_m = 2.0 * cy - new_y
        new_deg_m = (old_deg + 180.0) % 360.0

        new_verts_base = get_tree_vertices(new_x, new_y, old_deg)
        new_verts_mirror = get_tree_vertices(new_x_m, new_y_m, new_deg_m)
        if not has_overlap_for_pair(base_idx, mirror_idx, new_verts_base, new_verts_mirror, polygons):
            low = mid
        else:
            high = mid

    return low if low >= min_step else 0.0


def evaluate_pair_slide_candidates(  # noqa: PLR0913
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    polygons: list[np.ndarray],
    base_idx: int,
    mirror_idx: int,
    cx: float,
    cy: float,
    dir_x: float,
    dir_y: float,
    old_x: float,
    old_y: float,
    old_deg: float,
    old_verts_base: np.ndarray,
    old_verts_mirror: np.ndarray,
    max_feasible: float,
    min_step: float,
    candidate_fracs: list[float],
    current_score: float,
    eps: float,
) -> tuple[bool, float, float, float, np.ndarray, np.ndarray]:
    best_score = current_score
    best_x = old_x
    best_y = old_y
    best_verts_base = old_verts_base
    best_verts_mirror = old_verts_mirror
    moved = False

    for frac in candidate_fracs:
        step = max_feasible * frac
        if step < min_step:
            continue
        new_x = old_x - dir_x * step
        new_y = old_y - dir_y * step
        new_x_m = 2.0 * cx - new_x
        new_y_m = 2.0 * cy - new_y
        new_deg_m = (old_deg + 180.0) % 360.0

        new_verts_base = get_tree_vertices(new_x, new_y, old_deg)
        new_verts_mirror = get_tree_vertices(new_x_m, new_y_m, new_deg_m)
        if has_overlap_for_pair(base_idx, mirror_idx, new_verts_base, new_verts_mirror, polygons):
            continue

        polygons[base_idx] = new_verts_base
        polygons[mirror_idx] = new_verts_mirror
        xs[base_idx] = new_x
        ys[base_idx] = new_y
        xs[mirror_idx] = new_x_m
        ys[mirror_idx] = new_y_m
        degs[mirror_idx] = new_deg_m

        new_score, _ = calculate_score(polygons)

        if new_score < best_score - eps:
            best_score = new_score
            best_x = new_x
            best_y = new_y
            best_verts_base = new_verts_base.copy()
            best_verts_mirror = new_verts_mirror.copy()
            moved = True

        polygons[base_idx] = old_verts_base
        polygons[mirror_idx] = old_verts_mirror
        xs[base_idx] = old_x
        ys[base_idx] = old_y
        degs[base_idx] = old_deg
        xs[mirror_idx] = 2.0 * cx - old_x
        ys[mirror_idx] = 2.0 * cy - old_y
        degs[mirror_idx] = (old_deg + 180.0) % 360.0

    return moved, best_score, best_x, best_y, best_verts_base, best_verts_mirror


def try_slide_pair(  # noqa: PLR0913
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    polygons: list[np.ndarray],
    base_idx: int,
    mirror_idx: int,
    cx: float,
    cy: float,
    dir_x: float,
    dir_y: float,
    current_score: float,
    max_step: float,
    min_step: float,
    search_iters: int,
    candidate_fracs: list[float],
    eps: float,
) -> tuple[bool, float]:
    old_x, old_y, old_deg = xs[base_idx], ys[base_idx], degs[base_idx]
    old_verts_base = polygons[base_idx]
    old_verts_mirror = polygons[mirror_idx]

    max_feasible = compute_pair_max_feasible_step(
        old_x,
        old_y,
        old_deg,
        polygons,
        base_idx,
        mirror_idx,
        cx,
        cy,
        dir_x,
        dir_y,
        max_step,
        min_step,
        search_iters,
    )
    if max_feasible <= 0.0:
        return False, current_score

    moved, best_score, best_x, best_y, best_verts_base, best_verts_mirror = evaluate_pair_slide_candidates(
        xs,
        ys,
        degs,
        polygons,
        base_idx,
        mirror_idx,
        cx,
        cy,
        dir_x,
        dir_y,
        old_x,
        old_y,
        old_deg,
        old_verts_base,
        old_verts_mirror,
        max_feasible,
        min_step,
        candidate_fracs,
        current_score,
        eps,
    )

    if moved and best_score < current_score - eps:
        xs[base_idx] = best_x
        ys[base_idx] = best_y
        xs[mirror_idx] = 2.0 * cx - best_x
        ys[mirror_idx] = 2.0 * cy - best_y
        degs[mirror_idx] = (old_deg + 180.0) % 360.0
        polygons[base_idx] = best_verts_base
        polygons[mirror_idx] = best_verts_mirror
        return True, best_score

    return False, current_score


def slide_pairs(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    pairs: list[tuple[int, int]],
    directions: list[tuple[float, float]],
    sweeps: int,
    max_step_ratio: float,
    min_step: float,
    search_iters: int,
    candidate_fracs: list[float],
    eps: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    polygons = build_vertices(xs, ys, degs)
    current_score, current_side = calculate_score(polygons)

    base_indices = [base for base, _ in pairs]
    mirror_map = {base: mirror for base, mirror in pairs}

    for _ in range(sweeps):
        improved = False
        current_score, current_side = calculate_score(polygons)
        max_step = current_side * max_step_ratio

        for dir_x, dir_y in directions:
            proj = np.array([xs[idx] * dir_x + ys[idx] * dir_y for idx in base_indices])
            order = [base_indices[i] for i in np.argsort(proj)]
            for base_idx in order:
                mirror_idx = mirror_map[base_idx]
                moved, new_score = try_slide_pair(
                    xs,
                    ys,
                    degs,
                    polygons,
                    base_idx,
                    mirror_idx,
                    cx,
                    cy,
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


# -----------------------------------------------------------------------------
# Physics / Squeeze Utils (SA Implementation)
# -----------------------------------------------------------------------------


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
def calculate_continuous_penalty(all_vertices: list[np.ndarray]) -> float:
    total_depth = 0.0
    n = len(all_vertices)
    for i in range(n):
        min_x1, min_y1, max_x1, max_y1 = polygon_bounds(all_vertices[i])
        for j in range(n):
            if i == j:
                continue
            min_x2, min_y2, max_x2, max_y2 = polygon_bounds(all_vertices[j])
            if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                continue
            verts_i = all_vertices[i]
            verts_j = all_vertices[j]
            n_j = verts_j.shape[0]
            for k in range(verts_i.shape[0]):
                px, py = verts_i[k, 0], verts_i[k, 1]
                if point_in_polygon(px, py, verts_j):
                    min_dist_sq = 1e15
                    for m in range(n_j):
                        m_next = (m + 1) % n_j
                        d_sq = point_to_segment_dist_sq(
                            px, py, verts_j[m, 0], verts_j[m, 1], verts_j[m_next, 0], verts_j[m_next, 1]
                        )
                        min_dist_sq = min(d_sq, min_dist_sq)
                    total_depth += math.sqrt(min_dist_sq)
    return total_depth


@njit(cache=True, fastmath=True)
def optimize_overlap_resolution_sa(
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    target_scale: float,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    seed: int,
    t_max: float,
    t_min: float = 0.000001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    np.random.seed(seed)
    n = len(init_xs)
    cx, cy = np.mean(init_xs), np.mean(init_ys)
    # スケール適用 (Squeeze)
    xs = (init_xs - cx) * target_scale + cx
    ys = (init_ys - cy) * target_scale + cy
    degs = init_degs.copy()

    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    current_penalty = calculate_continuous_penalty(all_vertices)

    best_penalty = current_penalty
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()

    if current_penalty <= 1e-9:
        return best_xs, best_ys, best_degs, True

    t_factor = -math.log(t_max / t_min)
    for i in range(n_iters):
        if current_penalty <= 1e-9:
            best_xs[:], best_ys[:], best_degs[:] = xs[:], ys[:], degs[:]
            return best_xs, best_ys, best_degs, True

        progress = i / n_iters
        temp = t_max * math.exp(t_factor * progress)
        idx = np.random.randint(0, n)
        old_x, old_y, old_deg = xs[idx], ys[idx], degs[idx]
        old_verts = all_vertices[idx]

        move = np.random.randint(0, 3)
        if move == 0:
            xs[idx] += (np.random.random() * 2 - 1) * pos_delta * (1.0 - 0.5 * progress)
        elif move == 1:
            ys[idx] += (np.random.random() * 2 - 1) * pos_delta * (1.0 - 0.5 * progress)
        else:
            degs[idx] = (degs[idx] + (np.random.random() * 2 - 1) * ang_delta * (1.0 - 0.5 * progress)) % 360.0

        all_vertices[idx] = get_tree_vertices(xs[idx], ys[idx], degs[idx])
        new_penalty = calculate_continuous_penalty(all_vertices)
        delta = new_penalty - current_penalty

        if delta < 0:
            accept = True
        elif temp < 1e-15:
            accept = False
        else:
            accept = np.random.random() < math.exp(-delta / temp)

        if accept:
            current_penalty = new_penalty
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_xs[:], best_ys[:], best_degs[:] = xs[:], ys[:], degs[:]
        else:
            xs[idx], ys[idx], degs[idx] = old_x, old_y, old_deg
            all_vertices[idx] = old_verts

    return best_xs, best_ys, best_degs, (best_penalty <= 1e-9)


def apply_squeeze_sa(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    ratios: list[float],
    iters: int,
    pos_delta: float,
    ang_delta: float,
    seed: int,
    temp_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """
    強制的に拡大・縮小(ratio)した状態から物理演算で重なりを解消する。
    """
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    poly = build_vertices(best_xs, best_ys, best_degs)
    best_score, _ = calculate_score(poly)

    curr_xs, curr_ys, curr_degs = xs.copy(), ys.copy(), degs.copy()
    updated_any = False

    # スケールに応じた温度設定
    t_max = 0.5 * temp_scale

    for i, r in enumerate(ratios):
        sq_xs, sq_ys, sq_degs, success = optimize_overlap_resolution_sa(
            curr_xs,
            curr_ys,
            curr_degs,
            target_scale=r,
            n_iters=iters,
            pos_delta=pos_delta,
            ang_delta=ang_delta,
            seed=seed + i * 111,
            t_max=t_max,
        )

        if success:
            curr_xs[:], curr_ys[:], curr_degs[:] = sq_xs[:], sq_ys[:], sq_degs[:]
            poly = build_vertices(curr_xs, curr_ys, curr_degs)
            sc, _ = calculate_score(poly)

            if sc < best_score - 1e-9:
                best_score = sc
                best_xs[:] = curr_xs[:]
                best_ys[:] = curr_ys[:]
                best_degs[:] = curr_degs[:]
                updated_any = True

    return best_xs, best_ys, best_degs, best_score, updated_any


if __name__ == "__main__":
    print("点対称スライド最適化 + Squeeze V2 (exp020_symmetric_slide_squeeze_v2)")
    print(f"設定: {CONFIG_PATH}")

    paths_cfg = CONFIG["paths"]
    baseline_path = paths_cfg.get("baseline", "submissions/baseline.csv")
    fallback_path = paths_cfg.get("baseline_fallback", "submissions/baseline.csv")
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
    target_ranges = opt_cfg.get("target_ranges", [])
    even_only = bool(opt_cfg.get("even_only", False))

    sym_cfg = opt_cfg.get("symmetry", {})
    spread_factor = float(sym_cfg.get("spread_factor", 1.05))
    max_spread = float(sym_cfg.get("max_spread", 4.0))
    center_mode = str(sym_cfg.get("center_mode", "bbox"))

    shake_cfg = opt_cfg.get("shake", {})
    shake_enabled = bool(shake_cfg.get("enabled", True))
    jitter_radius_ratio = float(shake_cfg.get("jitter_radius_ratio", 0.12))
    jitter_deg = float(shake_cfg.get("jitter_deg", 0.0))
    per_tree_tries = int(shake_cfg.get("per_tree_tries", 6))
    seed = shake_cfg.get("seed", 0)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    slide_cfg = opt_cfg.get("slide", {})
    sweeps = int(slide_cfg.get("sweeps", 2))
    max_step_ratio = float(slide_cfg.get("max_step_ratio", 0.6))
    min_step = float(slide_cfg.get("min_step", 1e-4))
    search_iters = int(slide_cfg.get("search_iters", 16))
    candidate_fracs = [float(v) for v in slide_cfg.get("candidate_fracs", [1.0, 0.75, 0.5, 0.25])]
    directions_deg = [float(v) for v in slide_cfg.get("directions_deg", [0.0, 90.0, 180.0, 270.0, 45.0, 135.0])]

    # Squeezeの設定読み込み
    squeeze_cfg = opt_cfg.get("physics_squeeze", {})
    use_squeeze = bool(squeeze_cfg.get("enabled", True))
    squeeze_iters = int(squeeze_cfg.get("iters", 5000))
    # Micro-Squeeze用の細かい比率
    default_ratios = [1.005, 0.998, 0.998, 0.999, 0.999, 0.9995]
    squeeze_ratios = [float(v) for v in squeeze_cfg.get("ratios", default_ratios)]

    directions = []
    for angle in directions_deg:
        rad = math.radians(angle)
        directions.append((math.cos(rad), math.sin(rad)))

    range_min = n_min
    range_max = n_max
    if isinstance(target_range, (list, tuple)) and len(target_range) == 2:  # noqa: UP038
        range_min = max(n_min, int(target_range[0]))
        range_max = min(n_max, int(target_range[1]))

    target_group_set = None
    if target_groups:
        target_groups = sorted({int(n) for n in target_groups if range_min <= int(n) <= range_max})
        target_group_set = set(target_groups)
        print(f"対象グループ数: {len(target_groups)}")
    elif target_ranges:
        selected = set()
        for r in target_ranges:
            if not isinstance(r, (list, tuple)) or len(r) != 2:  # noqa: UP038
                continue
            r_start = max(range_min, int(r[0]))
            r_end = min(range_max, int(r[1]))
            if r_start > r_end:
                continue
            selected.update(range(r_start, r_end + 1))
        target_groups = sorted(selected)
        if target_groups:
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
        if even_only and n % 2 == 1:
            continue

        start_idx = n * (n - 1) // 2
        xs = new_xs[start_idx : start_idx + n].copy()
        ys = new_ys[start_idx : start_idx + n].copy()
        degs = new_degs[start_idx : start_idx + n].copy()

        polygons = build_vertices(xs, ys, degs)
        orig_score, _ = calculate_score(polygons)
        if center_mode == "centroid":
            cx, cy = compute_center_from_centers(xs, ys)
        else:
            cx, cy = compute_center_from_polygons(polygons)

        pairs = build_symmetric_pairs(xs, ys, cx, cy)
        apply_symmetry(xs, ys, degs, pairs, cx, cy)

        xs, ys, degs, polygons, ok = spread_if_overlap(xs, ys, degs, cx, cy, spread_factor, max_spread)
        if not ok:
            continue

        if shake_enabled:
            _, side = calculate_score(polygons)
            jitter_radius = side * jitter_radius_ratio
            xs, ys, degs, polygons = jitter_pairs(
                xs,
                ys,
                degs,
                polygons,
                pairs,
                cx,
                cy,
                jitter_radius,
                jitter_deg,
                per_tree_tries,
                rng,
            )

        xs, ys, degs, new_score = slide_pairs(
            xs,
            ys,
            degs,
            pairs,
            directions,
            sweeps,
            max_step_ratio,
            min_step,
            search_iters,
            candidate_fracs,
            eps,
            cx,
            cy,
        )

        # Guided SA with Squeeze (スケール適応版)
        if use_squeeze:
            sq_xs, sq_ys, sq_degs, sq_score, sq_updated = apply_squeeze_sa(
                xs,
                ys,
                degs,
                ratios=squeeze_ratios,
                iters=squeeze_iters,
                pos_delta=0.05,
                ang_delta=2.0,
                seed=n * 999,
                temp_scale=TEMP_SCALE_FACTOR,  # スケール係数を渡す
            )
            if sq_updated and sq_score < new_score - eps:
                xs, ys, degs = sq_xs, sq_ys, sq_degs
                new_score = sq_score

        if new_score < orig_score - eps:
            improved = orig_score - new_score
            new_xs[start_idx : start_idx + n] = xs
            new_ys[start_idx : start_idx + n] = ys
            new_degs[start_idx : start_idx + n] = degs
            improved_groups += 1
            total_improved += improved
            print(f"  グループ {n}: {orig_score:.6f} -> {new_score:.6f} (改善 {improved:.6f})")

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
