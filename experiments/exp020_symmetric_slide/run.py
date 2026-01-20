"""
exp020_symmetric_slide: 偶数Nに対する点対称制約 + 揺らし + スライド最適化

偶数Nのグループで点対称を強制しつつ、揺らしと方向スライドでスコア改善を狙う。
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


if __name__ == "__main__":
    print("点対称スライド最適化 (exp020_symmetric_slide)")
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

    directions = []
    for angle in directions_deg:
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
    elif target_ranges:
        selected = set()
        for r in target_ranges:
            if not isinstance(r, list | tuple) or len(r) != 2:
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
