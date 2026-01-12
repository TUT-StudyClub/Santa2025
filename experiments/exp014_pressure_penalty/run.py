"""
exp014_pressure_penalty: 圧縮 + ペナルティ付き焼きなまし

重なりを一時的に許容しつつ圧縮し、段階的に罰則を強めて
密な配置へ収束させる実験。
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
# 設定読み込み
# -----------------------------------------------------------------------------


def resolve_config_path() -> str:
    config_name = "000"
    for arg in sys.argv[1:]:
        if arg.startswith("exp="):
            config_name = arg.split("=", 1)[1]
    return os.path.join("experiments", "exp014_pressure_penalty", "exp", f"{config_name}.yaml")


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
# ジオメトリ関数
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


@njit(cache=True)
def compute_bounding_box(all_vertices: list[np.ndarray]) -> tuple[float, float, float, float]:
    min_x, min_y = math.inf, math.inf
    max_x, max_y = -math.inf, -math.inf
    for verts in all_vertices:
        x1, y1, x2, y2 = polygon_bounds(verts)
        min_x = min(x1, min_x)
        min_y = min(y1, min_y)
        max_x = max(x2, max_x)
        max_y = max(y2, max_y)
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def get_side_length(all_vertices: list[np.ndarray]) -> float:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True, fastmath=True)
def calculate_score(all_vertices: list[np.ndarray]) -> float:
    side = get_side_length(all_vertices)
    return side * side / len(all_vertices)


@njit(cache=True)
def count_overlap_pairs(all_vertices: list[np.ndarray]) -> int:
    n = len(all_vertices)
    penalty = 0
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(all_vertices[i], all_vertices[j]):
                penalty += 1
    return penalty


@njit(cache=True)
def count_overlaps_for_tree(tree_idx: int, tree_verts: np.ndarray, all_vertices: list[np.ndarray]) -> int:
    n = len(all_vertices)
    penalty = 0
    for j in range(n):
        if j == tree_idx:
            continue
        if polygons_overlap(tree_verts, all_vertices[j]):
            penalty += 1
    return penalty


@njit(cache=True)
def spread_positions(
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    max_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    scale = 1.0
    xs = init_xs.copy()
    ys = init_ys.copy()
    degs = init_degs.copy()
    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(len(xs))]
    while has_any_overlap(all_vertices) and scale < max_scale:
        scale *= 1.1
        xs = init_xs * scale
        ys = init_ys * scale
        all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(len(xs))]
    return xs, ys, degs, has_any_overlap(all_vertices)


@njit(cache=True)
def sa_penalty_schedule(  # noqa: PLR0915
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    t_max: float,
    t_min: float,
    weight_start: float,
    weight_end: float,
    weight_power: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    bool,
]:
    np.random.seed(seed)
    n = len(init_xs)

    if n_iters <= 0:
        all_vertices = [get_tree_vertices(init_xs[i], init_ys[i], init_degs[i]) for i in range(n)]
        score = calculate_score(all_vertices)
        penalty = count_overlap_pairs(all_vertices)
        return (
            init_xs.copy(),
            init_ys.copy(),
            init_degs.copy(),
            score,
            penalty,
            init_xs.copy(),
            init_ys.copy(),
            init_degs.copy(),
            score,
            penalty == 0,
        )

    if t_max <= 0:
        t_max = 1e-6
    if t_min <= 0:
        t_min = t_max * 0.1
    if t_min > t_max:
        tmp = t_min
        t_min = t_max
        t_max = tmp

    xs = init_xs.copy()
    ys = init_ys.copy()
    degs = init_degs.copy()

    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]

    current_score = calculate_score(all_vertices)
    current_penalty = count_overlap_pairs(all_vertices)
    current_weight = weight_start
    current_cost = current_score + current_weight * current_penalty

    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()
    best_score = current_score
    best_penalty = current_penalty
    best_cost = current_cost

    best_valid_xs = xs.copy()
    best_valid_ys = ys.copy()
    best_valid_degs = degs.copy()
    best_valid_score = math.inf
    has_valid = False
    if current_penalty == 0:
        best_valid_score = current_score
        has_valid = True

    t_factor = -math.log(t_max / t_min) if t_max != t_min else 0.0

    for step in range(n_iters):
        temp = t_max * math.exp(t_factor * step / n_iters) if t_factor != 0.0 else t_min
        decay = 1.0 - 0.8 * (step / n_iters)
        cur_pos = pos_delta * decay
        cur_ang = ang_delta * decay
        progress = step / n_iters
        weight = weight_start + (weight_end - weight_start) * (progress**weight_power)

        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 4)

        old_x, old_y, old_deg = xs[tree_idx], ys[tree_idx], degs[tree_idx]
        old_verts = all_vertices[tree_idx]

        if move_type in (0, 3):
            xs[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type in (1, 3):
            ys[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type in (2, 3):
            degs[tree_idx] = (degs[tree_idx] + (np.random.random() * 2.0 - 1.0) * cur_ang) % 360.0

        new_verts = get_tree_vertices(xs[tree_idx], ys[tree_idx], degs[tree_idx])

        old_overlap = count_overlaps_for_tree(tree_idx, old_verts, all_vertices)
        new_overlap = count_overlaps_for_tree(tree_idx, new_verts, all_vertices)
        new_penalty = current_penalty - old_overlap + new_overlap

        all_vertices[tree_idx] = new_verts
        new_score = calculate_score(all_vertices)
        new_cost = new_score + weight * new_penalty

        delta = new_cost - current_cost
        accept = delta < 0
        if not accept and temp > 1e-12:
            accept = np.random.random() < math.exp(-delta / temp)

        if accept:
            current_score = new_score
            current_penalty = new_penalty
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_score = new_score
                best_penalty = new_penalty
                best_xs[:] = xs[:]
                best_ys[:] = ys[:]
                best_degs[:] = degs[:]
            if new_penalty == 0 and new_score < best_valid_score:
                best_valid_score = new_score
                best_valid_xs[:] = xs[:]
                best_valid_ys[:] = ys[:]
                best_valid_degs[:] = degs[:]
                has_valid = True
        else:
            xs[tree_idx], ys[tree_idx], degs[tree_idx] = old_x, old_y, old_deg
            all_vertices[tree_idx] = old_verts

    return (
        best_xs,
        best_ys,
        best_degs,
        best_score,
        best_penalty,
        best_valid_xs,
        best_valid_ys,
        best_valid_degs,
        best_valid_score,
        has_valid,
    )


@njit(cache=True)
def sa_no_overlap(  # noqa: PLR0915
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    t_max: float,
    t_min: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    np.random.seed(seed)
    n = len(init_xs)

    if n_iters <= 0:
        all_vertices = [get_tree_vertices(init_xs[i], init_ys[i], init_degs[i]) for i in range(n)]
        return init_xs.copy(), init_ys.copy(), init_degs.copy(), calculate_score(all_vertices)

    if t_max <= 0:
        t_max = 1e-6
    if t_min <= 0:
        t_min = t_max * 0.1
    if t_min > t_max:
        tmp = t_min
        t_min = t_max
        t_max = tmp

    xs, ys, degs, still_overlap = spread_positions(init_xs, init_ys, init_degs, 4.0)
    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]

    if still_overlap:
        score = calculate_score(all_vertices)
        return xs, ys, degs, score

    current_score = calculate_score(all_vertices)
    best_score = current_score
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()

    t_factor = -math.log(t_max / t_min) if t_max != t_min else 0.0

    for step in range(n_iters):
        temp = t_max * math.exp(t_factor * step / n_iters) if t_factor != 0.0 else t_min
        decay = 1.0 - 0.8 * (step / n_iters)
        cur_pos = pos_delta * decay
        cur_ang = ang_delta * decay

        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 4)

        old_x, old_y, old_deg = xs[tree_idx], ys[tree_idx], degs[tree_idx]

        if move_type in (0, 3):
            xs[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type in (1, 3):
            ys[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type in (2, 3):
            degs[tree_idx] = (degs[tree_idx] + (np.random.random() * 2.0 - 1.0) * cur_ang) % 360.0

        new_verts = get_tree_vertices(xs[tree_idx], ys[tree_idx], degs[tree_idx])
        overlap = False
        for j in range(n):
            if j != tree_idx and polygons_overlap(new_verts, all_vertices[j]):
                overlap = True
                break
        if overlap:
            xs[tree_idx], ys[tree_idx], degs[tree_idx] = old_x, old_y, old_deg
            continue

        all_vertices[tree_idx] = new_verts
        new_score = calculate_score(all_vertices)
        delta = new_score - current_score
        accept = delta < 0
        if not accept and temp > 1e-12:
            accept = np.random.random() < math.exp(-delta / temp)

        if accept:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_xs[:] = xs[:]
                best_ys[:] = ys[:]
                best_degs[:] = degs[:]
        else:
            xs[tree_idx], ys[tree_idx], degs[tree_idx] = old_x, old_y, old_deg
            all_vertices[tree_idx] = get_tree_vertices(old_x, old_y, old_deg)

    return best_xs, best_ys, best_degs, best_score


# -----------------------------------------------------------------------------
# 入出力
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
        vertices = [get_tree_vertices(all_xs[start + i], all_ys[start + i], all_degs[start + i]) for i in range(n)]
        total += calculate_score(vertices)
    return total


# -----------------------------------------------------------------------------
# 付加処理
# -----------------------------------------------------------------------------


def build_vertices(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> list[np.ndarray]:
    return [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(len(xs))]


def apply_pressure(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    ratio: float,
    jitter_xy: float,
    jitter_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = build_vertices(xs, ys, degs)
    min_x, min_y, max_x, max_y = compute_bounding_box(vertices)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    new_xs = center_x + ratio * (xs - center_x)
    new_ys = center_y + ratio * (ys - center_y)
    new_degs = degs.copy()

    if jitter_xy > 0:
        new_xs += rng.uniform(-jitter_xy, jitter_xy, size=len(xs))
        new_ys += rng.uniform(-jitter_xy, jitter_xy, size=len(xs))
    if jitter_deg > 0:
        new_degs = (new_degs + rng.uniform(-jitter_deg, jitter_deg, size=len(xs))) % 360.0

    return new_xs, new_ys, new_degs


# -----------------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("圧縮ペナルティ最適化 (exp014_pressure_penalty)")
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
    seed_base = int(opt_cfg.get("seed_base", 42))
    target_groups = opt_cfg.get("target_groups", [])
    target_top_k = int(opt_cfg.get("target_top_k", 0))
    target_range = opt_cfg.get("target_range")

    pressure_cfg = opt_cfg.get("pressure", {})
    penalty_cfg = opt_cfg.get("penalty_sa", {})
    refine_cfg = opt_cfg.get("refine", {})

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
            verts = [
                get_tree_vertices(all_xs[start_idx + i], all_ys[start_idx + i], all_degs[start_idx + i])
                for i in range(n)
            ]
            score = calculate_score(verts)
            scores.append((score, n))
        scores.sort(reverse=True, key=lambda x: x[0])
        k = min(target_top_k, len(scores))
        target_groups = sorted([n for _, n in scores[:k]])
        target_group_set = set(target_groups)
        print(f"対象グループ数: {len(target_groups)} (top_k={k}, 範囲={range_min}-{range_max})")

    ratios = pressure_cfg.get("ratios", [])
    jitter_xy = float(pressure_cfg.get("jitter_xy", 0.0))
    jitter_deg = float(pressure_cfg.get("jitter_deg", 0.0))
    trials = int(pressure_cfg.get("trials", 1))

    penalty_enabled = bool(penalty_cfg.get("enabled", True))
    penalty_iters = int(penalty_cfg.get("n_iters", 0))
    penalty_pos = float(penalty_cfg.get("pos_delta", 0.005))
    penalty_ang = float(penalty_cfg.get("ang_delta", 1.0))
    penalty_t_max = float(penalty_cfg.get("T_max", 0.01))
    penalty_t_min = float(penalty_cfg.get("T_min", 0.0001))
    penalty_w_start = float(penalty_cfg.get("weight_start", 0.0))
    penalty_w_end = float(penalty_cfg.get("weight_end", 1.0))
    penalty_w_power = float(penalty_cfg.get("weight_power", 1.0))

    refine_enabled = bool(refine_cfg.get("enabled", False))
    refine_iters = int(refine_cfg.get("n_iters", 0))
    refine_pos = float(refine_cfg.get("pos_delta", 0.003))
    refine_ang = float(refine_cfg.get("ang_delta", 0.6))
    refine_t_max = float(refine_cfg.get("T_max", 0.005))
    refine_t_min = float(refine_cfg.get("T_min", 0.00005))

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

        start_idx = n * (n - 1) // 2
        xs = new_xs[start_idx : start_idx + n].copy()
        ys = new_ys[start_idx : start_idx + n].copy()
        degs = new_degs[start_idx : start_idx + n].copy()

        orig_score = calculate_score(build_vertices(xs, ys, degs))
        best_xs = xs.copy()
        best_ys = ys.copy()
        best_degs = degs.copy()
        best_score = orig_score

        if ratios:
            for ratio in ratios:
                for trial in range(trials):
                    rng = np.random.default_rng(seed_base + n * 1000 + int(ratio * 1000) + trial * 17)
                    cand_xs, cand_ys, cand_degs = apply_pressure(xs, ys, degs, ratio, jitter_xy, jitter_deg, rng)

                    if penalty_enabled:
                        (
                            _best_xs,
                            _best_ys,
                            _best_degs,
                            _best_score,
                            _best_penalty,
                            best_valid_xs,
                            best_valid_ys,
                            best_valid_degs,
                            best_valid_score,
                            has_valid,
                        ) = sa_penalty_schedule(
                            cand_xs,
                            cand_ys,
                            cand_degs,
                            penalty_iters,
                            penalty_pos,
                            penalty_ang,
                            penalty_t_max,
                            penalty_t_min,
                            penalty_w_start,
                            penalty_w_end,
                            penalty_w_power,
                            seed_base + n * 37 + trial * 11,
                        )

                        if not has_valid:
                            continue
                        cand_xs, cand_ys, cand_degs, cand_score = (
                            best_valid_xs,
                            best_valid_ys,
                            best_valid_degs,
                            best_valid_score,
                        )
                    else:
                        cand_score = calculate_score(build_vertices(cand_xs, cand_ys, cand_degs))

                    if refine_enabled:
                        cand_xs, cand_ys, cand_degs, cand_score = sa_no_overlap(
                            cand_xs,
                            cand_ys,
                            cand_degs,
                            refine_iters,
                            refine_pos,
                            refine_ang,
                            refine_t_max,
                            refine_t_min,
                            seed_base + n * 53 + trial * 19,
                        )

                    if cand_score < best_score:
                        best_score = cand_score
                        best_xs = cand_xs.copy()
                        best_ys = cand_ys.copy()
                        best_degs = cand_degs.copy()

        if best_score < orig_score - 1e-9:
            improved = orig_score - best_score
            new_xs[start_idx : start_idx + n] = best_xs
            new_ys[start_idx : start_idx + n] = best_ys
            new_degs[start_idx : start_idx + n] = best_degs
            improved_groups += 1
            total_improved += improved
            print(f"  グループ {n}: {orig_score:.6f} -> {best_score:.6f} (改善 {improved:.6f})")

    final_score = calculate_total_score(new_xs, new_ys, new_degs)
    print("\n最終結果")
    print(f"  最適化後: {final_score:.6f}")
    print(f"  総改善量: {baseline_total - final_score:+.6f}")
    print(f"  改善グループ数: {improved_groups}")

    if os.path.exists(output_path):
        ref_xs, ref_ys, ref_degs = load_submission_data(output_path, fallback_path)
        ref_score = calculate_total_score(ref_xs, ref_ys, ref_degs)
        print(f"  既存提出スコア: {ref_score:.6f}")
        if final_score < ref_score - 1e-9:
            save_submission(output_path, new_xs, new_ys, new_degs)
            print(f"提出ファイルを更新しました: {output_path}")
        else:
            print("提出ファイルより改善なしのため上書きしません")
    elif final_score < baseline_total - 1e-9:
        save_submission(output_path, new_xs, new_ys, new_degs)
        print(f"提出ファイルを作成しました: {output_path}")
    else:
        print("ベースラインから改善なしのため提出ファイルを作成しません")
