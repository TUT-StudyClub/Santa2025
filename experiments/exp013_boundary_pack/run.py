"""
exp013_boundary_pack: 境界押し込み + 段階的縮小 + 回転最適化

境界にいる木を内側へ誘導しつつ、段階的に配置を縮めてスコアを下げる実験。
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
    return os.path.join("experiments", "exp013_boundary_pack", "exp", f"{config_name}.yaml")


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


# -----------------------------------------------------------------------------
# ユーティリティ
# -----------------------------------------------------------------------------


def build_vertices(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> list[np.ndarray]:
    return [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(len(xs))]


def get_bounds(all_vertices: list[np.ndarray]) -> tuple[float, float, float, float]:
    return compute_bounding_box(all_vertices)


def get_center_bounds(all_vertices: list[np.ndarray]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return (min_x + max_x) * 0.5, (min_y + max_y) * 0.5


def spread_positions(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    max_scale: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    base_xs = xs.copy()
    base_ys = ys.copy()
    scale = 1.0
    vertices = build_vertices(xs, ys, degs)
    while has_any_overlap(vertices) and scale < max_scale:
        scale *= 1.05
        xs = base_xs * scale
        ys = base_ys * scale
        vertices = build_vertices(xs, ys, degs)
    return xs, ys, degs, has_any_overlap(vertices)


def pick_boundary_indices(
    vertices: list[np.ndarray],
    bounds: tuple[float, float, float, float],
    margin: float,
) -> list[int]:
    min_x, min_y, max_x, max_y = bounds
    indices = []
    for i, verts in enumerate(vertices):
        v_min_x, v_min_y, v_max_x, v_max_y = polygon_bounds(verts)
        if (
            v_min_x <= min_x + margin
            or v_max_x >= max_x - margin
            or v_min_y <= min_y + margin
            or v_max_y >= max_y - margin
        ):
            indices.append(i)
    return indices


def guided_sa(  # noqa: PLR0915
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_iters = int(cfg.get("n_iters", 0))
    pos_delta = float(cfg.get("pos_delta", 0.002))
    ang_delta = float(cfg.get("ang_delta", 1.0))
    t_max = float(cfg.get("T_max", 0.01))
    t_min = float(cfg.get("T_min", 0.0001))
    boundary_bias = float(cfg.get("boundary_bias", 0.7))
    boundary_margin = float(cfg.get("boundary_margin", 0.01))
    refresh_every = max(1, int(cfg.get("refresh_every", 50)))

    if n_iters <= 0:
        score = calculate_score(build_vertices(init_xs, init_ys, init_degs))
        return init_xs.copy(), init_ys.copy(), init_degs.copy(), score

    rng = np.random.default_rng(seed)
    xs = init_xs.copy()
    ys = init_ys.copy()
    degs = init_degs.copy()

    vertices = build_vertices(xs, ys, degs)
    if has_any_overlap(vertices):
        xs, ys, degs, still_overlap = spread_positions(xs, ys, degs)
        if still_overlap:
            return xs, ys, degs, math.inf
        vertices = build_vertices(xs, ys, degs)

    current_score = calculate_score(vertices)
    best_score = current_score
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()

    if t_max > 0 and t_min > 0 and t_max != t_min:
        t_factor = math.log(t_min / t_max)
    else:
        t_factor = 0.0
    bounds = get_bounds(vertices)
    side_len = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    current_margin = boundary_margin * side_len if boundary_margin <= 1.0 else boundary_margin
    boundary_indices = pick_boundary_indices(vertices, bounds, current_margin)

    for step in range(n_iters):
        if t_factor == 0.0:
            temp = t_min if t_min > 0 else 1e-12
        else:
            temp = t_max * math.exp(t_factor * step / n_iters)
        decay = 1.0 - 0.8 * (step / n_iters)
        cur_pos = pos_delta * decay
        cur_ang = ang_delta * decay

        if step % refresh_every == 0:
            bounds = get_bounds(vertices)
            side_len = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            current_margin = boundary_margin * side_len if boundary_margin <= 1.0 else boundary_margin
            boundary_indices = pick_boundary_indices(vertices, bounds, current_margin)

        use_boundary = boundary_indices and rng.random() < boundary_bias
        if use_boundary:
            tree_idx = int(rng.choice(boundary_indices))
        else:
            tree_idx = int(rng.integers(0, len(xs)))

        old_x, old_y, old_deg = xs[tree_idx], ys[tree_idx], degs[tree_idx]
        dx = (rng.random() * 2.0 - 1.0) * cur_pos
        dy = (rng.random() * 2.0 - 1.0) * cur_pos
        ddeg = (rng.random() * 2.0 - 1.0) * cur_ang

        if use_boundary:
            min_x, min_y, max_x, max_y = bounds
            v_min_x, v_min_y, v_max_x, v_max_y = polygon_bounds(vertices[tree_idx])
            dx = 0.0
            dy = 0.0
            if v_min_x <= min_x + current_margin:
                dx += cur_pos
            if v_max_x >= max_x - current_margin:
                dx -= cur_pos
            if v_min_y <= min_y + current_margin:
                dy += cur_pos
            if v_max_y >= max_y - current_margin:
                dy -= cur_pos
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                cx = (min_x + max_x) * 0.5
                cy = (min_y + max_y) * 0.5
                dx = (cx - old_x) * 0.05
                dy = (cy - old_y) * 0.05
            dx += (rng.random() * 2.0 - 1.0) * cur_pos * 0.25
            dy += (rng.random() * 2.0 - 1.0) * cur_pos * 0.25

        xs[tree_idx] = old_x + dx
        ys[tree_idx] = old_y + dy
        degs[tree_idx] = (old_deg + ddeg) % 360.0

        new_verts = get_tree_vertices(xs[tree_idx], ys[tree_idx], degs[tree_idx])
        overlap = False
        for j in range(len(xs)):
            if j != tree_idx and polygons_overlap(new_verts, vertices[j]):
                overlap = True
                break
        if overlap:
            xs[tree_idx], ys[tree_idx], degs[tree_idx] = old_x, old_y, old_deg
            continue

        vertices[tree_idx] = new_verts
        new_score = calculate_score(vertices)
        delta = new_score - current_score
        accept = delta < 0
        if not accept and temp > 1e-12:
            accept = rng.random() < math.exp(-delta / temp)

        if accept:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_xs[:] = xs[:]
                best_ys[:] = ys[:]
                best_degs[:] = degs[:]
        else:
            xs[tree_idx], ys[tree_idx], degs[tree_idx] = old_x, old_y, old_deg
            vertices[tree_idx] = get_tree_vertices(old_x, old_y, old_deg)

    return best_xs, best_ys, best_degs, best_score


def shrink_and_refine(
    base_xs: np.ndarray,
    base_ys: np.ndarray,
    base_degs: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ratios = cfg.get("ratios", [])
    if not ratios:
        score = calculate_score(build_vertices(base_xs, base_ys, base_degs))
        return base_xs.copy(), base_ys.copy(), base_degs.copy(), score

    best_xs = base_xs.copy()
    best_ys = base_ys.copy()
    best_degs = base_degs.copy()
    best_score = calculate_score(build_vertices(best_xs, best_ys, best_degs))

    current_xs = best_xs.copy()
    current_ys = best_ys.copy()
    current_degs = best_degs.copy()

    for idx, ratio in enumerate(ratios):
        vertices = build_vertices(current_xs, current_ys, current_degs)
        center_x, center_y = get_center_bounds(vertices)

        cand_xs = center_x + ratio * (current_xs - center_x)
        cand_ys = center_y + ratio * (current_ys - center_y)
        cand_degs = current_degs.copy()

        cand_xs, cand_ys, cand_degs, cand_score = guided_sa(
            cand_xs,
            cand_ys,
            cand_degs,
            cfg,
            seed + idx * 13,
        )

        if cand_score < best_score:
            best_score = cand_score
            best_xs = cand_xs.copy()
            best_ys = cand_ys.copy()
            best_degs = cand_degs.copy()
            current_xs = best_xs.copy()
            current_ys = best_ys.copy()
            current_degs = best_degs.copy()

    return best_xs, best_ys, best_degs, best_score


def rotate_group(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    center_x: float,
    center_y: float,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = xs - center_x
    dy = ys - center_y
    rot_xs = dx * cos_a - dy * sin_a + center_x
    rot_ys = dx * sin_a + dy * cos_a + center_y
    rot_degs = (degs + angle_deg) % 360.0
    return rot_xs, rot_ys, rot_degs


def global_rotate_opt(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not cfg.get("enabled", False):
        score = calculate_score(build_vertices(xs, ys, degs))
        return xs.copy(), ys.copy(), degs.copy(), score

    coarse_step = float(cfg.get("coarse_step_deg", 3.0))
    fine_step = float(cfg.get("fine_step_deg", 0.25))
    fine_window = float(cfg.get("fine_window_deg", 3.0))

    vertices = build_vertices(xs, ys, degs)
    center_x, center_y = get_center_bounds(vertices)

    best_angle = 0.0
    best_score = calculate_score(vertices)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()

    angle = 0.0
    while angle <= 90.0 + 1e-9:
        rot_xs, rot_ys, rot_degs = rotate_group(xs, ys, degs, center_x, center_y, angle)
        rot_score = calculate_score(build_vertices(rot_xs, rot_ys, rot_degs))
        if rot_score < best_score:
            best_score = rot_score
            best_angle = angle
            best_xs = rot_xs
            best_ys = rot_ys
            best_degs = rot_degs
        angle += coarse_step

    start = max(0.0, best_angle - fine_window)
    end = min(90.0, best_angle + fine_window)
    angle = start
    while angle <= end + 1e-9:
        rot_xs, rot_ys, rot_degs = rotate_group(xs, ys, degs, center_x, center_y, angle)
        rot_score = calculate_score(build_vertices(rot_xs, rot_ys, rot_degs))
        if rot_score < best_score:
            best_score = rot_score
            best_xs = rot_xs
            best_ys = rot_ys
            best_degs = rot_degs
        angle += fine_step

    return best_xs, best_ys, best_degs, best_score


# -----------------------------------------------------------------------------
# データ入出力
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
# メイン
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("境界押し込み最適化 (exp013_boundary_pack)")
    print(f"設定: {CONFIG_PATH}")

    paths_cfg = CONFIG["paths"]
    baseline_path = paths_cfg.get("baseline", "submissions/baseline.csv")
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

    stage1_cfg = opt_cfg.get("stage1", {})
    shrink_cfg = opt_cfg.get("shrink", {})
    rotate_cfg = opt_cfg.get("rotate", {})

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

        if stage1_cfg.get("enabled", False):
            cand_xs, cand_ys, cand_degs, cand_score = guided_sa(
                best_xs,
                best_ys,
                best_degs,
                stage1_cfg,
                seed_base + n * 17,
            )
            if cand_score < best_score:
                best_score = cand_score
                best_xs, best_ys, best_degs = cand_xs, cand_ys, cand_degs

        if shrink_cfg.get("enabled", False):
            cand_xs, cand_ys, cand_degs, cand_score = shrink_and_refine(
                best_xs,
                best_ys,
                best_degs,
                shrink_cfg,
                seed_base + n * 29,
            )
            if cand_score < best_score:
                best_score = cand_score
                best_xs, best_ys, best_degs = cand_xs, cand_ys, cand_degs

        if rotate_cfg.get("enabled", False):
            cand_xs, cand_ys, cand_degs, cand_score = global_rotate_opt(
                best_xs,
                best_ys,
                best_degs,
                rotate_cfg,
            )
            if cand_score < best_score:
                best_score = cand_score
                best_xs, best_ys, best_degs = cand_xs, cand_ys, cand_degs

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
