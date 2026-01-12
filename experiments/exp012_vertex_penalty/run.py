"""
exp012_vertex_penalty: 頂点ペナルティ法を用いた密配置最適化

「『重なったら即アウト』という厳しいルールを一時的にやめて、
『重なった分だけ罰点』という緩いルールで無理やり詰め込む手法」

"""

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
    return os.path.join("experiments", "exp012_vertex_penalty", "exp", f"{config_name}.yaml")


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
def segments_intersect(
    p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float, p4x: float, p4y: float
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
    # バウンディングボックス判定 (AABB check)
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
# Dense Packing Patterns
# -----------------------------------------------------------------------------
@njit(cache=True)
def generate_interlocking_pattern(
    n: int, spacing: float, angle_offset: float, fixed_cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    互い違いの配置パターンを生成
    木を交互に上下逆にして密にパッキング
    """
    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    degs = np.zeros(n, dtype=np.float64)

    if fixed_cols > 0:
        cols = fixed_cols
    else:
        cols = int(math.ceil(math.sqrt(n)))

    row_spacing = spacing * 0.7  # 行間を狭く

    idx = 0
    row = 0
    while idx < n:
        for col in range(cols):
            if idx >= n:
                break
            # 偶数行と奇数行で半分ずらす
            x_offset = (spacing * 0.5) if row % 2 == 1 else 0.0
            xs[idx] = col * spacing + x_offset
            ys[idx] = row * row_spacing
            # 交互に180度回転
            degs[idx] = angle_offset if (row + col) % 2 == 0 else (angle_offset + 180.0) % 360.0
            idx += 1
        row += 1

    return xs, ys, degs


@njit(cache=True)
def generate_hexagonal_pattern(
    n: int, spacing: float, angle_offset: float, fixed_cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    六角形配置パターン（より密なパッキング）
    """
    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    degs = np.zeros(n, dtype=np.float64)

    if fixed_cols > 0:
        cols = fixed_cols
    else:
        cols = int(math.ceil(math.sqrt(n * 1.2)))

    row_spacing = spacing * 0.866  # sqrt(3)/2

    idx = 0
    row = 0
    while idx < n:
        for col in range(cols):
            if idx >= n:
                break
            x_offset = (spacing * 0.5) if row % 2 == 1 else 0.0
            xs[idx] = col * spacing + x_offset
            ys[idx] = row * row_spacing
            degs[idx] = angle_offset
            idx += 1
        row += 1

    return xs, ys, degs


@njit(cache=True)
def generate_diagonal_pattern(
    n: int, spacing: float, angle1: float, angle2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    対角配置パターン
    """
    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    degs = np.zeros(n, dtype=np.float64)

    for i in range(n):
        xs[i] = (i % 2) * spacing * 0.6
        ys[i] = i * spacing * 0.5
        degs[i] = angle1 if i % 2 == 0 else angle2

    return xs, ys, degs


@njit(cache=True)
def optimize_pattern_sa(  # noqa: PLR0915
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    t_max: float,  # T_max -> t_max
    t_min: float,  # T_min -> t_min
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """SA最適化"""
    np.random.seed(random_seed)
    n = len(init_xs)

    xs = init_xs.copy()
    ys = init_ys.copy()
    degs = init_degs.copy()

    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]

    # 重なりを解消
    spread = 1.0
    while has_any_overlap(all_vertices) and spread < 5.0:
        spread *= 1.1
        for i in range(n):
            xs[i] = init_xs[i] * spread
            ys[i] = init_ys[i] * spread
        all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]

    if has_any_overlap(all_vertices):
        return xs, ys, degs, math.inf

    current_score = calculate_score(all_vertices)
    best_score = current_score
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()

    # T_factor -> t_factor
    t_factor = -math.log(t_max / t_min)

    for step in range(n_iters):
        # T -> current_t
        current_t = t_max * math.exp(t_factor * step / n_iters)
        decay = 1.0 - 0.8 * (step / n_iters)
        cur_pos = pos_delta * decay
        cur_ang = ang_delta * decay

        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 4)

        old_x, old_y, old_deg = xs[tree_idx], ys[tree_idx], degs[tree_idx]

        # if A or B -> if A in (0, 1)
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

        # T -> current_t
        if delta < 0 or (current_t > 1e-10 and np.random.random() < math.exp(-delta / current_t)):
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
def optimize_pattern_sa_penalty(  # noqa: PLR0915
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    t_max: float,
    t_min: float,
    penalty_weight: float,
    penalty_growth: float,
    random_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    bool,
]:
    """重なり許容のペナルティ付きSA最適化"""
    np.random.seed(random_seed)
    n = len(init_xs)

    xs = init_xs.copy()
    ys = init_ys.copy()
    degs = init_degs.copy()

    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]

    current_score = calculate_score(all_vertices)
    current_penalty = count_overlap_pairs(all_vertices)
    current_cost = current_score + penalty_weight * current_penalty

    best_cost = current_cost
    best_score = current_score
    best_penalty = current_penalty
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()

    best_valid_score = math.inf
    best_valid_xs = xs.copy()
    best_valid_ys = ys.copy()
    best_valid_degs = degs.copy()
    has_valid = False
    if current_penalty == 0:
        best_valid_score = current_score
        best_valid_xs = xs.copy()
        best_valid_ys = ys.copy()
        best_valid_degs = degs.copy()
        has_valid = True

    t_factor = -math.log(t_max / t_min)

    for step in range(n_iters):
        current_t = t_max * math.exp(t_factor * step / n_iters)
        progress = step / n_iters
        decay = 1.0 - 0.8 * progress
        cur_pos = pos_delta * decay
        cur_ang = ang_delta * decay
        weight = penalty_weight * (1.0 + penalty_growth * progress)

        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 4)

        old_x, old_y, old_deg = xs[tree_idx], ys[tree_idx], degs[tree_idx]
        old_verts = all_vertices[tree_idx]
        old_overlap = count_overlaps_for_tree(tree_idx, old_verts, all_vertices)

        if move_type in (0, 3):
            xs[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type in (1, 3):
            ys[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type in (2, 3):
            degs[tree_idx] = (degs[tree_idx] + (np.random.random() * 2.0 - 1.0) * cur_ang) % 360.0

        new_verts = get_tree_vertices(xs[tree_idx], ys[tree_idx], degs[tree_idx])
        all_vertices[tree_idx] = new_verts
        new_overlap = count_overlaps_for_tree(tree_idx, new_verts, all_vertices)
        new_penalty = current_penalty - old_overlap + new_overlap

        new_score = calculate_score(all_vertices)
        new_cost = new_score + weight * new_penalty
        delta = new_cost - current_cost

        accept = False
        if delta < 0:
            accept = True
        elif current_t > 1e-10 and np.random.random() < math.exp(-delta / current_t):  # noqa: PLR2004
            accept = True

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
        best_cost,
        best_valid_xs,
        best_valid_ys,
        best_valid_degs,
        best_valid_score,
        has_valid,
    )


# -----------------------------------------------------------------------------
# Data Loading/Saving
# -----------------------------------------------------------------------------
def load_submission_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
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


@njit(cache=True, fastmath=True)
def calculate_continuous_penalty(all_vertices: list[np.ndarray]) -> float:
    """
    めり込み深度（Penetration Depth）の総和を計算する。
    戻り値: float (0.0なら重なりなし)
    """
    total_depth = 0.0
    n = len(all_vertices)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # バウンディングボックスによる早期枝刈り
            min_x1, min_y1, max_x1, max_y1 = polygon_bounds(all_vertices[i])
            min_x2, min_y2, max_x2, max_y2 = polygon_bounds(all_vertices[j])

            if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                continue

            # 詳細計算
            verts_i = all_vertices[i]
            verts_j = all_vertices[j]
            n_j = verts_j.shape[0]

            # 「ポリゴンiの頂点」が「ポリゴンj」に入り込んでいるか？
            for k in range(verts_i.shape[0]):
                px, py = verts_i[k, 0], verts_i[k, 1]

                # 重なっている場合のみ距離を計算
                if point_in_polygon(px, py, verts_j):
                    # 脱出に必要な「最短距離」を探す
                    min_dist_sq = 1e15

                    for m in range(n_j):
                        m_next = (m + 1) % n_j
                        d_sq = point_to_segment_dist_sq(
                            px, py, verts_j[m, 0], verts_j[m, 1], verts_j[m_next, 0], verts_j[m_next, 1]
                        )
                        min_dist_sq = min(d_sq, min_dist_sq)

                    # 距離の二乗から距離へ戻して加算
                    total_depth += math.sqrt(min_dist_sq)

    return total_depth


@njit(cache=True, fastmath=True)
def optimize_overlap_resolution(
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    target_scale: float,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    # 物理ベースのSqueeze & Resolveソルバー (連続ペナルティ版)
    np.random.seed(seed)
    n = len(init_xs)

    # 1. Squeeze
    cx = np.mean(init_xs)
    cy = np.mean(init_ys)

    xs = (init_xs - cx) * target_scale + cx
    ys = (init_ys - cy) * target_scale + cy
    degs = init_degs.copy()

    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]

    # 連続値関数の呼び出し
    current_penalty = calculate_continuous_penalty(all_vertices)

    if current_penalty <= 1e-9:
        return xs, ys, degs, True

    # 2. Resolve Loop
    for _ in range(n_iters):
        if current_penalty <= 1e-9:
            return xs, ys, degs, True

        idx = np.random.randint(0, n)
        old_x, old_y, old_deg = xs[idx], ys[idx], degs[idx]
        old_verts = all_vertices[idx]

        move_type = np.random.randint(0, 3)
        if move_type == 0:
            xs[idx] += (np.random.random() * 2.0 - 1.0) * pos_delta
        elif move_type == 1:
            ys[idx] += (np.random.random() * 2.0 - 1.0) * pos_delta
        else:
            degs[idx] = (degs[idx] + (np.random.random() * 2.0 - 1.0) * ang_delta) % 360.0

        all_vertices[idx] = get_tree_vertices(xs[idx], ys[idx], degs[idx])

        # 連続値関数の呼び出し
        new_penalty = calculate_continuous_penalty(all_vertices)

        if new_penalty <= current_penalty:
            current_penalty = new_penalty
        else:
            xs[idx], ys[idx], degs[idx] = old_x, old_y, old_deg
            all_vertices[idx] = old_verts

    # 成功判定のしきい値
    return xs, ys, degs, (current_penalty <= 1e-9)


@njit(cache=True, fastmath=True)
def point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
    """
    点(px, py) と 線分(x1, y1)-(x2, y2) の最短距離の「二乗」を返す。
    線分の端点も考慮する。
    """
    # 線分の長さの二乗
    l2 = (x1 - x2) ** 2 + (y1 - y2) ** 2

    if l2 == 0:
        # 線分がつぶれている場合
        return (px - x1) ** 2 + (py - y1) ** 2

    # 線分上の射影点 t (0.0 <= t <= 1.0) を求める
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2
    t = max(0.0, min(1.0, t))

    # 射影点の座標
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    # 距離の二乗
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Dense Packing Optimizer (Screening + Final + Physics Squeeze)")
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    all_xs, all_ys, all_degs = load_submission_data(baseline_path)
    baseline_total = calculate_total_score(all_xs, all_ys, all_degs)
    print(f"Baseline total score: {baseline_total:.6f}")

    opt_cfg = CONFIG["optimization"]
    n_min = int(opt_cfg["n_min"])
    n_max = int(opt_cfg["n_max"])
    n_iters = int(opt_cfg["n_iters"])
    pos_delta = float(opt_cfg["pos_delta"])
    ang_delta = float(opt_cfg["ang_delta"])
    T_max = float(opt_cfg["T_max"])
    T_min = float(opt_cfg["T_min"])
    seed_base = int(opt_cfg.get("seed_base", 42))
    use_penalty_sa = bool(opt_cfg.get("use_penalty_sa", False))
    penalty_weight = float(opt_cfg.get("penalty_weight", 0.0))
    penalty_growth = float(opt_cfg.get("penalty_growth", 0.0))
    target_groups = opt_cfg.get("target_groups", [])
    target_top_k = int(opt_cfg.get("target_top_k", 0))
    target_range = opt_cfg.get("target_range")

    # Two-stage Optimization Parameters
    screening_iters = int(opt_cfg.get("screening_iters", 5000))  # 高速に形状の良し悪しを判定
    final_iters = int(opt_cfg.get("final_iters", n_iters))  # 十分な時間をかけて最適化

    # Physics Squeeze Parameters
    physics_iters = int(opt_cfg.get("physics_iters", 100000))  # 物理演算の試行回数
    physics_ratios = opt_cfg.get("physics_ratios", [0.90, 0.93, 0.95, 0.97, 0.98, 0.99])  # 圧縮率の候補
    physics_pos_delta = float(opt_cfg.get("physics_pos_delta", 0.05))
    physics_ang_delta = float(opt_cfg.get("physics_ang_delta", 5.0))

    print(f"\nOptimizing groups {n_min} to {n_max}...")
    if use_penalty_sa:
        print(f"  Penalty SA: weight={penalty_weight}, growth={penalty_growth}")

    range_min = n_min
    range_max = n_max
    if isinstance(target_range, (list, tuple)) and len(target_range) == 2:
        range_min = max(n_min, int(target_range[0]))
        range_max = min(n_max, int(target_range[1]))

    target_group_set = None
    if target_groups:
        target_groups = sorted({int(n) for n in target_groups if n_min <= int(n) <= n_max})
        target_group_set = set(target_groups)
        print(f"  対象グループ数: {len(target_groups)}")
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
        print(f"  対象グループ数: {len(target_groups)} (top_k={k}, range={range_min}-{range_max})")

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()
    improved_groups = 0
    total_improved = 0.0

    # Grid Search Parameters
    spacings = [0.65, 0.70, 0.75, 0.80]
    angles = [0.0, 45.0, 90.0, 135.0]

    for n in tqdm(range(n_min, n_max + 1), desc="Optimizing"):
        if target_group_set is not None and n not in target_group_set:
            continue
        # データインデックスの計算（N=1からの累積和）
        start_idx = n * (n - 1) // 2

        # 現在のベースラインスコア
        orig_verts = [
            get_tree_vertices(new_xs[start_idx + i], new_ys[start_idx + i], new_degs[start_idx + i]) for i in range(n)
        ]
        orig_score = calculate_score(orig_verts)

        # ---------------------------------------------------------
        # Phase 1: Screening (予選)
        # ---------------------------------------------------------
        best_candidate_params = None
        best_candidate_score = math.inf

        # アスペクト比探索：正方形(sqrt(n))周辺の列数を重点的に探索
        base_cols = int(math.ceil(math.sqrt(n)))
        col_search_range = range(max(1, base_cols - 2), base_cols + 4)

        for spacing in spacings:
            for angle in angles:
                for n_cols in col_search_range:
                    seed = seed_base + n * 1000 + int(spacing * 100) + int(angle) + n_cols
                    candidates = []

                    # Pattern A: Interlocking
                    # ※generate関数には fixed_cols 引数が必要
                    xs1, ys1, degs1 = generate_interlocking_pattern(n, spacing, angle, n_cols)
                    if len(xs1) == n:
                        candidates.append((xs1, ys1, degs1))

                    # Pattern B: Hexagonal
                    xs2, ys2, degs2 = generate_hexagonal_pattern(n, spacing, angle, n_cols)
                    if len(xs2) == n:
                        candidates.append((xs2, ys2, degs2))

                    for c_xs, c_ys, c_degs in candidates:
                        # 短時間実行で有望度を判定
                        _, _, _, score = optimize_pattern_sa(
                            c_xs,
                            c_ys,
                            c_degs,
                            screening_iters,
                            pos_delta,
                            ang_delta,
                            T_max,
                            T_min,
                            seed,
                        )
                        if score < best_candidate_score:
                            best_candidate_score = score
                            best_candidate_params = (c_xs, c_ys, c_degs, seed)

        # ---------------------------------------------------------
        # Phase 2: Final Optimization (決勝)
        # ---------------------------------------------------------
        best_score = orig_score

        # 作業用バッファ
        best_xs_group = new_xs[start_idx : start_idx + n].copy()
        best_ys_group = new_ys[start_idx : start_idx + n].copy()
        best_degs_group = new_degs[start_idx : start_idx + n].copy()

        physics_seed_cost = math.inf
        physics_seed_xs = best_xs_group.copy()
        physics_seed_ys = best_ys_group.copy()
        physics_seed_degs = best_degs_group.copy()

        # 予選勝者の本番実行
        if best_candidate_params is not None:
            init_xs, init_ys, init_degs, best_seed = best_candidate_params
            if use_penalty_sa:
                (
                    opt_xs,
                    opt_ys,
                    opt_degs,
                    opt_score,
                    opt_penalty,
                    opt_cost,
                    valid_xs,
                    valid_ys,
                    valid_degs,
                    valid_score,
                    has_valid,
                ) = optimize_pattern_sa_penalty(
                    init_xs,
                    init_ys,
                    init_degs,
                    final_iters,
                    pos_delta,
                    ang_delta,
                    T_max,
                    T_min,
                    penalty_weight,
                    penalty_growth,
                    best_seed,
                )
                if has_valid and valid_score < best_score:
                    best_score = valid_score
                    best_xs_group[:] = valid_xs[:]
                    best_ys_group[:] = valid_ys[:]
                    best_degs_group[:] = valid_degs[:]
                if opt_cost < physics_seed_cost and opt_penalty >= 0:
                    physics_seed_cost = opt_cost
                    physics_seed_xs = opt_xs.copy()
                    physics_seed_ys = opt_ys.copy()
                    physics_seed_degs = opt_degs.copy()
            else:
                opt_xs, opt_ys, opt_degs, score = optimize_pattern_sa(
                    init_xs,
                    init_ys,
                    init_degs,
                    final_iters,
                    pos_delta,
                    ang_delta,
                    T_max,
                    T_min,
                    best_seed,
                )
                if score < best_score:
                    best_score = score
                    best_xs_group[:] = opt_xs[:]
                    best_ys_group[:] = opt_ys[:]
                    best_degs_group[:] = opt_degs[:]
                if score < physics_seed_cost:
                    physics_seed_cost = score
                    physics_seed_xs = opt_xs.copy()
                    physics_seed_ys = opt_ys.copy()
                    physics_seed_degs = opt_degs.copy()

        # 既存配置からの微修正も並行して実施
        base_xs = new_xs[start_idx : start_idx + n].copy()
        base_ys = new_ys[start_idx : start_idx + n].copy()
        base_degs = new_degs[start_idx : start_idx + n].copy()

        if use_penalty_sa:
            (
                opt_xs,
                opt_ys,
                opt_degs,
                opt_score,
                opt_penalty,
                opt_cost,
                valid_xs,
                valid_ys,
                valid_degs,
                valid_score,
                has_valid,
            ) = optimize_pattern_sa_penalty(
                base_xs,
                base_ys,
                base_degs,
                final_iters,
                pos_delta,
                ang_delta,
                T_max,
                T_min,
                penalty_weight,
                penalty_growth,
                seed_base + n * 9999,
            )
            if has_valid and valid_score < best_score:
                best_score = valid_score
                best_xs_group[:] = valid_xs[:]
                best_ys_group[:] = valid_ys[:]
                best_degs_group[:] = valid_degs[:]
            if opt_cost < physics_seed_cost and opt_penalty >= 0:
                physics_seed_cost = opt_cost
                physics_seed_xs = opt_xs.copy()
                physics_seed_ys = opt_ys.copy()
                physics_seed_degs = opt_degs.copy()
        else:
            opt_xs, opt_ys, opt_degs, score = optimize_pattern_sa(
                base_xs,
                base_ys,
                base_degs,
                final_iters,
                pos_delta,
                ang_delta,
                T_max,
                T_min,
                seed_base + n * 9999,
            )
            if score < best_score:
                best_score = score
                best_xs_group[:] = opt_xs[:]
                best_ys_group[:] = opt_ys[:]
                best_degs_group[:] = opt_degs[:]
            if score < physics_seed_cost:
                physics_seed_cost = score
                physics_seed_xs = opt_xs.copy()
                physics_seed_ys = opt_ys.copy()
                physics_seed_degs = opt_degs.copy()

        # ---------------------------------------------------------
        # Phase 3: Physics Squeeze (物理ベース重なり解消)
        # ---------------------------------------------------------
        # ここまでのベスト解をさらに強制圧縮して、壁抜けを試みる
        current_best_xs = physics_seed_xs.copy()
        current_best_ys = physics_seed_ys.copy()
        current_best_degs = physics_seed_degs.copy()

        for ratio in physics_ratios:
            sq_xs, sq_ys, sq_degs, success = optimize_overlap_resolution(
                current_best_xs,
                current_best_ys,
                current_best_degs,
                target_scale=ratio,
                n_iters=physics_iters,
                pos_delta=physics_pos_delta,
                ang_delta=physics_ang_delta,
                seed=seed_base + n * 777,
            )

            if success:
                # 圧縮成功時のみスコア計算（重い計算を避けるため）
                sq_verts = [get_tree_vertices(sq_xs[i], sq_ys[i], sq_degs[i]) for i in range(n)]
                sq_score = calculate_score(sq_verts)

                if sq_score < best_score:
                    # Physicsで記録更新
                    best_score = sq_score
                    best_xs_group[:] = sq_xs[:]
                    best_ys_group[:] = sq_ys[:]
                    best_degs_group[:] = sq_degs[:]
                    # さらに深い圧縮を試すためにループを継続するか、
                    # ここでbreakするかは戦略次第（今回は全探索）

        # ---------------------------------------------------------
        # Result Update
        # ---------------------------------------------------------
        if best_score < orig_score - 1e-9:
            improvement = orig_score - best_score
            total_improved += improvement
            improved_groups += 1

            new_xs[start_idx : start_idx + n] = best_xs_group
            new_ys[start_idx : start_idx + n] = best_ys_group
            new_degs[start_idx : start_idx + n] = best_degs_group

            print(f"  Group {n}: {orig_score:.6f} -> {best_score:.6f} (improved {improvement:.6f})")

    # 最終集計と保存
    final_score = calculate_total_score(new_xs, new_ys, new_degs)

    print("\nOptimization Summary")
    print(f"  Baseline total:    {baseline_total:.6f}")
    print(f"  After optimization: {final_score:.6f}")
    print(f"  Total improvement: {baseline_total - final_score:+.6f}")
    print(f"  Improved groups:   {improved_groups}")

    baseline_improved = final_score < baseline_total - 1e-9
    if baseline_improved:
        print("Baseline比較: 改善あり")
    else:
        print("Baseline比較: 改善なし")

    out_path = CONFIG["paths"]["output"]
    if os.path.exists(out_path):
        ref_xs, ref_ys, ref_degs = load_submission_data(out_path)
        ref_score = calculate_total_score(ref_xs, ref_ys, ref_degs)
        print(f"  既存submissionスコア: {ref_score:.6f}")
        should_save = final_score < ref_score - 1e-9
        no_save_reason = "submissionより改善なしのため上書きしません"
    else:
        should_save = baseline_improved
        no_save_reason = "Baselineから改善なしのためsubmissionを作成しません"

    if should_save:
        save_submission(out_path, new_xs, new_ys, new_degs)
        print(f"submissionを更新しました: {out_path}")
    else:
        print(no_save_reason)
