import math
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import yaml
from numba import njit

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
with open("experiments/exp004_baseline/exp/001.yaml") as f:
    CONFIG = yaml.safe_load(f)

TREE_CFG = CONFIG["tree_shape"]
TRUNK_W = float(TREE_CFG["trunk_w"])
TRUNK_H = float(TREE_CFG["trunk_h"])
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
    """
    座標を回転させる
    params:
      - x: 元のX座標
      - y: 元のY座標
      - cos_a: 回転角のcos値
      - sin_a: 回転角のsin値
    """
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


@njit(cache=True)
def get_tree_vertices(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    """
    指定された位置と角度における木の15個の頂点座標を取得
    params:
      - cx: 中心X座標
      - cy: 中心Y座標
      - angle_deg: 回転角度（度数法）
    """
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
    """
    多角形のバウンディングボックスを取得
    params:
      - vertices: 多角形の頂点配列 (N, 2)
    """
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
    """
    レイキャスティング法による点内包判定
    params:
      - px: 点のX座標
      - py: 点のY座標
      - vertices: 多角形の頂点配列
    """
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
    """
    2つの線分が交差しているか判定
    params:
      - p1x, p1y: 線分1の始点
      - p2x, p2y: 線分1の終点
      - p3x, p3y: 線分2の始点
      - p4x, p4y: 線分2の終点
    """
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
    """
    2つの多角形が重なっているか判定
    params:
      - verts1: 多角形1の頂点配列
      - verts2: 多角形2の頂点配列
    """
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
def has_any_overlap(all_vertices: list[np.ndarray]) -> bool:
    """
    全ての多角形のペアについて重なりをチェック
    params:
      - all_vertices: 全多角形の頂点リスト
    """
    n = len(all_vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(all_vertices[i], all_vertices[j]):
                return True
    return False


@njit(cache=True)
def compute_bounding_box(all_vertices: list[np.ndarray]) -> tuple[float, float, float, float]:
    """
    全体のバウンディングボックスを計算
    params:
      - all_vertices: 全多角形の頂点リスト
    """
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for verts in all_vertices:
        x1, y1, x2, y2 = polygon_bounds(verts)
        if x1 < min_x:
            min_x = x1
        if y1 < min_y:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def get_side_length(all_vertices: list[np.ndarray]) -> float:
    """
    バウンディングボックスの長辺（正方形の辺の長さ）を取得
    params:
      - all_vertices: 全多角形の頂点リスト
    """
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True)
def calculate_score_numba(all_vertices: list[np.ndarray]) -> float:
    """
    スコア計算: max(width, height)^2 / n
    params:
      - all_vertices: 全多角形の頂点リスト
    """
    side = get_side_length(all_vertices)
    return side * side / len(all_vertices)


# -----------------------------------------------------------------------------
# Grid Generation
# -----------------------------------------------------------------------------
@njit(cache=True)
def create_grid_vertices_extended(
    seed_xs: np.ndarray,
    seed_ys: np.ndarray,
    seed_degs: np.ndarray,
    a: float,
    b: float,
    ncols: int,
    nrows: int,
    append_x: bool,
    append_y: bool,
) -> list[np.ndarray]:
    """
    並進によるグリッド配置を生成（端の追加配置オプション付き）
    params:
      - seed_xs, seed_ys, seed_degs: シード（初期配置）のパラメータ
      - a, b: グリッドの間隔
      - ncols, nrows: グリッドの列数・行数
      - append_x, append_y: 端に追加配置を行うかのフラグ
    """
    n_seeds = len(seed_xs)
    all_vertices = []

    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                cx = seed_xs[s] + col * a
                cy = seed_ys[s] + row * b
                all_vertices.append(get_tree_vertices(cx, cy, seed_degs[s]))

    if append_x and n_seeds > 1:
        for row in range(nrows):
            cx = seed_xs[1] + ncols * a
            cy = seed_ys[1] + row * b
            all_vertices.append(get_tree_vertices(cx, cy, seed_degs[1]))

    if append_y and n_seeds > 1:
        for col in range(ncols):
            cx = seed_xs[1] + col * a
            cy = seed_ys[1] + nrows * b
            all_vertices.append(get_tree_vertices(cx, cy, seed_degs[1]))

    return all_vertices


@njit(cache=True)
def get_initial_translations(
    seed_xs: np.ndarray, seed_ys: np.ndarray, seed_degs: np.ndarray
) -> tuple[float, float]:
    """
    シードのバウンディングボックスから初期並進距離を算出
    params:
      - seed_xs, seed_ys, seed_degs: シードのパラメータ
    """
    seed_vertices = [
        get_tree_vertices(seed_xs[i], seed_ys[i], seed_degs[i]) for i in range(len(seed_xs))
    ]
    min_x, min_y, max_x, max_y = compute_bounding_box(seed_vertices)
    return max_x - min_x, max_y - min_y


# -----------------------------------------------------------------------------
# Optimization Logic (Simulated Annealing)
# -----------------------------------------------------------------------------
@njit(cache=True)
def sa_optimize_improved(
    seed_xs_init: np.ndarray,
    seed_ys_init: np.ndarray,
    seed_degs_init: np.ndarray,
    a_init: float,
    b_init: float,
    ncols: int,
    nrows: int,
    append_x: bool,
    append_y: bool,
    Tmax: float,
    Tmin: float,
    nsteps: int,
    nsteps_per_T: int,
    position_delta: float,
    angle_delta: float,
    angle_delta2: float,
    delta_t: float,
    random_seed: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    SAによる最適化を実行
    params:
      - seed_*_init: 初期シードパラメータ
      - a_init, b_init: 初期グリッド間隔
      - ncols, nrows, append_*: グリッド構成設定
      - Tmax, Tmin, nsteps_*: SAのハイパーパラメータ
      - *_delta: 探索のステップサイズ
      - random_seed: 乱数シード
    """
    np.random.seed(random_seed)
    n_seeds = len(seed_xs_init)

    seed_xs = seed_xs_init.copy()
    seed_ys = seed_ys_init.copy()
    seed_degs = seed_degs_init.copy()
    a, b = a_init, b_init

    all_vertices = create_grid_vertices_extended(
        seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y
    )
    if has_any_overlap(all_vertices):
        a_test, b_test = get_initial_translations(seed_xs, seed_ys, seed_degs)
        a = max(a, a_test * 1.5)
        b = max(b, b_test * 1.5)
        all_vertices = create_grid_vertices_extended(
            seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y
        )

    current_score = calculate_score_numba(all_vertices)
    best_score = current_score
    best_xs, best_ys, best_degs = seed_xs.copy(), seed_ys.copy(), seed_degs.copy()
    best_a, best_b = a, b

    T = Tmax
    Tfactor = -math.log(Tmax / Tmin)
    n_move_types = n_seeds + 2

    # Variable initialization for Numba typing
    old_x, old_y, old_deg = 0.0, 0.0, 0.0
    old_a, old_b = 0.0, 0.0

    patience = nsteps // 5
    no_improve_count = 0
    last_best_score = best_score

    for step in range(nsteps):
        progress = step / nsteps
        decay = 1.0 - 0.9 * progress
        cur_pos_delta = position_delta * decay
        cur_ang_delta = angle_delta * decay

        for _ in range(nsteps_per_T):
            move_type = np.random.randint(0, n_move_types)

            # Move logic
            if move_type < n_seeds:
                i = move_type
                old_x, old_y, old_deg = seed_xs[i], seed_ys[i], seed_degs[i]

                dx = (np.random.random() * 2.0 - 1.0) * cur_pos_delta
                dy = (np.random.random() * 2.0 - 1.0) * cur_pos_delta
                ddeg = (np.random.random() * 2.0 - 1.0) * cur_ang_delta

                seed_xs[i] += dx
                seed_ys[i] += dy
                seed_degs[i] = (seed_degs[i] + ddeg) % 360.0

            elif move_type == n_seeds:
                old_a, old_b = a, b
                da = (np.random.random() * 2.0 - 1.0) * delta_t
                db = (np.random.random() * 2.0 - 1.0) * delta_t
                a += a * da
                b += b * db

            else:
                old_degs_array = seed_degs.copy()
                ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta2
                for k in range(n_seeds):
                    seed_degs[k] = (seed_degs[k] + ddeg) % 360.0

            # Constraint Check & Revert

            # Check 1: Simple 2x2 grid check
            test_vertices = create_grid_vertices_extended(
                seed_xs, seed_ys, seed_degs, a, b, 2, 2, False, False
            )
            if has_any_overlap(test_vertices):
                if move_type < n_seeds:
                    seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = (
                        old_x,
                        old_y,
                        old_deg,
                    )
                elif move_type == n_seeds:
                    a, b = old_a, old_b
                else:
                    seed_degs[:] = old_degs_array[:]
                continue

            # Check 2: Full grid check
            new_vertices = create_grid_vertices_extended(
                seed_xs, seed_ys, seed_degs, a, b, ncols, nrows, append_x, append_y
            )
            if has_any_overlap(new_vertices):
                if move_type < n_seeds:
                    seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = (
                        old_x,
                        old_y,
                        old_deg,
                    )
                elif move_type == n_seeds:
                    a, b = old_a, old_b
                else:
                    seed_degs[:] = old_degs_array[:]
                continue

            # Score Calculation
            new_score = calculate_score_numba(new_vertices)
            delta = new_score - current_score
            accept = False

            if delta < 0:
                accept = True
            elif T > 1e-10 and np.random.random() < math.exp(-delta / T):
                accept = True

            if accept:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_xs, best_ys, best_degs = seed_xs.copy(), seed_ys.copy(), seed_degs.copy()
                    best_a, best_b = a, b
            else:
                if move_type < n_seeds:
                    seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = (
                        old_x,
                        old_y,
                        old_deg,
                    )
                elif move_type == n_seeds:
                    a, b = old_a, old_b
                else:
                    seed_degs[:] = old_degs_array[:]

        # Early Stopping
        if last_best_score - best_score > 1e-9:
            no_improve_count = 0
            last_best_score = best_score
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

        T = Tmax * math.exp(Tfactor * (step + 1) / nsteps)

    return best_score, best_xs, best_ys, best_degs, best_a, best_b


@njit(cache=True)
def get_final_grid_positions_extended(
    seed_xs: np.ndarray,
    seed_ys: np.ndarray,
    seed_degs: np.ndarray,
    a: float,
    b: float,
    ncols: int,
    nrows: int,
    append_x: bool,
    append_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    最適化結果から最終的な全ツリーの配置座標を生成
    params:
      - seed_*: 最適化されたシードパラメータ
      - a, b: 最適化されたグリッド間隔
      - ncols, nrows, append_*: グリッド構成
    """
    n_seeds = len(seed_xs)
    n_base = n_seeds * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_total = n_base + n_append_x + n_append_y

    xs = np.empty(n_total, dtype=np.float64)
    ys = np.empty(n_total, dtype=np.float64)
    degs = np.empty(n_total, dtype=np.float64)

    idx = 0
    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                xs[idx] = seed_xs[s] + col * a
                ys[idx] = seed_ys[s] + row * b
                degs[idx] = seed_degs[s]
                idx += 1

    if append_x and n_seeds > 1:
        for row in range(nrows):
            xs[idx] = seed_xs[1] + ncols * a
            ys[idx] = seed_ys[1] + row * b
            degs[idx] = seed_degs[1]
            idx += 1

    if append_y and n_seeds > 1:
        for col in range(ncols):
            xs[idx] = seed_xs[1] + col * a
            ys[idx] = seed_ys[1] + nrows * b
            degs[idx] = seed_degs[1]
            idx += 1

    return xs, ys, degs


@njit(cache=True)
def deletion_cascade_numba(
    all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    大きな構成から木を1つずつ削除し、より小さな構成での最適解を探索する
    params:
      - all_xs, all_ys, all_degs: 全ツリーの初期座標と角度
    """
    group_start = np.zeros(201, dtype=np.int64)
    for n in range(1, 201):
        group_start[n] = group_start[n - 1] + (n - 1) if n > 1 else 0

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    side_lengths = np.zeros(201, dtype=np.float64)
    for n in range(1, 201):
        start = group_start[n]
        vertices = [
            get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) for i in range(start, start + n)
        ]
        side_lengths[n] = get_side_length(vertices)

    for n in range(200, 1, -1):
        start_n = group_start[n]
        start_prev = group_start[n - 1]
        best_side = side_lengths[n - 1]
        best_delete_idx = -1

        for del_idx in range(n):
            vertices = []
            for i in range(n):
                if i != del_idx:
                    idx = start_n + i
                    vertices.append(get_tree_vertices(new_xs[idx], new_ys[idx], new_degs[idx]))

            candidate_side = get_side_length(vertices)
            if candidate_side < best_side:
                best_side = candidate_side
                best_delete_idx = del_idx

        if best_delete_idx >= 0:
            out_idx = start_prev
            for i in range(n):
                if i != best_delete_idx:
                    in_idx = start_n + i
                    new_xs[out_idx] = new_xs[in_idx]
                    new_ys[out_idx] = new_ys[in_idx]
                    new_degs[out_idx] = new_degs[in_idx]
                    out_idx += 1
            side_lengths[n - 1] = best_side

    return new_xs, new_ys, new_degs, side_lengths


# -----------------------------------------------------------------------------
# Data & Worker Handlers
# -----------------------------------------------------------------------------
def optimize_grid_config(args: tuple) -> tuple[int, float, list[tuple[float, float, float]]]:
    """
    並列処理用ワーカ関数
    params:
      - args: 各種設定パラメータを含むタプル
    """
    ncols, nrows, append_x, append_y, initial_seeds, a_init, b_init, params, seed = args

    seed_xs = np.array([s[0] for s in initial_seeds], dtype=np.float64)
    seed_ys = np.array([s[1] for s in initial_seeds], dtype=np.float64)
    seed_degs = np.array([s[2] for s in initial_seeds], dtype=np.float64)

    n_trees = (
        len(initial_seeds) * ncols * nrows + (nrows if append_x else 0) + (ncols if append_y else 0)
    )

    best_score, best_xs, best_ys, best_degs, best_a, best_b = sa_optimize_improved(
        seed_xs,
        seed_ys,
        seed_degs,
        a_init,
        b_init,
        ncols,
        nrows,
        append_x,
        append_y,
        params["Tmax"],
        params["Tmin"],
        params["nsteps"],
        params["nsteps_per_T"],
        params["position_delta"],
        params["angle_delta"],
        params["angle_delta2"],
        params["delta_t"],
        seed,
    )

    final_xs, final_ys, final_degs = get_final_grid_positions_extended(
        best_xs, best_ys, best_degs, best_a, best_b, ncols, nrows, append_x, append_y
    )

    tree_data = [(final_xs[i], final_ys[i], final_degs[i]) for i in range(len(final_xs))]
    return n_trees, best_score, tree_data


def load_submission_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提出形式のCSVをロード
    params:
      - filepath: CSVファイルのパス
    """
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


def save_submission(
    filepath: str, all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray
) -> None:
    """
    結果をCSVに保存
    params:
      - filepath: 保存先のパス
      - all_xs, all_ys, all_degs: 結果データ
    """
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
    """
    全レベルのトータルスコアを計算
    params:
      - all_xs, all_ys, all_degs: 全ツリーのデータ
    """
    total = 0.0
    idx = 0
    for n in range(1, 201):
        vertices = [
            get_tree_vertices(all_xs[idx + i], all_ys[idx + i], all_degs[idx + i]) for i in range(n)
        ]
        total += calculate_score_numba(vertices)
        idx += n
    return total


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Improved SA Translation Optimizer (Numba-accelerated)")
    print("=" * 80)

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nBaseline: {baseline_path}")

    baseline_xs, baseline_ys, baseline_degs = load_submission_data(baseline_path)
    baseline_total = calculate_total_score(baseline_xs, baseline_ys, baseline_degs)
    print(f"Baseline total score: {baseline_total:.6f}")

    # Generate Grid Configurations
    grid_cfg = CONFIG["grid_search"]
    grid_configs = []

    for ncols in range(grid_cfg["col_min"], grid_cfg["col_max"]):
        for nrows in range(ncols, grid_cfg["row_max_limit"]):
            n_trees = 2 * ncols * nrows
            max_t = grid_cfg["max_trees"]

            if 20 <= n_trees <= max_t:
                if (ncols, nrows, False, False) not in grid_configs:
                    grid_configs.append((ncols, nrows, False, False))
                if n_trees + ncols <= max_t:
                    grid_configs.append((ncols, nrows, False, True))
                if n_trees + nrows <= max_t:
                    grid_configs.append((ncols, nrows, True, False))

    grid_configs = sorted(
        list(set(grid_configs)),
        key=lambda x: (2 * x[0] * x[1] + (x[1] if x[2] else 0) + (x[0] if x[3] else 0)),
    )
    print(f"Generated {len(grid_configs)} grid configurations")

    # Prepare Tasks
    tasks = []
    init_state = CONFIG["initial_state"]
    seeds = init_state["seeds"]
    a_init = init_state["translation_a"]
    b_init = init_state["translation_b"]
    sa_params = CONFIG["sa_params"]

    for i, (ncols, nrows, append_x, append_y) in enumerate(grid_configs):
        n_trees = 2 * ncols * nrows + (nrows if append_x else 0) + (ncols if append_y else 0)
        if n_trees > 200:
            continue
        seed = sa_params["random_seed_base"] + i * 1000
        tasks.append((ncols, nrows, append_x, append_y, seeds, a_init, b_init, sa_params, seed))

    # Execute Parallel Optimization
    print(f"Running SA optimization on {len(tasks)} configurations...")
    num_workers = min(cpu_count(), len(tasks))
    t0 = time.time()

    with Pool(num_workers) as pool:
        results = pool.map(optimize_grid_config, tasks)

    print(f"SA optimization completed in {time.time() - t0:.1f}s")

    # Process Results
    new_trees = {}
    improved_count = 0
    for n_trees, score, tree_data in results:
        idx = sum(range(1, n_trees))
        baseline_vertices = [
            get_tree_vertices(baseline_xs[idx + i], baseline_ys[idx + i], baseline_degs[idx + i])
            for i in range(n_trees)
        ]
        baseline_score = calculate_score_numba(baseline_vertices)

        if score < baseline_score:
            new_trees[n_trees] = tree_data
            if baseline_score - score > 1e-6:
                improved_count += 1
                print(f"  n={n_trees}: {score:.6f} (baseline: {baseline_score:.6f})")

    # Merge with Baseline
    print("Merging with baseline...")
    merged_xs = baseline_xs.copy()
    merged_ys = baseline_ys.copy()
    merged_degs = baseline_degs.copy()

    for n_trees, tree_data in new_trees.items():
        idx = sum(range(1, n_trees))
        for i in range(n_trees):
            merged_xs[idx + i] = tree_data[i][0]
            merged_ys[idx + i] = tree_data[i][1]
            merged_degs[idx + i] = tree_data[i][2]

    # Deletion Cascade
    print("Applying tree deletion cascade...")
    final_xs, final_ys, final_degs, _ = deletion_cascade_numba(merged_xs, merged_ys, merged_degs)

    final_score = calculate_total_score(final_xs, final_ys, final_degs)
    print("=" * 80)
    print(f"  Baseline total:      {baseline_total:.6f}")
    print(f"  After cascade:       {final_score:.6f}")
    print(f"  Total improvement:   {baseline_total - final_score:+.6f}")
    print("=" * 80)

    # Save Results
    if final_score < baseline_total:
        out_path = CONFIG["paths"]["output"]
        save_submission(out_path, final_xs, final_ys, final_degs)
        print(f"Saved to {out_path}")

        script_path = CONFIG["paths"]["overlap_script"]
        if os.path.exists(script_path):
            cmd = f"python {script_path} {baseline_path} {out_path}"
            os.system(cmd)
    else:
        print("No improvement - keeping baseline")
