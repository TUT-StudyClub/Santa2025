"""
exp009_seed_search: 異なるシード配置の探索

様々なシード角度の組み合わせを試して、より良い配置を見つける。
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
    return os.path.join("experiments", "exp009_seed_search", "exp", f"{config_name}.yaml")


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
def has_any_overlap(all_vertices: list[np.ndarray]) -> bool:
    n = len(all_vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(all_vertices[i], all_vertices[j]):
                return True
    return False


@njit(cache=True)
def compute_bounding_box(all_vertices: list[np.ndarray]) -> tuple[float, float, float, float]:
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
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True)
def calculate_score(all_vertices: list[np.ndarray]) -> float:
    side = get_side_length(all_vertices)
    return side * side / len(all_vertices)


# -----------------------------------------------------------------------------
# Grid Generation & Optimization
# -----------------------------------------------------------------------------
@njit(cache=True)
def create_grid_vertices(
    seed_xs: np.ndarray,
    seed_ys: np.ndarray,
    seed_degs: np.ndarray,
    a: float,
    b: float,
    ncols: int,
    nrows: int,
) -> list[np.ndarray]:
    n_seeds = len(seed_xs)
    all_vertices = []

    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                cx = seed_xs[s] + col * a
                cy = seed_ys[s] + row * b
                all_vertices.append(get_tree_vertices(cx, cy, seed_degs[s]))

    return all_vertices


@njit(cache=True)
def sa_optimize_grid(
    seed_xs_init: np.ndarray,
    seed_ys_init: np.ndarray,
    seed_degs_init: np.ndarray,
    a_init: float,
    b_init: float,
    ncols: int,
    nrows: int,
    Tmax: float,
    Tmin: float,
    nsteps: int,
    nsteps_per_T: int,
    position_delta: float,
    angle_delta: float,
    delta_t: float,
    random_seed: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float]:
    np.random.seed(random_seed)
    n_seeds = len(seed_xs_init)

    seed_xs = seed_xs_init.copy()
    seed_ys = seed_ys_init.copy()
    seed_degs = seed_degs_init.copy()
    a, b = a_init, b_init

    all_vertices = create_grid_vertices(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows)
    
    # 初期状態で重なりがある場合は間隔を広げる
    while has_any_overlap(all_vertices):
        a *= 1.1
        b *= 1.1
        all_vertices = create_grid_vertices(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows)

    current_score = calculate_score(all_vertices)
    best_score = current_score
    best_xs, best_ys, best_degs = seed_xs.copy(), seed_ys.copy(), seed_degs.copy()
    best_a, best_b = a, b

    T = Tmax
    Tfactor = -math.log(Tmax / Tmin)
    n_move_types = n_seeds + 1

    old_x, old_y, old_deg = 0.0, 0.0, 0.0
    old_a, old_b = 0.0, 0.0

    for step in range(nsteps):
        progress = step / nsteps
        decay = 1.0 - 0.8 * progress
        cur_pos_delta = position_delta * decay
        cur_ang_delta = angle_delta * decay

        for _ in range(nsteps_per_T):
            move_type = np.random.randint(0, n_move_types)

            if move_type < n_seeds:
                i = move_type
                old_x, old_y, old_deg = seed_xs[i], seed_ys[i], seed_degs[i]
                dx = (np.random.random() * 2.0 - 1.0) * cur_pos_delta
                dy = (np.random.random() * 2.0 - 1.0) * cur_pos_delta
                ddeg = (np.random.random() * 2.0 - 1.0) * cur_ang_delta
                seed_xs[i] += dx
                seed_ys[i] += dy
                seed_degs[i] = (seed_degs[i] + ddeg) % 360.0
            else:
                old_a, old_b = a, b
                da = (np.random.random() * 2.0 - 1.0) * delta_t
                db = (np.random.random() * 2.0 - 1.0) * delta_t
                a += a * da
                b += b * db

            new_vertices = create_grid_vertices(seed_xs, seed_ys, seed_degs, a, b, ncols, nrows)
            if has_any_overlap(new_vertices):
                if move_type < n_seeds:
                    seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = old_x, old_y, old_deg
                else:
                    a, b = old_a, old_b
                continue

            new_score = calculate_score(new_vertices)
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
                    seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = old_x, old_y, old_deg
                else:
                    a, b = old_a, old_b

        T = Tmax * math.exp(Tfactor * (step + 1) / nsteps)

    return best_score, best_xs, best_ys, best_degs, best_a, best_b


@njit(cache=True)
def get_final_positions(
    seed_xs: np.ndarray,
    seed_ys: np.ndarray,
    seed_degs: np.ndarray,
    a: float,
    b: float,
    ncols: int,
    nrows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_seeds = len(seed_xs)
    n_total = n_seeds * ncols * nrows

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

    return xs, ys, degs


@njit(cache=True)
def deletion_cascade(
    all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    for n in range(200, 1, -1):
        start_n = n * (n - 1) // 2
        start_prev = (n - 1) * (n - 2) // 2

        # 現在のn-1グループのサイドを計算
        prev_verts = [get_tree_vertices(new_xs[start_prev + i], new_ys[start_prev + i], new_degs[start_prev + i]) for i in range(n - 1)]
        best_side = get_side_length(prev_verts)
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

    return new_xs, new_ys, new_degs


# -----------------------------------------------------------------------------
# Worker Function
# -----------------------------------------------------------------------------
def optimize_seed_config(args: tuple) -> tuple[float, list, float, float]:
    deg1, deg2, sa_params, seed = args
    
    # 2シードの初期配置
    seed_xs = np.array([0.0, 0.4], dtype=np.float64)
    seed_ys = np.array([0.0, 0.35], dtype=np.float64)
    seed_degs = np.array([deg1, deg2], dtype=np.float64)
    
    # 初期間隔
    a_init = 0.85
    b_init = 0.75
    
    # グリッドサイズ（200本に近い構成）
    ncols = 10
    nrows = 10
    
    score, opt_xs, opt_ys, opt_degs, opt_a, opt_b = sa_optimize_grid(
        seed_xs, seed_ys, seed_degs,
        a_init, b_init,
        ncols, nrows,
        sa_params["Tmax"],
        sa_params["Tmin"],
        sa_params["nsteps"],
        sa_params["nsteps_per_T"],
        sa_params["position_delta"],
        sa_params["angle_delta"],
        sa_params["delta_t"],
        seed,
    )
    
    return score, [(opt_xs[i], opt_ys[i], opt_degs[i]) for i in range(len(opt_xs))], opt_a, opt_b


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
            rows.append({
                "id": f"{n:03d}_{t}",
                "x": f"s{all_xs[idx]}",
                "y": f"s{all_ys[idx]}",
                "deg": f"s{all_degs[idx]}",
            })
            idx += 1
    pd.DataFrame(rows).to_csv(filepath, index=False)


def calculate_total_score(all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray) -> float:
    total = 0.0
    idx = 0
    for n in range(1, 201):
        vertices = [get_tree_vertices(all_xs[idx + i], all_ys[idx + i], all_degs[idx + i]) for i in range(n)]
        total += calculate_score(vertices)
        idx += n
    return total


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Seed Configuration Search (exp009_seed_search)")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nBaseline: {baseline_path}")

    baseline_xs, baseline_ys, baseline_degs = load_submission_data(baseline_path)
    baseline_total = calculate_total_score(baseline_xs, baseline_ys, baseline_degs)
    print(f"Baseline total score: {baseline_total:.6f}")

    # シード角度の探索範囲
    search_cfg = CONFIG["seed_search"]
    angle_step = float(search_cfg["angle_step"])
    angle_min = float(search_cfg.get("angle_min", 0))
    angle_max = float(search_cfg.get("angle_max", 180))
    
    sa_params = CONFIG["sa_params"]
    seed_base = int(sa_params.get("random_seed_base", 42))

    # 角度の組み合わせを生成
    angles = np.arange(angle_min, angle_max, angle_step)
    tasks = []
    
    for i, deg1 in enumerate(angles):
        for j, deg2 in enumerate(angles):
            if deg2 >= deg1 + 90:  # 2つの角度の差が90度以上
                seed = seed_base + i * 1000 + j
                tasks.append((deg1, deg2 + 180, sa_params, seed))  # deg2に180度加えて反対向きに

    print(f"\nSearching {len(tasks)} seed configurations...")
    print(f"  Angle step: {angle_step}°")
    print(f"  Angle range: {angle_min}° to {angle_max}°")

    # 並列実行
    num_workers = min(cpu_count(), len(tasks))
    t0 = time.time()

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(optimize_seed_config, tasks), total=len(tasks), desc="Searching"):
            results.append(result)

    print(f"\nSearch completed in {time.time() - t0:.1f}s")

    # 最良結果を選択
    best_result = min(results, key=lambda x: x[0])
    best_score, best_seeds, best_a, best_b = best_result

    print(f"\nBest seed configuration:")
    print(f"  Score (200 trees): {best_score:.6f}")
    print(f"  Grid spacing: a={best_a:.4f}, b={best_b:.4f}")
    for i, (x, y, deg) in enumerate(best_seeds[:2]):
        print(f"  Seed {i}: x={x:.4f}, y={y:.4f}, deg={deg:.2f}")

    # この結果を使って全グループを最適化するかどうか
    if best_score < 0.35:  # 良い結果が見つかった場合
        print("\nGood seed configuration found! Use this for full optimization.")
        print(f"Suggested initial_state:")
        print(f"  seeds:")
        for x, y, deg in best_seeds[:2]:
            print(f"    - [{x}, {y}, {deg}]")
        print(f"  translation_a: {best_a}")
        print(f"  translation_b: {best_b}")

