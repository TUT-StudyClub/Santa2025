"""
exp013_lns: 大規模近傍探索 (Large Neighborhood Search)

解の一部を破壊して再構築することで、
局所最適から脱出してより良い解を探索する。
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
    return os.path.join("experiments", "exp013_lns", "exp", f"{config_name}.yaml")


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
    min_x, min_y = vertices[0, 0], vertices[0, 1]
    max_x, max_y = vertices[0, 0], vertices[0, 1]
    for i in range(1, vertices.shape[0]):
        x, y = vertices[i, 0], vertices[i, 1]
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
    p1x: float, p1y: float, p2x: float, p2y: float,
    p3x: float, p3y: float, p4x: float, p4y: float,
) -> bool:
    d1x, d1y = p2x - p1x, p2y - p1y
    d2x, d2y = p4x - p3x, p4y - p3y
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
    n1, n2 = verts1.shape[0], verts2.shape[0]
    for i in range(n1):
        j = (i + 1) % n1
        for k in range(n2):
            m = (k + 1) % n2
            if segments_intersect(
                verts1[i, 0], verts1[i, 1], verts1[j, 0], verts1[j, 1],
                verts2[k, 0], verts2[k, 1], verts2[m, 0], verts2[m, 1],
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
def calculate_score(all_vertices: list[np.ndarray]) -> float:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    side = max(max_x - min_x, max_y - min_y)
    return side * side / len(all_vertices)


# -----------------------------------------------------------------------------
# LNS Operations
# -----------------------------------------------------------------------------
@njit(cache=True)
def destroy_random(
    xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, destroy_ratio: float
) -> np.ndarray:
    """ランダムに木を選んで破壊（位置をリセット）"""
    n = len(xs)
    n_destroy = max(1, int(n * destroy_ratio))
    destroyed = np.random.choice(n, size=n_destroy, replace=False)
    return destroyed


@njit(cache=True)
def destroy_worst(
    xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, destroy_ratio: float
) -> np.ndarray:
    """バウンディングボックスの境界に近い木を破壊"""
    n = len(xs)
    n_destroy = max(1, int(n * destroy_ratio))

    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)

    # 各木の境界への寄与度を計算
    contributions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        vx1, vy1, vx2, vy2 = polygon_bounds(all_vertices[i])
        # 境界に近いほど高いスコア
        contributions[i] = max(
            max_x - vx2,  # 右端からの距離
            vx1 - min_x,  # 左端からの距離
            max_y - vy2,  # 上端からの距離
            vy1 - min_y,  # 下端からの距離
        )
        contributions[i] = 1.0 / (contributions[i] + 0.01)  # 近いほど高い

    # 寄与度が高い（境界に近い）木を破壊
    sorted_indices = np.argsort(-contributions)  # 降順
    return sorted_indices[:n_destroy]


@njit(cache=True)
def is_in_array(val: int, arr: np.ndarray) -> bool:
    """配列に値が含まれているかチェック"""
    for i in range(len(arr)):
        if arr[i] == val:
            return True
    return False


@njit(cache=True)
def get_index_in_array(val: int, arr: np.ndarray) -> int:
    """配列内での値のインデックスを取得（見つからない場合は-1）"""
    for i in range(len(arr)):
        if arr[i] == val:
            return i
    return -1


@njit(cache=True)
def repair_greedy(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    destroyed: np.ndarray,
    n_candidates: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """破壊された木を貪欲に再配置"""
    n = len(xs)
    new_xs = xs.copy()
    new_ys = ys.copy()
    new_degs = degs.copy()

    # すべての木の頂点を計算
    all_vertices = [get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) for i in range(n)]

    # 破壊されていない木だけでバウンディングボックスを計算
    fixed_verts = [all_vertices[i] for i in range(n) if not is_in_array(i, destroyed)]
    if len(fixed_verts) > 0:
        min_x, min_y, max_x, max_y = compute_bounding_box(fixed_verts)
    else:
        min_x, min_y, max_x, max_y = -1.0, -1.0, 1.0, 1.0

    # 破壊された木を1つずつ配置
    for d_idx in range(len(destroyed)):
        tree_idx = destroyed[d_idx]
        best_score = math.inf
        best_x, best_y, best_deg = new_xs[tree_idx], new_ys[tree_idx], new_degs[tree_idx]

        # バウンディングボックス内でランダムに候補を生成
        for _ in range(n_candidates):
            cx = np.random.uniform(min_x, max_x)
            cy = np.random.uniform(min_y, max_y)
            angle = np.random.uniform(0, 360)

            new_verts = get_tree_vertices(cx, cy, angle)

            # 衝突チェック
            overlap = False
            for i in range(n):
                if i == tree_idx:
                    continue
                # 破壊されていない木、または既に配置した破壊された木
                idx_in_destroyed = get_index_in_array(i, destroyed)
                if idx_in_destroyed == -1 or idx_in_destroyed < d_idx:
                    if polygons_overlap(new_verts, all_vertices[i]):
                        overlap = True
                        break

            if not overlap:
                # スコアを計算
                temp_vertices = []
                for i in range(n):
                    idx_in_destroyed = get_index_in_array(i, destroyed)
                    if idx_in_destroyed == -1:  # 破壊されていない
                        temp_vertices.append(all_vertices[i])
                    elif idx_in_destroyed < d_idx:  # 既に配置した破壊された木
                        temp_vertices.append(all_vertices[i])
                temp_vertices.append(new_verts)
                score = calculate_score(temp_vertices)

                if score < best_score:
                    best_score = score
                    best_x, best_y, best_deg = cx, cy, angle

        new_xs[tree_idx] = best_x
        new_ys[tree_idx] = best_y
        new_degs[tree_idx] = best_deg
        all_vertices[tree_idx] = get_tree_vertices(best_x, best_y, best_deg)

    # 最終的な重なりチェック
    final_vertices = [get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) for i in range(n)]
    valid = not has_any_overlap(final_vertices)

    return new_xs, new_ys, new_degs, valid


@njit(cache=True)
def repair_sa(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    destroyed: np.ndarray,
    T_max: float,
    T_min: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """破壊された木をSAで再配置"""
    n = len(xs)
    new_xs = xs.copy()
    new_ys = ys.copy()
    new_degs = degs.copy()

    # まず貪欲に初期配置
    new_xs, new_ys, new_degs, _ = repair_greedy(new_xs, new_ys, new_degs, destroyed, 100)

    all_vertices = [get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) for i in range(n)]

    if has_any_overlap(all_vertices):
        return new_xs, new_ys, new_degs, False

    best_xs = new_xs.copy()
    best_ys = new_ys.copy()
    best_degs = new_degs.copy()
    best_score = calculate_score(all_vertices)

    T = T_max
    T_decay = (T_min / T_max) ** (1.0 / n_steps)

    for _ in range(n_steps):
        # 破壊された木からランダムに選択
        tree_idx = destroyed[np.random.randint(0, len(destroyed))]

        old_x, old_y, old_deg = new_xs[tree_idx], new_ys[tree_idx], new_degs[tree_idx]
        old_verts = all_vertices[tree_idx]

        # 摂動
        move_type = np.random.randint(0, 3)
        if move_type == 0:
            new_xs[tree_idx] += (np.random.random() * 2 - 1) * 0.05
        elif move_type == 1:
            new_ys[tree_idx] += (np.random.random() * 2 - 1) * 0.05
        else:
            new_degs[tree_idx] = (new_degs[tree_idx] + (np.random.random() * 2 - 1) * 10) % 360.0

        new_verts = get_tree_vertices(new_xs[tree_idx], new_ys[tree_idx], new_degs[tree_idx])

        # 衝突チェック
        overlap = False
        for j in range(n):
            if j != tree_idx and polygons_overlap(new_verts, all_vertices[j]):
                overlap = True
                break

        if overlap:
            new_xs[tree_idx], new_ys[tree_idx], new_degs[tree_idx] = old_x, old_y, old_deg
            continue

        all_vertices[tree_idx] = new_verts
        new_score = calculate_score(all_vertices)
        delta = new_score - best_score

        if delta < 0 or np.random.random() < math.exp(-delta / T):
            if new_score < best_score:
                best_xs = new_xs.copy()
                best_ys = new_ys.copy()
                best_degs = new_degs.copy()
                best_score = new_score
        else:
            new_xs[tree_idx], new_ys[tree_idx], new_degs[tree_idx] = old_x, old_y, old_deg
            all_vertices[tree_idx] = old_verts

        T *= T_decay

    return best_xs, best_ys, best_degs, True


@njit(cache=True)
def lns_optimize(
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    n_iterations: int,
    destroy_ratio: float,
    T_max: float,
    T_min: float,
    repair_steps: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """LNSメインループ"""
    np.random.seed(random_seed)
    n = len(init_xs)

    best_xs = init_xs.copy()
    best_ys = init_ys.copy()
    best_degs = init_degs.copy()

    all_vertices = [get_tree_vertices(best_xs[i], best_ys[i], best_degs[i]) for i in range(n)]
    if has_any_overlap(all_vertices):
        return best_xs, best_ys, best_degs, math.inf

    best_score = calculate_score(all_vertices)

    for iteration in range(n_iterations):
        # 破壊
        if np.random.random() < 0.5:
            destroyed = destroy_random(best_xs, best_ys, best_degs, destroy_ratio)
        else:
            destroyed = destroy_worst(best_xs, best_ys, best_degs, destroy_ratio)

        # 修復
        new_xs, new_ys, new_degs, valid = repair_sa(
            best_xs, best_ys, best_degs, destroyed, T_max, T_min, repair_steps
        )

        if valid:
            new_verts = [get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) for i in range(n)]
            if not has_any_overlap(new_verts):
                new_score = calculate_score(new_verts)
                if new_score < best_score:
                    best_xs = new_xs.copy()
                    best_ys = new_ys.copy()
                    best_degs = new_degs.copy()
                    best_score = new_score

    return best_xs, best_ys, best_degs, best_score


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


def save_submission(
    filepath: str, all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray
) -> None:
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
    for n in range(1, 201):
        start = n * (n - 1) // 2
        vertices = [
            get_tree_vertices(all_xs[start + i], all_ys[start + i], all_degs[start + i])
            for i in range(n)
        ]
        total += calculate_score(vertices)
    return total


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Large Neighborhood Search (LNS) Optimizer (exp013_lns)")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nBaseline: {baseline_path}")

    all_xs, all_ys, all_degs = load_submission_data(baseline_path)
    baseline_total = calculate_total_score(all_xs, all_ys, all_degs)
    print(f"Baseline total score: {baseline_total:.6f}")

    # パラメータ
    lns_cfg = CONFIG["lns"]
    n_min = int(lns_cfg["n_min"])
    n_max = int(lns_cfg["n_max"])
    n_iterations = int(lns_cfg["n_iterations"])
    destroy_ratio = float(lns_cfg["destroy_ratio"])
    T_max = float(lns_cfg["T_max"])
    T_min = float(lns_cfg["T_min"])
    repair_steps = int(lns_cfg["repair_steps"])
    seed_base = int(lns_cfg.get("seed_base", 42))

    print(f"\nOptimizing groups {n_min} to {n_max}...")
    print(f"  LNS iterations: {n_iterations}")
    print(f"  Destroy ratio: {destroy_ratio}")

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    total_improved = 0.0
    improved_groups = 0

    for n in tqdm(range(n_min, n_max + 1), desc="Optimizing"):
        start = n * (n - 1) // 2

        # 現在のスコア
        orig_verts = [
            get_tree_vertices(new_xs[start + i], new_ys[start + i], new_degs[start + i])
            for i in range(n)
        ]
        orig_score = calculate_score(orig_verts)

        # LNS最適化
        init_xs = new_xs[start:start + n].copy()
        init_ys = new_ys[start:start + n].copy()
        init_degs = new_degs[start:start + n].copy()

        seed = seed_base + n * 1000
        opt_xs, opt_ys, opt_degs, opt_score = lns_optimize(
            init_xs, init_ys, init_degs,
            n_iterations, destroy_ratio, T_max, T_min, repair_steps, seed
        )

        if opt_score < orig_score - 1e-9:
            improvement = orig_score - opt_score
            total_improved += improvement
            improved_groups += 1
            new_xs[start:start + n] = opt_xs
            new_ys[start:start + n] = opt_ys
            new_degs[start:start + n] = opt_degs
            print(f"  Group {n}: {orig_score:.6f} -> {opt_score:.6f} (improved {improvement:.6f})")

    final_score = calculate_total_score(new_xs, new_ys, new_degs)

    print("\n" + "=" * 80)
    print(f"  Baseline total:    {baseline_total:.6f}")
    print(f"  After LNS:         {final_score:.6f}")
    print(f"  Total improvement: {baseline_total - final_score:+.6f}")
    print(f"  Improved groups:   {improved_groups}")
    print("=" * 80)

    if final_score < baseline_total:
        out_path = CONFIG["paths"]["output"]
        save_submission(out_path, new_xs, new_ys, new_degs)
        print(f"Saved to {out_path}")
    else:
        print("No improvement - keeping baseline")

