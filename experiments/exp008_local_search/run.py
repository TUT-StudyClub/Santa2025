"""
exp008_local_search: 個別ツリーの局所最適化

既存の配置を読み込み、各グループごとに個々のツリーの位置・角度を
直接最適化してスコアを改善する。
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
    return os.path.join("experiments", "exp008_local_search", "exp", f"{config_name}.yaml")


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
def has_any_overlap_with_others(
    tree_idx: int, new_verts: np.ndarray, all_vertices: list[np.ndarray]
) -> bool:
    """指定したツリーが他のツリーと重なるかチェック"""
    n = len(all_vertices)
    for j in range(n):
        if j != tree_idx:
            if polygons_overlap(new_verts, all_vertices[j]):
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
# Local Search Optimization
# -----------------------------------------------------------------------------
@njit(cache=True)
def local_search_group(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    グループ内の各ツリーを個別に最適化
    """
    np.random.seed(random_seed)
    n = len(xs)
    if n <= 1:
        verts = [get_tree_vertices(xs[0], ys[0], degs[0])]
        return xs, ys, degs, calculate_score(verts)

    # 現在の頂点を計算
    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    current_score = calculate_score(all_vertices)
    best_score = current_score

    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()

    for _ in range(n_iters):
        # ランダムにツリーを選択
        tree_idx = np.random.randint(0, n)

        # 移動タイプを選択: 0=位置, 1=角度, 2=位置+角度
        move_type = np.random.randint(0, 3)

        old_x = xs[tree_idx]
        old_y = ys[tree_idx]
        old_deg = degs[tree_idx]

        if move_type == 0 or move_type == 2:
            dx = (np.random.random() * 2.0 - 1.0) * pos_delta
            dy = (np.random.random() * 2.0 - 1.0) * pos_delta
            xs[tree_idx] += dx
            ys[tree_idx] += dy

        if move_type == 1 or move_type == 2:
            ddeg = (np.random.random() * 2.0 - 1.0) * ang_delta
            degs[tree_idx] = (degs[tree_idx] + ddeg) % 360.0

        # 新しい頂点を計算
        new_verts = get_tree_vertices(xs[tree_idx], ys[tree_idx], degs[tree_idx])

        # 衝突チェック
        if has_any_overlap_with_others(tree_idx, new_verts, all_vertices):
            # 元に戻す
            xs[tree_idx] = old_x
            ys[tree_idx] = old_y
            degs[tree_idx] = old_deg
            continue

        # 頂点を更新してスコア計算
        all_vertices[tree_idx] = new_verts
        new_score = calculate_score(all_vertices)

        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_xs[:] = xs[:]
                best_ys[:] = ys[:]
                best_degs[:] = degs[:]
        else:
            # 元に戻す
            xs[tree_idx] = old_x
            ys[tree_idx] = old_y
            degs[tree_idx] = old_deg
            all_vertices[tree_idx] = get_tree_vertices(old_x, old_y, old_deg)

    return best_xs, best_ys, best_degs, best_score


@njit(cache=True)
def local_search_with_sa(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    T_max: float,
    T_min: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    SAベースの局所探索（悪化も確率的に受け入れる）
    """
    np.random.seed(random_seed)
    n = len(xs)
    if n <= 1:
        verts = [get_tree_vertices(xs[0], ys[0], degs[0])]
        return xs, ys, degs, calculate_score(verts)

    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    current_score = calculate_score(all_vertices)
    best_score = current_score

    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()

    T_factor = -math.log(T_max / T_min)

    for step in range(n_iters):
        T = T_max * math.exp(T_factor * step / n_iters)
        progress = step / n_iters
        decay = 1.0 - 0.8 * progress
        cur_pos_delta = pos_delta * decay
        cur_ang_delta = ang_delta * decay

        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 3)

        old_x = xs[tree_idx]
        old_y = ys[tree_idx]
        old_deg = degs[tree_idx]

        if move_type == 0 or move_type == 2:
            dx = (np.random.random() * 2.0 - 1.0) * cur_pos_delta
            dy = (np.random.random() * 2.0 - 1.0) * cur_pos_delta
            xs[tree_idx] += dx
            ys[tree_idx] += dy

        if move_type == 1 or move_type == 2:
            ddeg = (np.random.random() * 2.0 - 1.0) * cur_ang_delta
            degs[tree_idx] = (degs[tree_idx] + ddeg) % 360.0

        new_verts = get_tree_vertices(xs[tree_idx], ys[tree_idx], degs[tree_idx])

        if has_any_overlap_with_others(tree_idx, new_verts, all_vertices):
            xs[tree_idx] = old_x
            ys[tree_idx] = old_y
            degs[tree_idx] = old_deg
            continue

        all_vertices[tree_idx] = new_verts
        new_score = calculate_score(all_vertices)
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
                best_xs[:] = xs[:]
                best_ys[:] = ys[:]
                best_degs[:] = degs[:]
        else:
            xs[tree_idx] = old_x
            ys[tree_idx] = old_y
            degs[tree_idx] = old_deg
            all_vertices[tree_idx] = get_tree_vertices(old_x, old_y, old_deg)

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
    idx = 0
    for n in range(1, 201):
        vertices = [
            get_tree_vertices(all_xs[idx + i], all_ys[idx + i], all_degs[idx + i]) for i in range(n)
        ]
        total += calculate_score(vertices)
        idx += n
    return total


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Local Search Optimizer (exp008_local_search)")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nBaseline: {baseline_path}")

    all_xs, all_ys, all_degs = load_submission_data(baseline_path)
    baseline_total = calculate_total_score(all_xs, all_ys, all_degs)
    print(f"Baseline total score: {baseline_total:.6f}")
    print(f"Total trees loaded: {len(all_xs)} (expected: {200*201//2})")
    
    # サンプルグループのスコアを表示
    for test_n in [10, 50, 100, 200]:
        start = test_n * (test_n - 1) // 2
        test_verts = [get_tree_vertices(all_xs[start + i], all_ys[start + i], all_degs[start + i]) for i in range(test_n)]
        test_score = calculate_score(test_verts)
        print(f"  Group {test_n} score: {test_score:.6f}")

    # パラメータ
    ls_cfg = CONFIG["local_search"]
    n_iters = int(ls_cfg["n_iters"])
    pos_delta = float(ls_cfg["pos_delta"])
    ang_delta = float(ls_cfg["ang_delta"])
    use_sa = bool(ls_cfg.get("use_sa", True))
    T_max = float(ls_cfg.get("T_max", 0.01))
    T_min = float(ls_cfg.get("T_min", 0.0001))
    n_min = int(ls_cfg.get("n_min", 2))
    n_max = int(ls_cfg.get("n_max", 200))
    seed_base = int(ls_cfg.get("seed_base", 42))
    debug = bool(ls_cfg.get("debug", False))

    # グループごとに最適化
    print(f"\nOptimizing groups {n_min} to {n_max}...")
    print(f"  Iterations per group: {n_iters}")
    print(f"  Position delta: {pos_delta}")
    print(f"  Angle delta: {ang_delta}")
    print(f"  Use SA: {use_sa}")

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    total_improved = 0.0
    improved_groups = 0

    # group_start[n] = グループnの開始インデックス = sum(1, 2, ..., n-1) = n*(n-1)/2
    group_start = [0]  # group_start[1] = 0
    for i in range(1, 201):
        group_start.append(group_start[-1] + i)
    # group_start[n] は グループn の開始位置

    for n in tqdm(range(n_min, n_max + 1), desc="Optimizing"):
        # グループn: インデックス n*(n-1)/2 から n個
        start = n * (n - 1) // 2
        end = start + n

        xs = new_xs[start:end].copy()
        ys = new_ys[start:end].copy()
        degs = new_degs[start:end].copy()

        if len(xs) != n:
            print(f"Warning: Group {n} has {len(xs)} trees instead of {n}")
            continue

        # 元のスコア
        orig_verts = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
        orig_score = calculate_score(orig_verts)

        seed = seed_base + n * 1000

        if use_sa:
            opt_xs, opt_ys, opt_degs, opt_score = local_search_with_sa(
                xs, ys, degs, n_iters, pos_delta, ang_delta, T_max, T_min, seed
            )
        else:
            opt_xs, opt_ys, opt_degs, opt_score = local_search_group(
                xs, ys, degs, n_iters, pos_delta, ang_delta, seed
            )

        if opt_score < orig_score - 1e-9:
            improvement = orig_score - opt_score
            total_improved += improvement
            improved_groups += 1
            # インデックスを再計算して代入
            start_idx = n * (n - 1) // 2
            new_xs[start_idx:start_idx + n] = opt_xs
            new_ys[start_idx:start_idx + n] = opt_ys
            new_degs[start_idx:start_idx + n] = opt_degs
            if debug:
                print(f"  Group {n}: {orig_score:.6f} -> {opt_score:.6f} (improved {improvement:.6f})")
        elif debug and n % 20 == 0:
            print(f"  Group {n}: {orig_score:.6f} (no improvement)")

    final_score = calculate_total_score(new_xs, new_ys, new_degs)

    print("\n" + "=" * 80)
    print(f"  Baseline total:    {baseline_total:.6f}")
    print(f"  After local search: {final_score:.6f}")
    print(f"  Total improvement: {baseline_total - final_score:+.6f}")
    print(f"  Improved groups:   {improved_groups}")
    print("=" * 80)

    # 保存
    if final_score < baseline_total:
        out_path = CONFIG["paths"]["output"]
        save_submission(out_path, new_xs, new_ys, new_degs)
        print(f"Saved to {out_path}")
    else:
        print("No improvement - keeping baseline")

