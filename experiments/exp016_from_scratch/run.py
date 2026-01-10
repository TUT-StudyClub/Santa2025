"""
exp016_from_scratch: ゼロから最適な配置を探索

ベースラインに依存せず、完全に新しい配置パターンを生成。
複数の初期パターン（グリッド、六角形、らせん等）から始めて最適化。
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
    return os.path.join("experiments", "exp016_from_scratch", "exp", f"{config_name}.yaml")


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

    pts = np.array([
        [0.0, TIP_Y],
        [TOP_W / 2.0, TIER_1_Y], [TOP_W / 4.0, TIER_1_Y],
        [MID_W / 2.0, TIER_2_Y], [MID_W / 4.0, TIER_2_Y],
        [BASE_W / 2.0, BASE_Y], [TRUNK_W / 2.0, BASE_Y],
        [TRUNK_W / 2.0, TRUNK_BOTTOM_Y], [-TRUNK_W / 2.0, TRUNK_BOTTOM_Y],
        [-TRUNK_W / 2.0, BASE_Y], [-BASE_W / 2.0, BASE_Y],
        [-MID_W / 4.0, TIER_2_Y], [-MID_W / 2.0, TIER_2_Y],
        [-TOP_W / 4.0, TIER_1_Y], [-TOP_W / 2.0, TIER_1_Y],
    ], dtype=np.float64)

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
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
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
def segments_intersect(p1x: float, p1y: float, p2x: float, p2y: float,
                       p3x: float, p3y: float, p4x: float, p4y: float) -> bool:
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
            if segments_intersect(verts1[i, 0], verts1[i, 1], verts1[j, 0], verts1[j, 1],
                                  verts2[k, 0], verts2[k, 1], verts2[m, 0], verts2[m, 1]):
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
        if x1 < min_x: min_x = x1
        if y1 < min_y: min_y = y1
        if x2 > max_x: max_x = x2
        if y2 > max_y: max_y = y2
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def calculate_score(all_vertices: list[np.ndarray]) -> float:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    side = max(max_x - min_x, max_y - min_y)
    return side * side / len(all_vertices)


# -----------------------------------------------------------------------------
# Initial Pattern Generators
# -----------------------------------------------------------------------------
@njit(cache=True)
def generate_grid_pattern(n: int, spacing: float, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """グリッドパターン"""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    degs = np.full(n, angle, dtype=np.float64)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            xs[idx] = c * spacing
            ys[idx] = r * spacing
            idx += 1
    
    # 中心を原点に
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs -= cx
    ys -= cy
    
    return xs, ys, degs


@njit(cache=True)
def generate_hexagonal_pattern(n: int, spacing: float, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """六角形パターン（蜂の巣状）"""
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    degs = np.full(n, angle, dtype=np.float64)
    
    cols = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3))))
    
    idx = 0
    row = 0
    while idx < n:
        for c in range(cols):
            if idx >= n:
                break
            x = c * spacing
            if row % 2 == 1:
                x += spacing * 0.5
            y = row * spacing * math.sqrt(3) / 2
            xs[idx] = x
            ys[idx] = y
            idx += 1
        row += 1
    
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs -= cx
    ys -= cy
    
    return xs, ys, degs


@njit(cache=True)
def generate_spiral_pattern(n: int, base_radius: float, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """らせんパターン"""
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    degs = np.full(n, angle, dtype=np.float64)
    
    golden_angle = math.pi * (3 - math.sqrt(5))  # 黄金角
    
    for i in range(n):
        r = base_radius * math.sqrt(i + 1)
        theta = i * golden_angle
        xs[i] = r * math.cos(theta)
        ys[i] = r * math.sin(theta)
    
    return xs, ys, degs


@njit(cache=True)
def generate_interlocking_pattern(n: int, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """インターロッキングパターン（交互に向きを変える）"""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    degs = np.empty(n, dtype=np.float64)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            xs[idx] = c * spacing
            ys[idx] = r * spacing
            # 交互に180度回転
            if (r + c) % 2 == 0:
                degs[idx] = 0.0
            else:
                degs[idx] = 180.0
            idx += 1
    
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs -= cx
    ys -= cy
    
    return xs, ys, degs


@njit(cache=True)
def generate_diamond_pattern(n: int, spacing: float, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ダイヤモンドパターン（45度回転グリッド）"""
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    degs = np.full(n, angle, dtype=np.float64)
    
    # 45度回転したグリッド
    cos45 = math.cos(math.pi / 4)
    sin45 = math.sin(math.pi / 4)
    
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            gx = c * spacing
            gy = r * spacing
            xs[idx] = gx * cos45 - gy * sin45
            ys[idx] = gx * sin45 + gy * cos45
            idx += 1
    
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs -= cx
    ys -= cy
    
    return xs, ys, degs


# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------
@njit(cache=True)
def simulated_annealing(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    T_max: float,
    T_min: float,
    n_steps: int,
    pos_delta: float,
    deg_delta: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """焼きなまし法による最適化"""
    np.random.seed(random_seed)
    n = len(xs)
    
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()
    
    all_vertices = [get_tree_vertices(best_xs[i], best_ys[i], best_degs[i]) for i in range(n)]
    
    if has_any_overlap(all_vertices):
        return best_xs, best_ys, best_degs, math.inf
    
    best_score = calculate_score(all_vertices)
    current_score = best_score
    
    T = T_max
    T_decay = (T_min / T_max) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 3)
        
        old_x, old_y, old_deg = best_xs[tree_idx], best_ys[tree_idx], best_degs[tree_idx]
        old_verts = all_vertices[tree_idx]
        
        if move_type == 0:
            best_xs[tree_idx] += (np.random.random() * 2 - 1) * pos_delta
        elif move_type == 1:
            best_ys[tree_idx] += (np.random.random() * 2 - 1) * pos_delta
        else:
            best_degs[tree_idx] = (best_degs[tree_idx] + (np.random.random() * 2 - 1) * deg_delta) % 360.0
        
        new_verts = get_tree_vertices(best_xs[tree_idx], best_ys[tree_idx], best_degs[tree_idx])
        
        # 衝突チェック
        overlap = False
        for j in range(n):
            if j != tree_idx and polygons_overlap(new_verts, all_vertices[j]):
                overlap = True
                break
        
        if overlap:
            best_xs[tree_idx], best_ys[tree_idx], best_degs[tree_idx] = old_x, old_y, old_deg
            continue
        
        all_vertices[tree_idx] = new_verts
        new_score = calculate_score(all_vertices)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
        else:
            best_xs[tree_idx], best_ys[tree_idx], best_degs[tree_idx] = old_x, old_y, old_deg
            all_vertices[tree_idx] = old_verts
        
        T *= T_decay
    
    return best_xs, best_ys, best_degs, best_score


@njit(cache=True)
def shrink_to_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    min_scale: float,
    n_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """バウンディングボックスを縮小"""
    n = len(xs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()
    
    all_vertices = [get_tree_vertices(best_xs[i], best_ys[i], best_degs[i]) for i in range(n)]
    if has_any_overlap(all_vertices):
        return best_xs, best_ys, best_degs, math.inf
    
    best_score = calculate_score(all_vertices)
    
    for _ in range(n_iters):
        scale = min_scale + np.random.random() * (1.0 - min_scale)
        
        # 中心を計算
        min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        
        # 縮小
        new_xs = cx + (best_xs - cx) * scale
        new_ys = cy + (best_ys - cy) * scale
        
        new_vertices = [get_tree_vertices(new_xs[i], new_ys[i], best_degs[i]) for i in range(n)]
        
        if not has_any_overlap(new_vertices):
            new_score = calculate_score(new_vertices)
            if new_score < best_score:
                best_xs = new_xs.copy()
                best_ys = new_ys.copy()
                best_score = new_score
                all_vertices = new_vertices
    
    return best_xs, best_ys, best_degs, best_score


@njit(cache=True)
def optimize_group(
    n: int,
    patterns: int,
    spacings: np.ndarray,
    angles: np.ndarray,
    sa_steps: int,
    T_max: float,
    T_min: float,
    pos_delta: float,
    deg_delta: float,
    shrink_iters: int,
    min_scale: float,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """グループを最適化"""
    best_xs = np.zeros(n, dtype=np.float64)
    best_ys = np.zeros(n, dtype=np.float64)
    best_degs = np.zeros(n, dtype=np.float64)
    best_score = math.inf
    
    trial = 0
    for pattern in range(patterns):
        for spacing in spacings:
            for angle in angles:
                # パターン生成
                if pattern == 0:
                    xs, ys, degs = generate_grid_pattern(n, spacing, angle)
                elif pattern == 1:
                    xs, ys, degs = generate_hexagonal_pattern(n, spacing, angle)
                elif pattern == 2:
                    xs, ys, degs = generate_spiral_pattern(n, spacing * 0.3, angle)
                elif pattern == 3:
                    xs, ys, degs = generate_interlocking_pattern(n, spacing)
                else:
                    xs, ys, degs = generate_diamond_pattern(n, spacing, angle)
                
                # SA最適化
                seed = seed_base + trial * 1000
                xs, ys, degs, score = simulated_annealing(
                    xs, ys, degs, T_max, T_min, sa_steps, pos_delta, deg_delta, seed
                )
                
                if score < math.inf:
                    # 縮小
                    xs, ys, degs, score = shrink_to_fit(xs, ys, degs, min_scale, shrink_iters)
                    
                    if score < best_score:
                        best_xs = xs.copy()
                        best_ys = ys.copy()
                        best_degs = degs.copy()
                        best_score = score
                
                trial += 1
    
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
    print("From Scratch Optimizer (exp016_from_scratch)")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nBaseline: {baseline_path}")

    base_xs, base_ys, base_degs = load_submission_data(baseline_path)
    baseline_total = calculate_total_score(base_xs, base_ys, base_degs)
    print(f"Baseline total score: {baseline_total:.6f}")

    # パラメータ
    cfg = CONFIG["optimizer"]
    n_min = int(cfg["n_min"])
    n_max = int(cfg["n_max"])
    patterns = int(cfg["patterns"])
    spacings = np.array(cfg["spacings"], dtype=np.float64)
    angles = np.array(cfg["angles"], dtype=np.float64)
    sa_steps = int(cfg["sa_steps"])
    T_max = float(cfg["T_max"])
    T_min = float(cfg["T_min"])
    pos_delta = float(cfg["pos_delta"])
    deg_delta = float(cfg["deg_delta"])
    shrink_iters = int(cfg["shrink_iters"])
    min_scale = float(cfg["min_scale"])
    seed_base = int(cfg.get("seed_base", 42))

    print(f"\nOptimizing groups {n_min} to {n_max} from scratch...")
    print(f"  Patterns: {patterns}")
    print(f"  Spacings: {spacings}")
    print(f"  Angles: {angles}")

    new_xs = base_xs.copy()
    new_ys = base_ys.copy()
    new_degs = base_degs.copy()

    total_improved = 0.0
    improved_groups = 0

    for n in tqdm(range(n_min, n_max + 1), desc="Optimizing"):
        start = n * (n - 1) // 2

        # ベースラインのスコア
        orig_verts = [
            get_tree_vertices(base_xs[start + i], base_ys[start + i], base_degs[start + i])
            for i in range(n)
        ]
        orig_score = calculate_score(orig_verts)

        # ゼロから最適化
        opt_xs, opt_ys, opt_degs, opt_score = optimize_group(
            n, patterns, spacings, angles,
            sa_steps, T_max, T_min, pos_delta, deg_delta,
            shrink_iters, min_scale, seed_base + n * 10000
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
    print(f"  After optimization: {final_score:.6f}")
    print(f"  Total improvement: {baseline_total - final_score:+.6f}")
    print(f"  Improved groups:   {improved_groups}")
    print("=" * 80)

    if final_score < baseline_total:
        out_path = CONFIG["paths"]["output"]
        save_submission(out_path, new_xs, new_ys, new_degs)
        print(f"Saved to {out_path}")
    else:
        print("No improvement - keeping baseline")


