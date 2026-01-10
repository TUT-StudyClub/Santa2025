"""
exp019_new_patterns: 完全に新しい配置パターンを生成

アプローチ:
1. 複数の初期配置パターンを試す（六角形、インターロッキング、スパイラルなど）
2. 各パターンをSAで最適化
3. 最良のパターンを選択
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
        elif arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
    return os.path.join("experiments", "exp019_new_patterns", "exp", f"{config_name}.yaml")


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

TREE_HEIGHT = TIP_Y - TRUNK_BOTTOM_Y  # 1.0
TREE_WIDTH = BASE_W  # 0.7


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
def has_any_overlap(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> bool:
    n = len(xs)
    vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(vertices[i], vertices[j]):
                return True
    return False


@njit(cache=True)
def compute_score(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> float:
    n = len(xs)
    if n == 0:
        return 0.0

    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for i in range(n):
        verts = get_tree_vertices(xs[i], ys[i], degs[i])
        for j in range(15):
            x, y = verts[j, 0], verts[j, 1]
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

    side = max(max_x - min_x, max_y - min_y)
    return side * side / n


# -----------------------------------------------------------------------------
# Pattern Generators
# -----------------------------------------------------------------------------
def generate_grid_pattern(n: int, spacing: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """単純なグリッドパターン"""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    xs, ys, degs = [], [], []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            xs.append(col * spacing)
            ys.append(row * spacing)
            degs.append(0.0)
            idx += 1

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    degs = np.array(degs, dtype=np.float64)

    # 中心を原点に
    xs -= xs.mean()
    ys -= ys.mean()

    return xs, ys, degs


def generate_hexagonal_pattern(n: int, spacing: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """六角形パターン"""
    cols = int(math.ceil(math.sqrt(n * 1.15)))
    rows = int(math.ceil(n / cols))

    xs, ys, degs = [], [], []
    idx = 0
    for row in range(rows + 1):
        offset = (row % 2) * spacing * 0.5
        for col in range(cols + 1):
            if idx >= n:
                break
            xs.append(col * spacing + offset)
            ys.append(row * spacing * 0.866)  # sqrt(3)/2
            degs.append(0.0)
            idx += 1
        if idx >= n:
            break

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    degs = np.array(degs, dtype=np.float64)

    xs -= xs.mean()
    ys -= ys.mean()

    return xs, ys, degs


def generate_interlocking_pattern(n: int, spacing: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """インターロッキングパターン（交互に180度回転）"""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    xs, ys, degs = [], [], []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            xs.append(col * spacing)
            ys.append(row * spacing)
            # 交互に180度回転
            if (row + col) % 2 == 0:
                degs.append(0.0)
            else:
                degs.append(180.0)
            idx += 1

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    degs = np.array(degs, dtype=np.float64)

    xs -= xs.mean()
    ys -= ys.mean()

    return xs, ys, degs


def generate_diagonal_pattern(n: int, spacing: float = 0.45) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """斜めグリッドパターン（45度回転したグリッド）"""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    xs, ys, degs = [], [], []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            # 45度回転したグリッド
            x = (col - row) * spacing * 0.707
            y = (col + row) * spacing * 0.707
            xs.append(x)
            ys.append(y)
            degs.append(45.0)
            idx += 1

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    degs = np.array(degs, dtype=np.float64)

    xs -= xs.mean()
    ys -= ys.mean()

    return xs, ys, degs


def generate_spiral_pattern(n: int, spacing: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """スパイラルパターン"""
    xs, ys, degs = [], [], []

    for i in range(n):
        angle = i * 2.4  # 黄金角に近い
        r = spacing * math.sqrt(i + 1)
        xs.append(r * math.cos(angle))
        ys.append(r * math.sin(angle))
        degs.append((angle * 180 / math.pi) % 360)

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    degs = np.array(degs, dtype=np.float64)

    xs -= xs.mean()
    ys -= ys.mean()

    return xs, ys, degs


def generate_concentric_pattern(n: int, spacing: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同心円パターン"""
    xs, ys, degs = [], [], []

    if n >= 1:
        xs.append(0.0)
        ys.append(0.0)
        degs.append(0.0)

    remaining = n - 1
    ring = 1
    while remaining > 0:
        # リングごとの木の数
        count = min(remaining, int(2 * math.pi * ring * spacing / 0.5))
        if count < 1:
            count = 1
        r = ring * spacing

        for i in range(count):
            angle = 2 * math.pi * i / count
            xs.append(r * math.cos(angle))
            ys.append(r * math.sin(angle))
            degs.append((angle * 180 / math.pi) % 360)
            remaining -= 1
            if remaining <= 0:
                break
        ring += 1

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    degs = np.array(degs, dtype=np.float64)

    return xs, ys, degs


# -----------------------------------------------------------------------------
# SA Optimization
# -----------------------------------------------------------------------------
@njit(cache=True)
def optimize_group_sa(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    n_iters: int,
    T_max: float,
    T_min: float,
    pos_delta: float,
    ang_delta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    np.random.seed(seed)
    n = len(xs)

    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()
    best_score = compute_score(xs, ys, degs)

    cur_xs = xs.copy()
    cur_ys = ys.copy()
    cur_degs = degs.copy()
    cur_score = best_score

    T_factor = -math.log(T_max / T_min)

    for it in range(n_iters):
        progress = it / n_iters
        T = T_max * math.exp(T_factor * progress)
        decay = 1.0 - 0.8 * progress

        tree_idx = np.random.randint(0, n)

        old_x = cur_xs[tree_idx]
        old_y = cur_ys[tree_idx]
        old_deg = cur_degs[tree_idx]

        dx = (np.random.random() * 2.0 - 1.0) * pos_delta * decay
        dy = (np.random.random() * 2.0 - 1.0) * pos_delta * decay
        ddeg = (np.random.random() * 2.0 - 1.0) * ang_delta * decay

        cur_xs[tree_idx] += dx
        cur_ys[tree_idx] += dy
        cur_degs[tree_idx] = (cur_degs[tree_idx] + ddeg) % 360.0

        if has_any_overlap(cur_xs, cur_ys, cur_degs):
            cur_xs[tree_idx] = old_x
            cur_ys[tree_idx] = old_y
            cur_degs[tree_idx] = old_deg
            continue

        new_score = compute_score(cur_xs, cur_ys, cur_degs)
        delta = new_score - cur_score

        accept = False
        if delta < 0:
            accept = True
        elif T > 1e-10 and np.random.random() < math.exp(-delta / T):
            accept = True

        if accept:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_xs[:] = cur_xs[:]
                best_ys[:] = cur_ys[:]
                best_degs[:] = cur_degs[:]
        else:
            cur_xs[tree_idx] = old_x
            cur_ys[tree_idx] = old_y
            cur_degs[tree_idx] = old_deg

    return best_xs, best_ys, best_degs, best_score


@njit(cache=True)
def shrink_group(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    min_scale: float,
    n_iters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(xs)
    if n <= 1:
        return xs, ys, compute_score(xs, ys, degs)

    cx = 0.0
    cy = 0.0
    for i in range(n):
        cx += xs[i]
        cy += ys[i]
    cx /= n
    cy /= n

    low = min_scale
    high = 1.0
    best_scale = 1.0

    for _ in range(n_iters):
        mid = (low + high) * 0.5
        test_xs = np.empty(n, dtype=np.float64)
        test_ys = np.empty(n, dtype=np.float64)
        for i in range(n):
            test_xs[i] = cx + mid * (xs[i] - cx)
            test_ys[i] = cy + mid * (ys[i] - cy)

        if not has_any_overlap(test_xs, test_ys, degs):
            best_scale = mid
            high = mid
        else:
            low = mid

    new_xs = np.empty(n, dtype=np.float64)
    new_ys = np.empty(n, dtype=np.float64)
    for i in range(n):
        new_xs[i] = cx + best_scale * (xs[i] - cx)
        new_ys[i] = cy + best_scale * (ys[i] - cy)

    score = compute_score(new_xs, new_ys, degs)
    return new_xs, new_ys, score


def expand_until_valid(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """オーバーラップがなくなるまで拡大"""
    n = len(xs)
    if n <= 1:
        return xs, ys

    cx = xs.mean()
    cy = ys.mean()

    scale = 1.0
    while has_any_overlap(xs, ys, degs):
        scale *= 1.1
        xs = cx + scale * (xs - cx)
        ys = cy + scale * (ys - cy)
        if scale > 100:
            break

    return xs, ys


def optimize_single_group(args):
    """単一グループの最適化"""
    n, baseline_xs, baseline_ys, baseline_degs, cfg, seed = args

    baseline_score = compute_score(baseline_xs, baseline_ys, baseline_degs)

    # 複数のパターンを試す
    patterns = []

    # パターン生成
    for spacing in [0.4, 0.45, 0.5, 0.55]:
        patterns.append(("grid", generate_grid_pattern(n, spacing)))
        patterns.append(("hex", generate_hexagonal_pattern(n, spacing)))
        patterns.append(("interlock", generate_interlocking_pattern(n, spacing)))
        patterns.append(("diagonal", generate_diagonal_pattern(n, spacing)))
        patterns.append(("spiral", generate_spiral_pattern(n, spacing)))
        patterns.append(("concentric", generate_concentric_pattern(n, spacing)))

    best_xs = baseline_xs.copy()
    best_ys = baseline_ys.copy()
    best_degs = baseline_degs.copy()
    best_score = baseline_score

    sa_cfg = cfg.get("sa", {})
    shrink_cfg = cfg.get("shrink", {})

    for pattern_name, (xs, ys, degs) in patterns:
        # オーバーラップがあれば拡大
        xs, ys = expand_until_valid(xs, ys, degs)

        # SA最適化
        if sa_cfg.get("enabled", True):
            xs, ys, degs, _ = optimize_group_sa(
                xs,
                ys,
                degs,
                n_iters=int(sa_cfg.get("n_iters", 5000) * n),
                T_max=float(sa_cfg.get("T_max", 0.1)),
                T_min=float(sa_cfg.get("T_min", 0.0001)),
                pos_delta=float(sa_cfg.get("pos_delta", 0.05)),
                ang_delta=float(sa_cfg.get("ang_delta", 5.0)),
                seed=seed,
            )

        # 縮小
        if shrink_cfg.get("enabled", True):
            xs, ys, _ = shrink_group(
                xs,
                ys,
                degs,
                min_scale=float(shrink_cfg.get("min_scale", 0.8)),
                n_iters=int(shrink_cfg.get("n_iters", 30)),
            )

        score = compute_score(xs, ys, degs)

        if score < best_score:
            best_score = score
            best_xs = xs.copy()
            best_ys = ys.copy()
            best_degs = degs.copy()

    return n, best_xs, best_ys, best_degs, baseline_score, best_score


def load_baseline(filepath: str) -> tuple[dict, float]:
    df = pd.read_csv(filepath)
    groups = {}
    total_score = 0.0

    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")

        xs, ys, degs = [], [], []
        for _, row in group.iterrows():
            x = float(row["x"][1:]) if isinstance(row["x"], str) else float(row["x"])
            y = float(row["y"][1:]) if isinstance(row["y"], str) else float(row["y"])
            deg = float(row["deg"][1:]) if isinstance(row["deg"], str) else float(row["deg"])
            xs.append(x)
            ys.append(y)
            degs.append(deg)

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        degs = np.array(degs, dtype=np.float64)

        groups[n] = (xs, ys, degs)
        total_score += compute_score(xs, ys, degs)

    return groups, total_score


def save_submission(filepath: str, groups: dict) -> None:
    rows = []
    for n in range(1, 201):
        xs, ys, degs = groups[n]
        for t in range(n):
            rows.append(
                {
                    "id": f"{n:03d}_{t}",
                    "x": f"s{xs[t]}",
                    "y": f"s{ys[t]}",
                    "deg": f"s{degs[t]}",
                }
            )
    pd.DataFrame(rows).to_csv(filepath, index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("exp019_new_patterns: New Pattern Generation")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nLoading baseline: {baseline_path}")

    groups, baseline_total = load_baseline(baseline_path)
    print(f"Baseline total score: {baseline_total:.6f}")

    opt_cfg = CONFIG.get("optimization", {})
    n_min = int(opt_cfg.get("n_min", 2))
    n_max = int(opt_cfg.get("n_max", 200))
    seed_base = int(opt_cfg.get("seed", 42))

    tasks = []
    for n in range(n_min, n_max + 1):
        xs, ys, degs = groups[n]
        tasks.append((n, xs.copy(), ys.copy(), degs.copy(), opt_cfg, seed_base + n))

    print(f"\nOptimizing groups {n_min} to {n_max} with multiple patterns...")
    num_workers = min(cpu_count(), len(tasks))
    t0 = time.time()

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(optimize_single_group, tasks),
            total=len(tasks),
            desc="Optimizing",
        ):
            results.append(result)

    print(f"Optimization completed in {time.time() - t0:.1f}s")

    improved_groups = {}
    total_improvement = 0.0
    improved_count = 0

    for n, xs, ys, degs, orig_score, final_score in results:
        improvement = orig_score - final_score
        if improvement > 1e-9:
            improved_groups[n] = (xs, ys, degs)
            total_improvement += improvement
            improved_count += 1
            if improvement > 0.001:
                print(f"  Group {n}: {orig_score:.6f} -> {final_score:.6f} ({improvement:+.6f})")

    print(f"\nImproved {improved_count} groups")
    print(f"Total improvement: {total_improvement:.6f}")

    for n, (xs, ys, degs) in improved_groups.items():
        groups[n] = (xs, ys, degs)

    final_total = 0.0
    for n in range(1, 201):
        xs, ys, degs = groups[n]
        final_total += compute_score(xs, ys, degs)

    print("=" * 80)
    print(f"  Baseline total:    {baseline_total:.6f}")
    print(f"  Final total:       {final_total:.6f}")
    print(f"  Total improvement: {baseline_total - final_total:+.6f}")
    print("=" * 80)

    if final_total < baseline_total:
        out_path = CONFIG["paths"]["output"]
        save_submission(out_path, groups)
        print(f"Saved to {out_path}")
    else:
        print("No improvement - keeping baseline")
