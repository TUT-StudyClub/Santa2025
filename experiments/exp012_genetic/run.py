"""
exp012_genetic: 遺伝的アルゴリズムによる最適化

複数の解を並列に進化させ、交叉と突然変異により
局所最適を脱出してグローバルな最適解を探索する。
"""

import math
import os
import sys
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
    return os.path.join("experiments", "exp012_genetic", "exp", f"{config_name}.yaml")


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


@njit(cache=True)
def evaluate_solution(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> float:
    """解の評価（重なりがある場合はペナルティ）"""
    n = len(xs)
    all_vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    if has_any_overlap(all_vertices):
        return math.inf
    return calculate_score(all_vertices)


# -----------------------------------------------------------------------------
# Genetic Algorithm Operations
# -----------------------------------------------------------------------------
@njit(cache=True)
def repair_solution(
    xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, max_iters: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """重なりを解消するための修復操作"""
    n = len(xs)
    new_xs = xs.copy()
    new_ys = ys.copy()
    new_degs = degs.copy()

    all_vertices = [get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) for i in range(n)]

    for iteration in range(max_iters):
        if not has_any_overlap(all_vertices):
            return new_xs, new_ys, new_degs, True

        # 重なっているペアを見つけて修復
        for i in range(n):
            for j in range(i + 1, n):
                if polygons_overlap(all_vertices[i], all_vertices[j]):
                    # 2つの木を離す
                    dx = new_xs[j] - new_xs[i]
                    dy = new_ys[j] - new_ys[i]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1e-6:
                        dx, dy = 1.0, 0.0
                        dist = 1.0

                    # 少し離す
                    move = 0.05
                    new_xs[i] -= dx / dist * move
                    new_ys[i] -= dy / dist * move
                    new_xs[j] += dx / dist * move
                    new_ys[j] += dy / dist * move

                    all_vertices[i] = get_tree_vertices(new_xs[i], new_ys[i], new_degs[i])
                    all_vertices[j] = get_tree_vertices(new_xs[j], new_ys[j], new_degs[j])

    return new_xs, new_ys, new_degs, not has_any_overlap(all_vertices)


@njit(cache=True)
def crossover_blend(
    parent1_xs: np.ndarray,
    parent1_ys: np.ndarray,
    parent1_degs: np.ndarray,
    parent2_xs: np.ndarray,
    parent2_ys: np.ndarray,
    parent2_degs: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BLX-α交叉: 2つの親の間をブレンド"""
    n = len(parent1_xs)
    child_xs = np.empty(n, dtype=np.float64)
    child_ys = np.empty(n, dtype=np.float64)
    child_degs = np.empty(n, dtype=np.float64)

    for i in range(n):
        # 位置のブレンド
        min_x = min(parent1_xs[i], parent2_xs[i])
        max_x = max(parent1_xs[i], parent2_xs[i])
        range_x = max_x - min_x
        child_xs[i] = np.random.uniform(min_x - alpha * range_x, max_x + alpha * range_x)

        min_y = min(parent1_ys[i], parent2_ys[i])
        max_y = max(parent1_ys[i], parent2_ys[i])
        range_y = max_y - min_y
        child_ys[i] = np.random.uniform(min_y - alpha * range_y, max_y + alpha * range_y)

        # 角度のブレンド（円周上で）
        d1 = parent1_degs[i]
        d2 = parent2_degs[i]
        diff = d2 - d1
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        child_degs[i] = (d1 + np.random.uniform(-alpha, 1 + alpha) * diff) % 360.0

    return child_xs, child_ys, child_degs


@njit(cache=True)
def crossover_uniform(
    parent1_xs: np.ndarray,
    parent1_ys: np.ndarray,
    parent1_degs: np.ndarray,
    parent2_xs: np.ndarray,
    parent2_ys: np.ndarray,
    parent2_degs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一様交叉: 各木をランダムに親から選択"""
    n = len(parent1_xs)
    child_xs = np.empty(n, dtype=np.float64)
    child_ys = np.empty(n, dtype=np.float64)
    child_degs = np.empty(n, dtype=np.float64)

    for i in range(n):
        if np.random.random() < 0.5:
            child_xs[i] = parent1_xs[i]
            child_ys[i] = parent1_ys[i]
            child_degs[i] = parent1_degs[i]
        else:
            child_xs[i] = parent2_xs[i]
            child_ys[i] = parent2_ys[i]
            child_degs[i] = parent2_degs[i]

    return child_xs, child_ys, child_degs


@njit(cache=True)
def mutate(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    mutation_rate: float,
    pos_sigma: float,
    deg_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """突然変異: ランダムに木を摂動"""
    n = len(xs)
    new_xs = xs.copy()
    new_ys = ys.copy()
    new_degs = degs.copy()

    for i in range(n):
        if np.random.random() < mutation_rate:
            new_xs[i] += np.random.normal(0, pos_sigma)
            new_ys[i] += np.random.normal(0, pos_sigma)
            new_degs[i] = (new_degs[i] + np.random.normal(0, deg_sigma)) % 360.0

    return new_xs, new_ys, new_degs


@njit(cache=True)
def local_search(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    deg_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """局所探索による改善"""
    n = len(xs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_degs = degs.copy()

    all_vertices = [get_tree_vertices(best_xs[i], best_ys[i], best_degs[i]) for i in range(n)]
    if has_any_overlap(all_vertices):
        return best_xs, best_ys, best_degs, math.inf

    best_score = calculate_score(all_vertices)

    for _ in range(n_iters):
        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 3)

        old_x, old_y, old_deg = best_xs[tree_idx], best_ys[tree_idx], best_degs[tree_idx]

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

        if new_score < best_score:
            best_score = new_score
        else:
            best_xs[tree_idx], best_ys[tree_idx], best_degs[tree_idx] = old_x, old_y, old_deg
            all_vertices[tree_idx] = get_tree_vertices(old_x, old_y, old_deg)

    return best_xs, best_ys, best_degs, best_score


@njit(cache=True)
def genetic_algorithm(
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    pop_size: int,
    n_generations: int,
    mutation_rate: float,
    pos_sigma: float,
    deg_sigma: float,
    elite_ratio: float,
    local_search_iters: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """遺伝的アルゴリズムのメインループ"""
    np.random.seed(random_seed)
    n = len(init_xs)

    # 初期集団を生成
    population_xs = np.empty((pop_size, n), dtype=np.float64)
    population_ys = np.empty((pop_size, n), dtype=np.float64)
    population_degs = np.empty((pop_size, n), dtype=np.float64)
    fitness = np.empty(pop_size, dtype=np.float64)

    # 最初の個体はベースライン
    population_xs[0] = init_xs.copy()
    population_ys[0] = init_ys.copy()
    population_degs[0] = init_degs.copy()
    fitness[0] = evaluate_solution(init_xs, init_ys, init_degs)

    # 残りはベースラインに摂動を加えて生成
    for i in range(1, pop_size):
        population_xs[i] = init_xs + np.random.normal(0, pos_sigma * 2, n)
        population_ys[i] = init_ys + np.random.normal(0, pos_sigma * 2, n)
        population_degs[i] = (init_degs + np.random.normal(0, deg_sigma * 2, n)) % 360.0

        # 修復
        population_xs[i], population_ys[i], population_degs[i], valid = repair_solution(
            population_xs[i], population_ys[i], population_degs[i]
        )
        if valid:
            fitness[i] = evaluate_solution(population_xs[i], population_ys[i], population_degs[i])
        else:
            fitness[i] = math.inf

    best_idx = np.argmin(fitness)
    best_xs = population_xs[best_idx].copy()
    best_ys = population_ys[best_idx].copy()
    best_degs = population_degs[best_idx].copy()
    best_score = fitness[best_idx]

    n_elite = max(1, int(pop_size * elite_ratio))

    for gen in range(n_generations):
        # エリート選択
        sorted_indices = np.argsort(fitness)
        new_population_xs = np.empty((pop_size, n), dtype=np.float64)
        new_population_ys = np.empty((pop_size, n), dtype=np.float64)
        new_population_degs = np.empty((pop_size, n), dtype=np.float64)
        new_fitness = np.empty(pop_size, dtype=np.float64)

        # エリートをコピー
        for i in range(n_elite):
            idx = sorted_indices[i]
            new_population_xs[i] = population_xs[idx].copy()
            new_population_ys[i] = population_ys[idx].copy()
            new_population_degs[i] = population_degs[idx].copy()
            new_fitness[i] = fitness[idx]

        # 残りは交叉と突然変異で生成
        for i in range(n_elite, pop_size):
            # トーナメント選択
            t1 = np.random.randint(0, pop_size)
            t2 = np.random.randint(0, pop_size)
            parent1_idx = t1 if fitness[t1] < fitness[t2] else t2

            t1 = np.random.randint(0, pop_size)
            t2 = np.random.randint(0, pop_size)
            parent2_idx = t1 if fitness[t1] < fitness[t2] else t2

            # 交叉
            if np.random.random() < 0.5:
                child_xs, child_ys, child_degs = crossover_blend(
                    population_xs[parent1_idx],
                    population_ys[parent1_idx],
                    population_degs[parent1_idx],
                    population_xs[parent2_idx],
                    population_ys[parent2_idx],
                    population_degs[parent2_idx],
                    0.3,
                )
            else:
                child_xs, child_ys, child_degs = crossover_uniform(
                    population_xs[parent1_idx],
                    population_ys[parent1_idx],
                    population_degs[parent1_idx],
                    population_xs[parent2_idx],
                    population_ys[parent2_idx],
                    population_degs[parent2_idx],
                )

            # 突然変異
            child_xs, child_ys, child_degs = mutate(
                child_xs, child_ys, child_degs, mutation_rate, pos_sigma, deg_sigma
            )

            # 修復
            child_xs, child_ys, child_degs, valid = repair_solution(child_xs, child_ys, child_degs)

            new_population_xs[i] = child_xs
            new_population_ys[i] = child_ys
            new_population_degs[i] = child_degs

            if valid:
                new_fitness[i] = evaluate_solution(child_xs, child_ys, child_degs)
            else:
                new_fitness[i] = math.inf

        population_xs = new_population_xs
        population_ys = new_population_ys
        population_degs = new_population_degs
        fitness = new_fitness

        # 最良解の更新
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_score:
            best_xs = population_xs[gen_best_idx].copy()
            best_ys = population_ys[gen_best_idx].copy()
            best_degs = population_degs[gen_best_idx].copy()
            best_score = fitness[gen_best_idx]

    # 最後に局所探索で磨く
    if local_search_iters > 0:
        best_xs, best_ys, best_degs, ls_score = local_search(
            best_xs, best_ys, best_degs, local_search_iters, pos_sigma * 0.5, deg_sigma * 0.5
        )
        if ls_score < best_score:
            best_score = ls_score

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
    print("Genetic Algorithm Optimizer (exp012_genetic)")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nBaseline: {baseline_path}")

    all_xs, all_ys, all_degs = load_submission_data(baseline_path)
    baseline_total = calculate_total_score(all_xs, all_ys, all_degs)
    print(f"Baseline total score: {baseline_total:.6f}")

    # パラメータ
    ga_cfg = CONFIG["genetic"]
    n_min = int(ga_cfg["n_min"])
    n_max = int(ga_cfg["n_max"])
    pop_size = int(ga_cfg["pop_size"])
    n_generations = int(ga_cfg["n_generations"])
    mutation_rate = float(ga_cfg["mutation_rate"])
    pos_sigma = float(ga_cfg["pos_sigma"])
    deg_sigma = float(ga_cfg["deg_sigma"])
    elite_ratio = float(ga_cfg["elite_ratio"])
    local_search_iters = int(ga_cfg["local_search_iters"])
    seed_base = int(ga_cfg.get("seed_base", 42))

    print(f"\nOptimizing groups {n_min} to {n_max}...")
    print(f"  Population size: {pop_size}")
    print(f"  Generations: {n_generations}")
    print(f"  Mutation rate: {mutation_rate}")

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

        # GA最適化
        init_xs = new_xs[start : start + n].copy()
        init_ys = new_ys[start : start + n].copy()
        init_degs = new_degs[start : start + n].copy()

        seed = seed_base + n * 1000
        opt_xs, opt_ys, opt_degs, opt_score = genetic_algorithm(
            init_xs,
            init_ys,
            init_degs,
            pop_size,
            n_generations,
            mutation_rate,
            pos_sigma,
            deg_sigma,
            elite_ratio,
            local_search_iters,
            seed,
        )

        if opt_score < orig_score - 1e-9:
            improvement = orig_score - opt_score
            total_improved += improvement
            improved_groups += 1
            new_xs[start : start + n] = opt_xs
            new_ys[start : start + n] = opt_ys
            new_degs[start : start + n] = opt_degs
            print(f"  Group {n}: {orig_score:.6f} -> {opt_score:.6f} (improved {improvement:.6f})")

    final_score = calculate_total_score(new_xs, new_ys, new_degs)

    print("\n" + "=" * 80)
    print(f"  Baseline total:    {baseline_total:.6f}")
    print(f"  After GA:          {final_score:.6f}")
    print(f"  Total improvement: {baseline_total - final_score:+.6f}")
    print(f"  Improved groups:   {improved_groups}")
    print("=" * 80)

    if final_score < baseline_total:
        out_path = CONFIG["paths"]["output"]
        save_submission(out_path, new_xs, new_ys, new_degs)
        print(f"Saved to {out_path}")
    else:
        print("No improvement - keeping baseline")

