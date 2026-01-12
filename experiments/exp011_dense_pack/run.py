"""
exp011_dense_pack: より密なパッキングを探索

木の形状を考慮し、互い違いの配置や異なる角度の組み合わせで
より密なパッキングを実現する。
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
    return os.path.join("experiments", "exp011_dense_pack", "exp", f"{config_name}.yaml")


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
    p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float, p4x: float, p4y: float
) -> bool:
    d1x, d1y = p2x - p1x, p2y - p1y
    d2x, d2y = p4x - p3x, p4y - p3y
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-10:  # noqa: PLR2004
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
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
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


def generate_asymmetric_pattern(
    n: int,
    spacing: float,
    angle_offset: float,
    fixed_cols: int,
    rng: np.random.Generator,
    row_jitter: float,
    col_jitter: float,
    angle_jitter: float,
    row_spacing_factor: float,
    row_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = fixed_cols if fixed_cols > 0 else int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    row_jitter = min(max(row_jitter, 0.0), 0.9)
    col_jitter = min(max(col_jitter, 0.0), 0.9)
    row_spacing = spacing * max(row_spacing_factor, 1e-6)

    col_offsets = np.zeros(cols, dtype=np.float64)
    for col in range(1, cols):
        step = spacing * (1.0 + rng.uniform(-col_jitter, col_jitter))
        col_offsets[col] = col_offsets[col - 1] + step

    row_offsets = np.zeros(rows, dtype=np.float64)
    for row in range(1, rows):
        step = row_spacing * (1.0 + rng.uniform(-row_jitter, row_jitter))
        row_offsets[row] = row_offsets[row - 1] + step

    row_angle = rng.uniform(-angle_jitter, angle_jitter, size=rows)
    col_angle = rng.uniform(-angle_jitter, angle_jitter, size=cols)
    if row_shift > 0.0:
        row_shifts = rng.uniform(-row_shift, row_shift, size=rows)
    else:
        row_shifts = np.zeros(rows, dtype=np.float64)

    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    degs = np.zeros(n, dtype=np.float64)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            xs[idx] = col_offsets[col] + row_shifts[row]
            ys[idx] = row_offsets[row]
            degs[idx] = (angle_offset + row_angle[row] + col_angle[col]) % 360.0
            idx += 1

    return xs, ys, degs


def apply_pattern_jitter(
    xs: np.ndarray,
    ys: np.ndarray,
    degs: np.ndarray,
    jitter_xy: float,
    jitter_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if jitter_xy <= 0.0 and jitter_deg <= 0.0:
        return xs, ys, degs

    new_xs = xs.copy()
    new_ys = ys.copy()
    new_degs = degs.copy()
    if jitter_xy > 0.0:
        new_xs += rng.uniform(-jitter_xy, jitter_xy, size=new_xs.shape)
        new_ys += rng.uniform(-jitter_xy, jitter_xy, size=new_ys.shape)
    if jitter_deg > 0.0:
        new_degs = (new_degs + rng.uniform(-jitter_deg, jitter_deg, size=new_degs.shape)) % 360.0
    return new_xs, new_ys, new_degs


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
def optimize_pattern_sa(  # noqa: PLR0913, PLR0912, PLR0915, N803
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    t_max: float,
    t_min: float,
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
    while has_any_overlap(all_vertices) and spread < 5.0:  # noqa: PLR2004
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

    t_factor = -math.log(t_max / t_min)

    for step in range(n_iters):
        temp = t_max * math.exp(t_factor * step / n_iters)
        decay = 1.0 - 0.8 * (step / n_iters)
        cur_pos = pos_delta * decay
        cur_ang = ang_delta * decay

        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 4)

        old_x, old_y, old_deg = xs[tree_idx], ys[tree_idx], degs[tree_idx]

        if move_type in {0, 3}:
            xs[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type in {1, 3}:
            ys[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type in {2, 3}:
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

        if delta < 0 or (temp > 1e-10 and np.random.random() < math.exp(-delta / temp)):  # noqa: PLR2004
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 設定とベースラインの読み込み（省略なし、ロジック部分は変更なし）
    print("Dense Packing Optimizer (exp011_dense_pack)")
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    all_xs, all_ys, all_degs = load_submission_data(baseline_path)
    baseline_total = calculate_total_score(all_xs, all_ys, all_degs)
    print(f"Baseline total score: {baseline_total:.6f}")

    # 最適化パラメータの展開
    opt_cfg = CONFIG["optimization"]
    n_min = int(opt_cfg["n_min"])
    n_max = int(opt_cfg["n_max"])
    n_iters = int(opt_cfg["n_iters"])
    pos_delta = float(opt_cfg["pos_delta"])
    ang_delta = float(opt_cfg["ang_delta"])
    t_max = float(opt_cfg["T_max"])
    t_min = float(opt_cfg["T_min"])
    seed_base = int(opt_cfg.get("seed_base", 42))
    screening_iters = int(opt_cfg.get("screening_iters", 2000))
    final_iters = int(opt_cfg.get("final_iters", n_iters))
    spacings = opt_cfg.get("spacings", [0.65, 0.70, 0.75, 0.80])
    angles = opt_cfg.get("angles", [0.0, 45.0, 90.0, 135.0])
    col_minus = int(opt_cfg.get("col_minus", 2))
    col_plus = int(opt_cfg.get("col_plus", 4))
    baseline_restarts = int(opt_cfg.get("baseline_restarts", 1))
    baseline_jitter_xy = float(opt_cfg.get("baseline_jitter_xy", 0.0))
    baseline_jitter_deg = float(opt_cfg.get("baseline_jitter_deg", 0.0))
    candidate_jitter_xy = float(opt_cfg.get("candidate_jitter_xy", 0.0))
    candidate_jitter_deg = float(opt_cfg.get("candidate_jitter_deg", 0.0))
    candidate_jitter_seed_offset = int(opt_cfg.get("candidate_jitter_seed_offset", 0))
    asym_enabled = bool(opt_cfg.get("asym_enabled", False))
    asym_row_jitter = float(opt_cfg.get("asym_row_jitter", 0.0))
    asym_col_jitter = float(opt_cfg.get("asym_col_jitter", 0.0))
    asym_angle_jitter = float(opt_cfg.get("asym_angle_jitter", 0.0))
    asym_row_spacing_factor = float(opt_cfg.get("asym_row_spacing_factor", 0.75))
    asym_row_shift = float(opt_cfg.get("asym_row_shift", 0.0))
    asym_seed_offset = int(opt_cfg.get("asym_seed_offset", 10000))

    spacings = [float(s) for s in spacings]
    angles = [float(a) for a in angles]

    # Two-stage Optimizationの設定
    # Screening: 探索空間を広げるため、計算コストを抑えて構造の有望度のみを判定
    print(f"\nOptimizing groups {n_min} to {n_max}...")
    print(f"  Screening iters: {screening_iters}, Final iters: {final_iters}")
    print(
        f"  Baseline restarts: {baseline_restarts} (jitter_xy={baseline_jitter_xy}, jitter_deg={baseline_jitter_deg})"
    )
    if candidate_jitter_xy > 0.0 or candidate_jitter_deg > 0.0:
        print(
            "  Candidate jitter: "
            f"xy={candidate_jitter_xy}, deg={candidate_jitter_deg}, seed_offset={candidate_jitter_seed_offset}"
        )
    if asym_enabled:
        print(
            "  Asym pattern: "
            f"row_jitter={asym_row_jitter}, col_jitter={asym_col_jitter}, "
            f"angle_jitter={asym_angle_jitter}, row_spacing_factor={asym_row_spacing_factor}, "
            f"row_shift={asym_row_shift}"
        )

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    total_improved = 0.0
    improved_groups = 0

    # ハイパーパラメータ探索グリッド

    for n in tqdm(range(n_min, n_max + 1), desc="Optimizing"):
        # N個の要素を持つ群の開始インデックスを算出 (等差数列の和: Sum(1..n-1))
        # データ構造がフラットである前提に基づく
        start_idx = n * (n - 1) // 2

        # ベースライン(現状)のスコア算出
        orig_verts = [
            get_tree_vertices(new_xs[start_idx + i], new_ys[start_idx + i], new_degs[start_idx + i]) for i in range(n)
        ]
        orig_score = calculate_score(orig_verts)

        # 最適化候補の初期化
        best_candidate_params = None
        best_candidate_score = math.inf

        # アスペクト比探索範囲の決定
        # 理想的な正方形配置(sqrt(n))周辺の列数を探索し、バウンディングボックスの無駄を最小化する
        base_cols = int(math.ceil(math.sqrt(n)))
        # Nが小さい場合は狭く、大きい場合は広く探索範囲を取るヒューリスティック
        col_search_range = range(max(1, base_cols - col_minus), base_cols + col_plus)

        # --- Phase 1: Screening (広範な探索) ---
        for spacing in spacings:
            for angle in angles:
                for n_cols in col_search_range:
                    # シード値をパラメータごとに分離し、再現性を確保
                    seed = seed_base + n * 1000 + int(spacing * 100) + int(angle) + n_cols

                    candidates = []

                    # パターンA: Interlocking (互い違い配置)
                    # generator関数は fixed_cols 引数を受け取るように修正済みであることを前提とする
                    xs1, ys1, degs1 = generate_interlocking_pattern(n, spacing, angle, n_cols)
                    if len(xs1) == n:
                        candidates.append((xs1, ys1, degs1, 0))

                    # パターンB: Hexagonal (六角形配置)
                    xs2, ys2, degs2 = generate_hexagonal_pattern(n, spacing, angle, n_cols)
                    if len(xs2) == n:
                        candidates.append((xs2, ys2, degs2, 1))

                    # パターンC: Asymmetric grid (非対称配置)
                    if asym_enabled:
                        rng_asym = np.random.default_rng(seed + asym_seed_offset)
                        xs3, ys3, degs3 = generate_asymmetric_pattern(
                            n,
                            spacing,
                            angle,
                            n_cols,
                            rng_asym,
                            asym_row_jitter,
                            asym_col_jitter,
                            asym_angle_jitter,
                            asym_row_spacing_factor,
                            asym_row_shift,
                        )
                        if len(xs3) == n:
                            candidates.append((xs3, ys3, degs3, 2))

                    # 各候補に対して軽量なSAを実行
                    for c_xs, c_ys, c_degs, c_id in candidates:
                        xs_c = c_xs
                        ys_c = c_ys
                        degs_c = c_degs
                        if candidate_jitter_xy > 0.0 or candidate_jitter_deg > 0.0:
                            rng = np.random.default_rng(seed + candidate_jitter_seed_offset + c_id)
                            xs_c, ys_c, degs_c = apply_pattern_jitter(
                                xs_c,
                                ys_c,
                                degs_c,
                                candidate_jitter_xy,
                                candidate_jitter_deg,
                                rng,
                            )
                        _, _, _, score = optimize_pattern_sa(
                            xs_c,
                            ys_c,
                            degs_c,
                            screening_iters,
                            pos_delta,
                            ang_delta,
                            t_max,
                            t_min,
                            seed,
                        )

                        # 暫定ベストの更新
                        if score < best_candidate_score:
                            best_candidate_score = score
                            best_candidate_params = (c_xs, c_ys, c_degs, seed)

        # --- Phase 2: Final Optimization (局所解の精査) ---
        best_score = orig_score

        # 作業用バッファ（スライス参照ではなくコピーを作成して操作）
        best_xs_group = new_xs[start_idx : start_idx + n].copy()
        best_ys_group = new_ys[start_idx : start_idx + n].copy()
        best_degs_group = new_degs[start_idx : start_idx + n].copy()

        # Screening勝者の本番最適化
        if best_candidate_params is not None:
            init_xs, init_ys, init_degs, best_seed = best_candidate_params

            opt_xs, opt_ys, opt_degs, score = optimize_pattern_sa(
                init_xs,
                init_ys,
                init_degs,
                final_iters,
                pos_delta,
                ang_delta,
                t_max,
                t_min,
                best_seed,
            )

            if score < best_score:
                best_score = score
                best_xs_group[:] = opt_xs[:]
                best_ys_group[:] = opt_ys[:]
                best_degs_group[:] = opt_degs[:]

        # 既存配置(Baseline)からの摂動による最適化も並行して実施
        # 完全にランダムな再配置よりも、既存の良解を微修正する方が有利な場合があるため
        base_xs = new_xs[start_idx : start_idx + n].copy()
        base_ys = new_ys[start_idx : start_idx + n].copy()
        base_degs = new_degs[start_idx : start_idx + n].copy()

        for r in range(max(1, baseline_restarts)):
            if baseline_jitter_xy > 0.0 or baseline_jitter_deg > 0.0:
                rng = np.random.default_rng(seed_base + n * 10000 + r)
                jitter_xs = base_xs + rng.uniform(-baseline_jitter_xy, baseline_jitter_xy, size=n)
                jitter_ys = base_ys + rng.uniform(-baseline_jitter_xy, baseline_jitter_xy, size=n)
                jitter_degs = (base_degs + rng.uniform(-baseline_jitter_deg, baseline_jitter_deg, size=n)) % 360.0
            else:
                jitter_xs = base_xs
                jitter_ys = base_ys
                jitter_degs = base_degs

            opt_xs, opt_ys, opt_degs, score = optimize_pattern_sa(
                jitter_xs,
                jitter_ys,
                jitter_degs,
                final_iters,
                pos_delta,
                ang_delta,
                t_max,
                t_min,
                seed_base + n * 9999 + r,
            )

            if score < best_score:
                best_score = score
                best_xs_group[:] = opt_xs[:]
                best_ys_group[:] = opt_ys[:]
                best_degs_group[:] = opt_degs[:]

        # --- Result Update (Greedy Strategy) ---
        # 浮動小数点の誤差を考慮し、有意な改善（1e-9以上）があった場合のみ更新
        if best_score < orig_score - 1e-9:
            improvement = orig_score - best_score
            total_improved += improvement
            improved_groups += 1

            # 全体配列への書き戻し
            new_xs[start_idx : start_idx + n] = best_xs_group
            new_ys[start_idx : start_idx + n] = best_ys_group
            new_degs[start_idx : start_idx + n] = best_degs_group

            print(f"  Group {n}: {orig_score:.6f} -> {best_score:.6f} (improved {improvement:.6f})")

    # 最終結果の集計と保存
    final_score = calculate_total_score(new_xs, new_ys, new_degs)

    print("\noptimization summary")
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
        if final_score < ref_score - 1e-9:
            save_submission(out_path, new_xs, new_ys, new_degs)
            print(f"submissionを更新しました: {out_path}")
        else:
            print("submissionより改善なしのため上書きしません")
    elif baseline_improved:
        save_submission(out_path, new_xs, new_ys, new_degs)
        print(f"submissionを作成しました: {out_path}")
    else:
        print("Baselineから改善なしのためsubmissionを作成しません")
