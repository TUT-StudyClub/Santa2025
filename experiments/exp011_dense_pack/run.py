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
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
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
def optimize_pattern_sa(
    init_xs: np.ndarray,
    init_ys: np.ndarray,
    init_degs: np.ndarray,
    n_iters: int,
    pos_delta: float,
    ang_delta: float,
    T_max: float,
    T_min: float,
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

    T_factor = -math.log(T_max / T_min)

    for step in range(n_iters):
        T = T_max * math.exp(T_factor * step / n_iters)
        decay = 1.0 - 0.8 * (step / n_iters)
        cur_pos = pos_delta * decay
        cur_ang = ang_delta * decay

        tree_idx = np.random.randint(0, n)
        move_type = np.random.randint(0, 4)

        old_x, old_y, old_deg = xs[tree_idx], ys[tree_idx], degs[tree_idx]

        if move_type == 0 or move_type == 3:
            xs[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type == 1 or move_type == 3:
            ys[tree_idx] += (np.random.random() * 2.0 - 1.0) * cur_pos
        if move_type == 2 or move_type == 3:
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

        if delta < 0 or (T > 1e-10 and np.random.random() < math.exp(-delta / T)):
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
    T_max = float(opt_cfg["T_max"])
    T_min = float(opt_cfg["T_min"])
    seed_base = int(opt_cfg.get("seed_base", 42))

    # Two-stage Optimizationの設定
    # Screening: 探索空間を広げるため、計算コストを抑えて構造の有望度のみを判定
    screening_iters = 2000
    # Final: 有望な解に対して十分なステップ数を割り当て、局所解の探索を行う
    final_iters = n_iters

    print(f"\nOptimizing groups {n_min} to {n_max}...")
    print(f"  Screening iters: {screening_iters}, Final iters: {final_iters}")

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    total_improved = 0.0
    improved_groups = 0

    # ハイパーパラメータ探索グリッド
    spacings = [0.65, 0.70, 0.75, 0.80]
    angles = [0.0, 45.0, 90.0, 135.0]

    for n in tqdm(range(n_min, n_max + 1), desc="Optimizing"):
        # N個の要素を持つ群の開始インデックスを算出 (等差数列の和: Sum(1..n-1))
        # データ構造がフラットである前提に基づく
        start_idx = n * (n - 1) // 2

        # ベースライン(現状)のスコア算出
        orig_verts = [
            get_tree_vertices(new_xs[start_idx + i], new_ys[start_idx + i], new_degs[start_idx + i])
            for i in range(n)
        ]
        orig_score = calculate_score(orig_verts)

        # 最適化候補の初期化
        best_candidate_params = None
        best_candidate_score = math.inf

        # アスペクト比探索範囲の決定
        # 理想的な正方形配置(sqrt(n))周辺の列数を探索し、バウンディングボックスの無駄を最小化する
        base_cols = int(math.ceil(math.sqrt(n)))
        # Nが小さい場合は狭く、大きい場合は広く探索範囲を取るヒューリスティック
        col_search_range = range(max(1, base_cols - 2), base_cols + 4)

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
                        candidates.append((xs1, ys1, degs1))

                    # パターンB: Hexagonal (六角形配置)
                    xs2, ys2, degs2 = generate_hexagonal_pattern(n, spacing, angle, n_cols)
                    if len(xs2) == n:
                        candidates.append((xs2, ys2, degs2))

                    # 各候補に対して軽量なSAを実行
                    for c_xs, c_ys, c_degs in candidates:
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
                T_max,
                T_min,
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

    # Always save the latest result regardless of score improvement
    out_path = CONFIG["paths"]["output"]
    save_submission(out_path, new_xs, new_ys, new_degs)
    print(f"Saved to {out_path}")

    if final_score >= baseline_total:
        print("No improvement from baseline")
