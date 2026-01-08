"""
理論的な最小スコアと現在のベースラインの効率を分析
"""

import math

import numpy as np
import pandas as pd
from numba import njit

# Tree shape parameters
TRUNK_W = 0.15
BASE_W = 0.7
MID_W = 0.4
TOP_W = 0.25
TIP_Y = 0.8
TIER_1_Y = 0.5
TIER_2_Y = 0.25
BASE_Y = 0.0
TRUNK_BOTTOM_Y = -0.2

# 木の高さと幅
TREE_HEIGHT = TIP_Y - TRUNK_BOTTOM_Y  # 1.0
TREE_WIDTH = BASE_W  # 0.7

# 木の面積（概算）
# 三角形3つ + 幹の長方形
TREE_AREA = (
    0.5 * TOP_W * (TIP_Y - TIER_1_Y)  # 上の三角形
    + 0.5 * MID_W * (TIER_1_Y - TIER_2_Y)  # 中の三角形
    + 0.5 * BASE_W * (TIER_2_Y - BASE_Y)  # 下の三角形
    + TRUNK_W * (BASE_Y - TRUNK_BOTTOM_Y)  # 幹
)

print("=" * 60)
print("Tree Shape Analysis")
print("=" * 60)
print(f"Tree height: {TREE_HEIGHT}")
print(f"Tree width (base): {TREE_WIDTH}")
print(f"Tree area (approx): {TREE_AREA:.4f}")

# 理論的な最小スコア
# スコア = side^2 / n
# 理想的には、n本の木を正方形に配置したとき
# side = sqrt(n * tree_area / packing_efficiency)
# packing_efficiency は通常 0.7-0.9 程度

print("\n" + "=" * 60)
print("Theoretical Minimum Scores")
print("=" * 60)

# 異なるパッキング効率での理論値
for packing_eff in [0.5, 0.6, 0.7, 0.8, 0.9]:
    total_score = 0.0
    for n in range(1, 201):
        # 理論的な最小サイド長
        total_area = n * TREE_AREA / packing_eff
        side = math.sqrt(total_area)
        score = side * side / n
        total_score += score
    print(f"Packing efficiency {packing_eff * 100:.0f}%: total score = {total_score:.2f}")

# 現在のベースラインを分析
print("\n" + "=" * 60)
print("Current Baseline Analysis")
print("=" * 60)


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
def compute_bounding_box(all_vertices: list[np.ndarray]) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for verts in all_vertices:
        for i in range(verts.shape[0]):
            x, y = verts[i, 0], verts[i, 1]
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
def get_side_and_dims(all_vertices: list[np.ndarray]) -> tuple[float, float, float]:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    width = max_x - min_x
    height = max_y - min_y
    return max(width, height), width, height


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


filepath = "submissions/baseline.csv"
all_xs, all_ys, all_degs = load_submission_data(filepath)

# グループごとの効率を計算
header = (
    f"\n{'Group':>6} {'Score':>10} {'Side':>8} {'Width':>8} {'Height':>8} {'Aspect':>8} {'Eff%':>8}"
)
print(header)
print("-" * 70)

total_score = 0.0
aspect_ratios = []

for n in range(1, 201):
    start = n * (n - 1) // 2
    vertices = [
        get_tree_vertices(all_xs[start + i], all_ys[start + i], all_degs[start + i])
        for i in range(n)
    ]
    side, width, height = get_side_and_dims(vertices)
    score = side * side / n
    total_score += score

    # アスペクト比（1.0が理想）
    aspect = min(width, height) / max(width, height) if max(width, height) > 0 else 0
    aspect_ratios.append(aspect)

    # 効率 = 理論最小 / 実際
    theoretical_min = TREE_AREA
    efficiency = theoretical_min / score * 100

    if n <= 20 or n % 20 == 0:
        print(
            f"{n:>6} {score:>10.6f} {side:>8.4f} {width:>8.4f} "
            f"{height:>8.4f} {aspect:>8.3f} {efficiency:>7.1f}%"
        )

print(f"\nTotal score: {total_score:.6f}")
print(f"Average aspect ratio: {np.mean(aspect_ratios):.3f}")

# アスペクト比が悪いグループを特定
print("\n" + "=" * 60)
print("Groups with worst aspect ratios (room for improvement)")
print("=" * 60)

group_aspects = [(n, aspect_ratios[n - 1]) for n in range(1, 201)]
group_aspects.sort(key=lambda x: x[1])

print("Worst 10 aspect ratios (far from square):")
for n, aspect in group_aspects[:10]:
    start = n * (n - 1) // 2
    vertices = [
        get_tree_vertices(all_xs[start + i], all_ys[start + i], all_degs[start + i])
        for i in range(n)
    ]
    side, width, height = get_side_and_dims(vertices)
    score = side * side / n
    print(f"  Group {n:>3}: aspect={aspect:.3f}, score={score:.6f}, side={side:.4f}")
