"""
各グループのスコアを分析して、改善の余地があるグループを特定する
"""

import argparse

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

RANGES = [(1, 20), (21, 60), (61, 100), (101, 150), (151, 200)]


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
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def get_side_length(all_vertices: list[np.ndarray]) -> float:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True)
def calculate_score(all_vertices: list[np.ndarray]) -> float:
    side = get_side_length(all_vertices)
    return side * side / len(all_vertices)


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


def main():
    parser = argparse.ArgumentParser(description="提出ファイルのスコアを分析します。")
    parser.add_argument("--input", default="submissions/baseline.csv", help="提出CSVのパス")
    args = parser.parse_args()

    filepath = args.input
    print(f"Analyzing: {filepath}")

    all_xs, all_ys, all_degs = load_submission_data(filepath)

    # 各グループのスコアを計算
    scores = []
    total = 0.0

    for n in range(1, 201):
        start = n * (n - 1) // 2
        vertices = [get_tree_vertices(all_xs[start + i], all_ys[start + i], all_degs[start + i]) for i in range(n)]
        score = calculate_score(vertices)
        side = get_side_length(vertices)
        scores.append((n, score, side))
        total += score

    print(f"\nTotal score: {total:.6f}")
    print("\nRange totals:")
    for start, end in RANGES:
        range_score = sum(score for n, score, _ in scores if start <= n <= end)
        print(f"  {start:>3}-{end:<3}: {range_score:.6f}")
    print(f"\n{'Group':>6} {'Score':>12} {'Side':>10} {'Efficiency':>12}")
    print("-" * 45)

    # 理論的な最小スコア（完璧な正方形配置の場合）を計算
    # 1本の木の面積は約 base_w * (tip_y - trunk_bottom_y) = 0.7 * 1.0 = 0.7
    tree_area = 0.7 * 1.0

    # スコアが高い（悪い）グループを表示
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nWorst 20 groups (highest scores):")
    for n, score, side in sorted_scores[:20]:
        # 効率 = 理論最小 / 実際のスコア
        theoretical_min = tree_area  # 理想的には side^2/n = tree_area
        efficiency = theoretical_min / score * 100
        print(f"{n:>6} {score:>12.6f} {side:>10.4f} {efficiency:>11.1f}%")

    print("\nBest 20 groups (lowest scores):")
    for n, score, side in sorted_scores[-20:]:
        theoretical_min = tree_area
        efficiency = theoretical_min / score * 100
        print(f"{n:>6} {score:>12.6f} {side:>10.4f} {efficiency:>11.1f}%")

    # スコアの分布
    print("\nScore distribution by group size:")
    for start_n in range(1, 201, 20):
        end_n = min(start_n + 19, 200)
        group_scores = [s[1] for s in scores if start_n <= s[0] <= end_n]
        avg_score = sum(group_scores) / len(group_scores)
        print(f"  Groups {start_n:>3}-{end_n:>3}: avg score = {avg_score:.6f}")


if __name__ == "__main__":
    main()
