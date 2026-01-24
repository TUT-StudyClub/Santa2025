"""
各グループのスコアを分析して、改善の余地があるグループを特定する
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

# 木の形状パラメータ
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


def load_submission_data(filepath: str, fallback_path: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_path = filepath
    if not Path(target_path).exists():
        if fallback_path is None or not Path(fallback_path).exists():
            raise FileNotFoundError(f"submission not found: {filepath}")
        target_path = fallback_path

    df = pd.read_csv(target_path)
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


def main():  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="提出ファイルのスコアを分析します。")
    parser.add_argument("--input", default="submissions/submission.csv", help="提出CSVのパス")
    parser.add_argument(
        "--fallback",
        default="submissions/baseline.csv",
        help="--input が見つからない場合に使うfallback（例: submissions/baseline.csv）。空文字で無効化",
    )
    parser.add_argument(
        "--target-range",
        default=None,
        help="レンジ合計のカンマ区切り (例: 8.0196,14.3787,13.9196,17.0612,16.8016)",
    )
    parser.add_argument(
        "--worst-by-range",
        type=int,
        default=0,
        help="各レンジでスコアが高い上位Nグループを表示",
    )
    args = parser.parse_args()

    filepath = args.input
    print(f"分析対象: {filepath}")

    fallback = str(args.fallback).strip() or None
    if not Path(filepath).exists() and fallback is not None and Path(fallback).exists():
        print(f"  ※ {filepath} が見つからないため fallback を使用します: {fallback}")

    all_xs, all_ys, all_degs = load_submission_data(filepath, fallback_path=fallback)

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

    print(f"\n合計スコア: {total:.6f}")
    print("\nレンジ合計:")
    range_totals = []
    for start, end in RANGES:
        range_score = sum(score for n, score, _ in scores if start <= n <= end)
        range_totals.append(range_score)
        print(f"  {start:>3}-{end:<3}: {range_score:.6f}")

    if args.target_range:
        parts = [p.strip() for p in args.target_range.split(",") if p.strip()]
        if len(parts) != len(RANGES):
            raise SystemExit(f"--target-range は {len(RANGES)} 個の数値が必要です: {args.target_range}")
        target_vals = [float(p) for p in parts]
        target_total = sum(target_vals)
        diff_total = total - target_total
        print("\nターゲット差分:")
        for (start, end), my_val, tgt_val in zip(RANGES, range_totals, target_vals, strict=True):
            diff = my_val - tgt_val
            print(f"  {start:>3}-{end:<3}: 自分={my_val:.6f} ターゲット={tgt_val:.6f} 差分={diff:+.6f}")
        print(f"  合計: 自分={total:.6f} ターゲット={target_total:.6f} 差分={diff_total:+.6f}")

    print(f"\n{'グループ':>6} {'スコア':>12} {'辺長':>10} {'効率':>12}")
    print("-" * 45)

    # 理論的な最小スコア（完璧な正方形配置の場合）を計算
    # 1本の木の面積は約 base_w * (tip_y - trunk_bottom_y) = 0.7 * 1.0 = 0.7
    tree_area = 0.7 * 1.0

    # スコアが高い（悪い）グループを表示
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nワースト20グループ（スコアが高い順）:")
    for n, score, side in sorted_scores[:20]:
        # 効率 = 理論最小 / 実際のスコア
        theoretical_min = tree_area  # 理想的には side^2/n = tree_area
        efficiency = theoretical_min / score * 100
        print(f"{n:>6} {score:>12.6f} {side:>10.4f} {efficiency:>11.1f}%")

    print("\nベスト20グループ（スコアが低い順）:")
    for n, score, side in sorted_scores[-20:]:
        theoretical_min = tree_area
        efficiency = theoretical_min / score * 100
        print(f"{n:>6} {score:>12.6f} {side:>10.4f} {efficiency:>11.1f}%")

    # スコアの分布
    print("\nグループサイズ別の平均スコア:")
    for start_n in range(1, 201, 20):
        end_n = min(start_n + 19, 200)
        group_scores = [s[1] for s in scores if start_n <= s[0] <= end_n]
        avg_score = sum(group_scores) / len(group_scores)
        print(f"  グループ {start_n:>3}-{end_n:>3}: 平均スコア = {avg_score:.6f}")

    if args.worst_by_range > 0:
        print("\nレンジ別ワーストグループ:")
        for start, end in RANGES:
            range_scores = [s for s in scores if start <= s[0] <= end]
            worst = sorted(range_scores, key=lambda x: x[1], reverse=True)[: args.worst_by_range]
            print(f"  {start:>3}-{end:<3}:")
            for n, score, side in worst:
                theoretical_min = tree_area
                efficiency = theoretical_min / score * 100
                print(f"    {n:>3} {score:>12.6f} {side:>10.4f} {efficiency:>11.1f}%")


if __name__ == "__main__":
    main()
