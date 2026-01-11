import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TREE_POINTS = np.array(
    [
        [0.0, 0.8],
        [0.25 / 2.0, 0.5],
        [0.25 / 4.0, 0.5],
        [0.4 / 2.0, 0.25],
        [0.4 / 4.0, 0.25],
        [0.7 / 2.0, 0.0],
        [0.15 / 2.0, 0.0],
        [0.15 / 2.0, -0.2],
        [-0.15 / 2.0, -0.2],
        [-0.15 / 2.0, 0.0],
        [-0.7 / 2.0, 0.0],
        [-0.4 / 4.0, 0.25],
        [-0.4 / 2.0, 0.25],
        [-0.25 / 4.0, 0.5],
        [-0.25 / 2.0, 0.5],
    ],
    dtype=np.float64,
)

RANGES = [(1, 20), (21, 60), (61, 100), (101, 150), (151, 200)]


def parse_series(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lstrip("s").astype(float).to_numpy(dtype=np.float64)


def build_tree_vertices(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return TREE_POINTS @ rot.T + np.array([cx, cy])


def build_group_points(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> np.ndarray:
    points = [build_tree_vertices(x, y, deg) for x, y, deg in zip(xs, ys, degs)]
    return np.vstack(points)


def rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    return np.column_stack((x_vals * cos_a - y_vals * sin_a, x_vals * sin_a + y_vals * cos_a))


def compute_side_from_points(points: np.ndarray) -> float:
    min_x = float(points[:, 0].min())
    max_x = float(points[:, 0].max())
    min_y = float(points[:, 1].min())
    max_y = float(points[:, 1].max())
    return max(max_x - min_x, max_y - min_y)


def angle_grid(step: float) -> np.ndarray:
    step = max(step, 1e-6)
    return np.arange(0.0, 180.0 + step / 2.0, step, dtype=np.float64)


def find_best_rotation(
    points: np.ndarray, coarse_step: float, fine_step: float, fine_window: float
) -> tuple[float, float]:
    best_angle = 0.0
    best_side = math.inf
    for angle in angle_grid(coarse_step):
        side = compute_side_from_points(rotate_points(points, angle))
        if side < best_side:
            best_side = side
            best_angle = float(angle)

    if fine_step > 0.0:
        start = max(0.0, best_angle - fine_window)
        end = min(180.0, best_angle + fine_window)
        for angle in angle_grid(fine_step):
            if angle < start or angle > end:
                continue
            side = compute_side_from_points(rotate_points(points, angle))
            if side < best_side:
                best_side = side
                best_angle = float(angle)
    return best_angle, best_side


def rotate_centers(xs: np.ndarray, ys: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_xs = xs * cos_a - ys * sin_a
    new_ys = xs * sin_a + ys * cos_a
    return new_xs, new_ys


def summarize_ranges(group_scores: dict[int, float]) -> tuple[list[tuple[str, float]], float]:
    rows = []
    total = 0.0
    for start, end in RANGES:
        score = sum(group_scores.get(n, 0.0) for n in range(start, end + 1))
        rows.append((f"{start}–{end}", score))
        total += score
    return rows, total


def format_with_prefix(values: np.ndarray) -> list[str]:
    return [f"s{value}" for value in values]


def optimize_rotation(
    df: pd.DataFrame, coarse_step: float, fine_step: float, fine_window: float
) -> tuple[pd.DataFrame, dict[int, float], dict[int, float]]:
    new_df = df.copy()
    before_scores = {}
    after_scores = {}
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group_df = df[df["id"].str.startswith(prefix)].sort_values("id")
        if group_df.empty:
            continue
        xs = parse_series(group_df["x"])
        ys = parse_series(group_df["y"])
        degs = parse_series(group_df["deg"])
        points = build_group_points(xs, ys, degs)
        before_side = compute_side_from_points(points)
        before_scores[n] = before_side * before_side / len(xs)

        best_angle, best_side = find_best_rotation(points, coarse_step, fine_step, fine_window)
        new_xs, new_ys = rotate_centers(xs, ys, best_angle)
        new_degs = (degs + best_angle) % 360.0
        after_scores[n] = best_side * best_side / len(xs)

        new_df.loc[group_df.index, "x"] = format_with_prefix(new_xs)
        new_df.loc[group_df.index, "y"] = format_with_prefix(new_ys)
        new_df.loc[group_df.index, "deg"] = format_with_prefix(new_degs)

    return new_df, before_scores, after_scores


def print_summary(title: str, group_scores: dict[int, float]) -> None:
    rows, total = summarize_ranges(group_scores)
    print(title)
    for label, score in rows:
        print(f"{label:<7} {score:>10.4f}")
    print(f"{'Total':<7} {total:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="グループごとの回転最適化でスコアを改善します。")
    parser.add_argument("--input", default="submissions/baseline.csv", help="入力CSV")
    parser.add_argument("--output", default="submissions/rotated.csv", help="出力CSV")
    parser.add_argument("--coarse-step", type=float, default=2.0, help="粗い探索ステップ(度)")
    parser.add_argument("--fine-step", type=float, default=0.2, help="細かい探索ステップ(度)")
    parser.add_argument("--fine-window", type=float, default=2.0, help="細かい探索範囲(度)")
    parser.add_argument("--dry-run", action="store_true", help="出力せずに集計のみ表示")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {input_path}")

    df = pd.read_csv(input_path)
    new_df, before_scores, after_scores = optimize_rotation(df, args.coarse_step, args.fine_step, args.fine_window)

    print_summary("回転前スコア", before_scores)
    print_summary("回転後スコア", after_scores)

    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(output_path, index=False)
        print(f"出力CSV: {output_path}")


if __name__ == "__main__":
    main()
