"""
複数の提出ファイルから、グループごとに最良スコアを選んで結合するスクリプト。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="グループ単位で最良スコアを結合します。")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="入力の提出CSV（複数指定）",
    )
    parser.add_argument(
        "--config",
        default="experiments/exp018_direct_optimize/exp/000.yaml",
        help="tree_shape を含む設定ファイル",
    )
    parser.add_argument(
        "--output",
        default="submissions/merged_best.csv",
        help="出力CSV",
    )
    return parser.parse_args()


def load_tree_shape(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["tree_shape"]


def parse_value(v) -> float:
    if isinstance(v, str) and v.startswith("s"):
        return float(v[1:])
    return float(v)


def load_groups(path: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    df = pd.read_csv(path)
    groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")
        xs = np.array([parse_value(v) for v in group["x"]], dtype=np.float64)
        ys = np.array([parse_value(v) for v in group["y"]], dtype=np.float64)
        degs = np.array([parse_value(v) for v in group["deg"]], dtype=np.float64)
        groups[n] = (xs, ys, degs)
    return groups


def get_tree_vertices(cfg: dict, cx: float, cy: float, angle_deg: float) -> np.ndarray:
    trunk_w = float(cfg["trunk_w"])
    base_w = float(cfg["base_w"])
    mid_w = float(cfg["mid_w"])
    top_w = float(cfg["top_w"])
    tip_y = float(cfg["tip_y"])
    tier_1_y = float(cfg["tier_1_y"])
    tier_2_y = float(cfg["tier_2_y"])
    base_y = float(cfg["base_y"])
    trunk_bottom_y = float(cfg["trunk_bottom_y"])

    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    pts = np.array(
        [
            [0.0, tip_y],
            [top_w / 2.0, tier_1_y],
            [top_w / 4.0, tier_1_y],
            [mid_w / 2.0, tier_2_y],
            [mid_w / 4.0, tier_2_y],
            [base_w / 2.0, base_y],
            [trunk_w / 2.0, base_y],
            [trunk_w / 2.0, trunk_bottom_y],
            [-trunk_w / 2.0, trunk_bottom_y],
            [-trunk_w / 2.0, base_y],
            [-base_w / 2.0, base_y],
            [-mid_w / 4.0, tier_2_y],
            [-mid_w / 2.0, tier_2_y],
            [-top_w / 4.0, tier_1_y],
            [-top_w / 2.0, tier_1_y],
        ],
        dtype=np.float64,
    )

    vertices = np.empty((15, 2), dtype=np.float64)
    for i in range(15):
        x, y = pts[i, 0], pts[i, 1]
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        vertices[i, 0] = rx + cx
        vertices[i, 1] = ry + cy
    return vertices


def compute_score(cfg: dict, xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for i in range(n):
        verts = get_tree_vertices(cfg, xs[i], ys[i], degs[i])
        min_x = min(min_x, float(verts[:, 0].min()))
        max_x = max(max_x, float(verts[:, 0].max()))
        min_y = min(min_y, float(verts[:, 1].min()))
        max_y = max(max_y, float(verts[:, 1].max()))
    side = max(max_x - min_x, max_y - min_y)
    return side * side / n


def save_submission(path: str, groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
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
    pd.DataFrame(rows).to_csv(path, index=False)


def range_sum(scores: dict[int, float], start: int, end: int) -> float:
    return sum(scores[n] for n in range(start, end + 1))


def main() -> None:
    args = parse_args()
    cfg = load_tree_shape(args.config)

    print("入力ファイルを読み込み中...")
    inputs = [Path(p) for p in args.inputs]
    all_groups = [load_groups(str(p)) for p in inputs]

    best_groups: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    best_scores: dict[int, float] = {}
    winner_counts = {p.name: 0 for p in inputs}

    for n in range(1, 201):
        best_idx = -1
        best_score = math.inf
        for i, groups in enumerate(all_groups):
            xs, ys, degs = groups[n]
            score = compute_score(cfg, xs, ys, degs)
            if score < best_score:
                best_score = score
                best_idx = i
        best_groups[n] = all_groups[best_idx][n]
        best_scores[n] = best_score
        winner_counts[inputs[best_idx].name] += 1

    total_score = sum(best_scores.values())
    save_submission(args.output, best_groups)

    print("結合完了")
    print(f"合計スコア: {total_score:.6f}")
    print("採用グループ数:")
    for name, count in winner_counts.items():
        print(f"  {name}: {count}")

    r1 = range_sum(best_scores, 1, 21)
    r2 = range_sum(best_scores, 21, 60)
    r3 = range_sum(best_scores, 61, 100)
    r4 = range_sum(best_scores, 101, 150)
    r5 = range_sum(best_scores, 151, 200)
    print("レンジ別スコア:")
    print(f"  1-21:   {r1:.6f}")
    print(f"  21-60:  {r2:.6f}")
    print(f"  61-100: {r3:.6f}")
    print(f"  101-150:{r4:.6f}")
    print(f"  151-200:{r5:.6f}")

    print(f"出力: {args.output}")


if __name__ == "__main__":
    main()
