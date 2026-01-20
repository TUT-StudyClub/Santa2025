"""
2つの提出CSVを比較して、変更されたグループとスコア差分を表示する。

用途:
- sparroWASM 等で最適化した結果が「どのグループに効いたか」を特定する
- 提出ファイル更新ルール（baseline/submission より良い時だけ更新）の確認材料にする
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ====== Santa2025: 木ポリゴン（評価と同じ座標系: 原点が幹の付け根中央） ======
TRUNK_W = 0.15
BASE_W = 0.7
MID_W = 0.4
TOP_W = 0.25
TIP_Y = 0.8
TIER_1_Y = 0.5
TIER_2_Y = 0.25
BASE_Y = 0.0
TRUNK_BOTTOM_Y = -0.2

TREE_POINTS = np.array(
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


def parse_prefixed(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lstrip("s").astype(float).to_numpy(dtype=np.float64)


def build_tree_vertices(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    ang = math.radians(angle_deg)
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return TREE_POINTS @ rot.T + np.array([cx, cy], dtype=np.float64)


def group_score_from_df(df: pd.DataFrame, n: int) -> tuple[float, float]:
    prefix = f"{n:03d}_"
    g = df[df["id"].str.startswith(prefix)].sort_values("id")
    if len(g) != n:
        raise ValueError(f"group={n}: 行数が不正です（期待={n}, 実際={len(g)}）")
    xs = parse_prefixed(g["x"])
    ys = parse_prefixed(g["y"])
    degs = parse_prefixed(g["deg"])
    pts = np.vstack([build_tree_vertices(x, y, d) for x, y, d in zip(xs, ys, degs, strict=True)])
    min_x = float(pts[:, 0].min())
    max_x = float(pts[:, 0].max())
    min_y = float(pts[:, 1].min())
    max_y = float(pts[:, 1].max())
    side = max(max_x - min_x, max_y - min_y)
    score = side * side / n
    return score, side


def score_all_groups(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(200, dtype=np.float64)
    sides = np.zeros(200, dtype=np.float64)
    for n in range(1, 201):
        score, side = group_score_from_df(df, n)
        scores[n - 1] = score
        sides[n - 1] = side
    return scores, sides


def block_totals(scores: np.ndarray, block_size: int) -> list[float]:
    if len(scores) != 200:
        raise ValueError("scores の長さは 200 を想定しています")
    if block_size <= 0 or 200 % block_size != 0:
        raise ValueError("--block-size は 200 を割り切れる正の値が必要です（例: 20）")
    out: list[float] = []
    for i in range(0, 200, block_size):
        out.append(float(scores[i : i + block_size].sum()))
    return out


def load_submission_csv(path: Path, *, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} が見つかりません: {path}")
    df = pd.read_csv(path)
    need = {"id", "x", "y", "deg"}
    if not need.issubset(df.columns):
        raise ValueError(f"{name}: 必要列がありません: {sorted(need - set(df.columns))}")
    if df["id"].duplicated().any():
        dup = df[df["id"].duplicated()]["id"].iloc[0]
        raise ValueError(f"{name}: id が重複しています: {dup}")
    return df


def compute_row_diffs(
    base_df: pd.DataFrame,
    new_df: pd.DataFrame,
    *,
    tol: float,
) -> tuple[pd.Series, list[int], int, int]:
    merged = base_df.merge(new_df, on="id", how="outer", suffixes=("_base", "_new"), indicator=True)
    if (merged["_merge"] != "both").any():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"id が一致しません: {counts}")

    x_base = parse_prefixed(merged["x_base"])
    y_base = parse_prefixed(merged["y_base"])
    d_base = parse_prefixed(merged["deg_base"])
    x_new = parse_prefixed(merged["x_new"])
    y_new = parse_prefixed(merged["y_new"])
    d_new = parse_prefixed(merged["deg_new"])

    same = (
        np.isclose(x_base, x_new, rtol=0.0, atol=tol)
        & np.isclose(y_base, y_new, rtol=0.0, atol=tol)
        & np.isclose(d_base, d_new, rtol=0.0, atol=tol)
    )
    changed = ~same
    changed_ids = merged.loc[changed, "id"].astype(str)
    changed_groups = sorted(changed_ids.str.slice(0, 3).astype(int).unique().tolist())
    return changed_ids, changed_groups, int(changed.sum()), len(merged)


def print_group_diffs(
    changed_groups: list[int],
    base_scores: np.ndarray,
    new_scores: np.ndarray,
    base_sides: np.ndarray,
    new_sides: np.ndarray,
) -> None:
    if not changed_groups:
        return
    print("\nグループ別差分（score/side）:")
    for n in changed_groups:
        b = float(base_scores[n - 1])
        nn = float(new_scores[n - 1])
        sb = float(base_sides[n - 1])
        sn = float(new_sides[n - 1])
        print(f"  group={n:03d}: {b:.6f} -> {nn:.6f} diff={(nn - b):+.6f} (side {sb:.4f} -> {sn:.4f})")


def print_block_summary(base_scores: np.ndarray, new_scores: np.ndarray, *, block_size: int) -> None:
    base_blocks = block_totals(base_scores, block_size)
    new_blocks = block_totals(new_scores, block_size)
    print(f"\n==== {block_size}グループ刻み集計（base / new / diff）====")
    for i, (b, n) in enumerate(zip(base_blocks, new_blocks, strict=True), start=0):
        start = 1 + block_size * i
        end = start + block_size - 1
        print(f"{start:03d}-{end:03d}: {b:.6f} / {n:.6f} / diff={(n - b):+.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="2つの提出CSVを比較して差分を表示します。")
    parser.add_argument("--base", required=True, help="比較元CSV（例: submissions/archive/submission.csv）")
    parser.add_argument("--new", required=True, help="比較先CSV（例: submissions/submission.csv）")
    parser.add_argument("--tol", type=float, default=1e-12, help="x/y/deg の一致判定許容誤差")
    parser.add_argument("--block-size", type=int, default=20, help="ブロック集計の幅（例: 20）")
    parser.add_argument("--show-ids", action="store_true", help="差分がある id を表示")
    parser.add_argument("--max-ids", type=int, default=50, help="表示する id の最大件数（--show-ids 用）")
    args = parser.parse_args()

    base_path = Path(args.base)
    new_path = Path(args.new)
    base_df = load_submission_csv(base_path, name="base")
    new_df = load_submission_csv(new_path, name="new")
    changed_ids, changed_groups, changed_count, total_rows = compute_row_diffs(base_df, new_df, tol=float(args.tol))

    base_scores, base_sides = score_all_groups(base_df)
    new_scores, new_sides = score_all_groups(new_df)
    total_base = float(base_scores.sum())
    total_new = float(new_scores.sum())

    print(f"base: {base_path}")
    print(f" new: {new_path}")
    print(f"\n合計スコア: base={total_base:.6f} / new={total_new:.6f} / diff={(total_new - total_base):+.6f}")
    print(f"差分行数: {changed_count} / {total_rows}")
    print(f"差分グループ数: {len(changed_groups)} / 200")
    if changed_groups:
        print(f"差分グループ: {', '.join(str(g) for g in changed_groups)}")

    if args.show_ids and changed_count > 0:
        max_ids = max(int(args.max_ids), 1)
        print("\n差分id（先頭のみ）:")
        for id_ in changed_ids.tolist()[:max_ids]:
            print(f"  {id_}")
        if len(changed_ids) > max_ids:
            print(f"  ...（残り {len(changed_ids) - max_ids} 件）")

    print_group_diffs(changed_groups, base_scores, new_scores, base_sides, new_sides)
    print_block_summary(base_scores, new_scores, block_size=int(args.block_size))


if __name__ == "__main__":
    main()
