"""
提出CSVの指定グループだけ、重なり回避のために「中心からのスケール」をかける。

Kaggle側のバリデータが「ぎりぎり接している配置」を Overlap 扱いする場合があるため、
対象グループの (x, y) をグループ重心まわりにわずかに拡大して、クリアランスを作る用途。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_groups(expr: str) -> list[int]:
    expr = expr.strip()
    if not expr:
        raise ValueError("--groups が空です")
    groups: set[int] = set()
    for raw_part in expr.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            if start > end:
                start, end = end, start
            for n in range(start, end + 1):
                groups.add(n)
        else:
            groups.add(int(part))
    return sorted([n for n in groups if 1 <= n <= 200])


def parse_prefixed(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lstrip("s").astype(float).to_numpy(dtype=np.float64)


def format_prefixed(values: np.ndarray) -> list[str]:
    return [f"s{v}" for v in values]


def scale_group_centers(df: pd.DataFrame, group_id: int, delta: float) -> tuple[int, float, float]:
    if delta <= 0:
        raise ValueError("--delta は正の値が必要です")
    prefix = f"{group_id:03d}_"
    group_df = df[df["id"].str.startswith(prefix)].sort_values("id")
    n = len(group_df)
    if n != group_id:
        raise ValueError(f"group={group_id}: 行数が不正です（期待={group_id}, 実際={n}）")

    xs = parse_prefixed(group_df["x"])
    ys = parse_prefixed(group_df["y"])
    cx = float(xs.mean())
    cy = float(ys.mean())

    scale = 1.0 + float(delta)
    xs2 = cx + (xs - cx) * scale
    ys2 = cy + (ys - cy) * scale

    df.loc[group_df.index, "x"] = format_prefixed(xs2)
    df.loc[group_df.index, "y"] = format_prefixed(ys2)
    return n, cx, cy


def main() -> None:
    parser = argparse.ArgumentParser(description="提出CSVの指定グループに微小スケールを適用します。")
    parser.add_argument("--input", required=True, help="入力CSV")
    parser.add_argument("--output", required=True, help="出力CSV")
    parser.add_argument("--groups", required=True, help="対象グループ（例: 8, 8-30, 8,26,27）")
    parser.add_argument(
        "--delta",
        type=float,
        default=2e-5,
        help="グループ重心まわりの拡大率（例: 2e-5）。(1+delta) 倍に拡大",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    groups = parse_groups(args.groups)
    if not groups:
        raise ValueError("--groups の結果が空です（1..200 の範囲で指定してください）")

    for col in ("id", "x", "y", "deg"):
        if col not in df.columns:
            raise ValueError(f"必須列がありません: {col}")

    print(f"入力: {in_path}")
    print(f"出力: {out_path}")
    print(f"delta: {float(args.delta)}")
    print(f"対象: {groups}")

    for g in groups:
        n, cx, cy = scale_group_centers(df, g, float(args.delta))
        print(f"  group={g:03d}: n={n} center=({cx:.6f},{cy:.6f})")

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
