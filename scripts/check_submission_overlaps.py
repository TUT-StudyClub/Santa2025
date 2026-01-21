"""
提出CSVの「重なり/接触（touch）」を Shapely で厳密チェックする。

ローカルの NumPy/Numba 実装では「接しているだけ」を overlap と判定しない場合があり、
Kaggle 側のバリデータで `Overlapping trees in group XXX` になることがあるため、
提出前の検査用途で使う。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# 木の形状パラメータ（repo内の他スクリプトと合わせる）
TRUNK_W = 0.15
BASE_W = 0.7
MID_W = 0.4
TOP_W = 0.25
TIP_Y = 0.8
TIER_1_Y = 0.5
TIER_2_Y = 0.25
BASE_Y = 0.0
TRUNK_BOTTOM_Y = -0.2


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


def get_tree_vertices(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

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

    xs = pts[:, 0] * cos_a - pts[:, 1] * sin_a + cx
    ys = pts[:, 0] * sin_a + pts[:, 1] * cos_a + cy
    return np.stack([xs, ys], axis=1)


@dataclass(frozen=True)
class PairIssue:
    i: int
    j: int
    distance: float
    intersection_area: float


def build_group_polygons(df: pd.DataFrame, group_id: int) -> tuple[list[str], list[Polygon]]:
    prefix = f"{group_id:03d}_"
    group_df = df[df["id"].str.startswith(prefix)].sort_values("id")
    if len(group_df) != group_id:
        raise ValueError(f"group={group_id}: 行数が不正です（期待={group_id}, 実際={len(group_df)}）")

    ids = group_df["id"].astype(str).tolist()
    xs = parse_prefixed(group_df["x"])
    ys = parse_prefixed(group_df["y"])
    degs = parse_prefixed(group_df["deg"])

    polys: list[Polygon] = []
    for x, y, deg in zip(xs, ys, degs, strict=True):
        verts = get_tree_vertices(float(x), float(y), float(deg))
        polys.append(Polygon([(float(px), float(py)) for px, py in verts]))
    return ids, polys


def check_group(polys: list[Polygon], eps: float, area_eps: float) -> tuple[list[PairIssue], list[PairIssue], float]:
    min_dist = float("inf")
    touches: list[PairIssue] = []
    overlaps: list[PairIssue] = []
    n = len(polys)
    for i in range(n):
        pi = polys[i]
        for j in range(i + 1, n):
            pj = polys[j]
            d = float(pi.distance(pj))
            min_dist = min(d, min_dist)
            if d > eps:
                continue
            inter_area = float(pi.intersection(pj).area)
            issue = PairIssue(i=i, j=j, distance=d, intersection_area=inter_area)
            if inter_area > area_eps:
                overlaps.append(issue)
            else:
                touches.append(issue)
    if min_dist == float("inf"):
        min_dist = 0.0
    return overlaps, touches, min_dist


def main() -> None:
    parser = argparse.ArgumentParser(description="提出CSVの重なり/接触を Shapely でチェックします。")
    parser.add_argument("--input", default="submissions/submission.csv", help="提出CSVのパス")
    parser.add_argument("--groups", default="8", help="対象グループ（例: 8, 1-200, 8,26,27）")
    parser.add_argument("--eps", type=float, default=1e-12, help="距離が eps 以下なら接触扱い（Kaggle対策）")
    parser.add_argument("--area-eps", type=float, default=1e-12, help="交差面積が area_eps を超えたら overlap 扱い")
    parser.add_argument("--max-pairs", type=int, default=20, help="詳細表示するペア上限（0で非表示）")
    parser.add_argument("--only-ng", action="store_true", help="問題があるグループ（NG）だけ表示する")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {in_path}")

    df = pd.read_csv(in_path)
    for col in ("id", "x", "y", "deg"):
        if col not in df.columns:
            raise ValueError(f"必須列がありません: {col}")

    groups = parse_groups(str(args.groups))
    if not groups:
        raise ValueError("--groups の結果が空です（1..200 の範囲で指定してください）")

    eps = float(args.eps)
    area_eps = float(args.area_eps)
    max_pairs = int(args.max_pairs)
    only_ng = bool(args.only_ng)

    print(f"入力: {in_path}")
    print(f"対象: {groups}")
    print(f"eps: {eps:g} / area_eps: {area_eps:g}")

    any_issue = False
    for g in groups:
        ids, polys = build_group_polygons(df, g)
        overlaps, touches, min_dist = check_group(polys, eps=eps, area_eps=area_eps)

        if overlaps or touches:
            any_issue = True
            print(f"\n[NG] group={g:03d}: overlap={len(overlaps)} touch={len(touches)} min_dist={min_dist:.3e}")
            if max_pairs > 0:
                shown = 0
                for kind, issues in (("overlap", overlaps), ("touch", touches)):
                    for iss in issues:
                        if shown >= max_pairs:
                            break
                        i_id = ids[iss.i]
                        j_id = ids[iss.j]
                        print(f"  - {kind}: {i_id} x {j_id} dist={iss.distance:.3e} area={iss.intersection_area:.3e}")
                        shown += 1
        elif not only_ng:
            print(f"\n[OK] group={g:03d}: min_dist={min_dist:.3e}")

    if any_issue:
        print(
            "\n対策案: `scripts/scale_submission_groups.py` で対象グループを微小拡大してクリアランスを作ってください。"
        )
        print(
            "例: `uv run python scripts/scale_submission_groups.py --input submissions/submission.csv "
            "--output submissions/submission_valid.csv --groups 8 --delta 1e-6`"
        )


if __name__ == "__main__":
    main()
