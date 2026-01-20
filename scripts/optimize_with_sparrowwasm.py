"""
sparroWASM（sparrow-cli）を使って、提出CSVの一部グループを再配置して差し替える。

- 入力: submissions/*.csv（id,x,y,deg）
- 出力: 新しいCSV（改善したグループのみ差し替え）

注意:
sparrow-cli は「固定高さのストリップで幅を最小化」する問題設定なので、
このスクリプトでは出力レイアウトに対して「全体回転」を追加で探索し、
Kaggle側のスコア（max(width,height)^2 / n）が改善した場合のみ採用する。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def parse_float_list(expr: str) -> list[float]:
    expr = expr.strip()
    if not expr:
        return []
    vals: list[float] = []
    for raw_part in expr.split(","):
        part = raw_part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def polygon_signed_area(points: np.ndarray) -> float:
    poly = np.vstack([points, points[0]])
    return 0.5 * float(np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1]))


def make_ccw_closed(points: np.ndarray) -> np.ndarray:
    pts = points.copy()
    if polygon_signed_area(pts) < 0:
        pts = pts[::-1].copy()
    pts = np.vstack([pts, pts[0]])
    return pts


def rotate_points_xy(points_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    ang = math.radians(angle_deg)
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return np.column_stack((x * cos_a - y * sin_a, x * sin_a + y * cos_a))


def compute_side_from_points(points_xy: np.ndarray) -> float:
    min_x = float(points_xy[:, 0].min())
    max_x = float(points_xy[:, 0].max())
    min_y = float(points_xy[:, 1].min())
    max_y = float(points_xy[:, 1].max())
    return max(max_x - min_x, max_y - min_y)


def build_tree_vertices(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    ang = math.radians(angle_deg)
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return TREE_POINTS @ rot.T + np.array([cx, cy], dtype=np.float64)


def build_group_points(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> np.ndarray:
    points = [build_tree_vertices(x, y, d) for x, y, d in zip(xs, ys, degs, strict=True)]
    return np.vstack(points)


def group_score(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> tuple[float, float]:
    pts = build_group_points(xs, ys, degs)
    side = compute_side_from_points(pts)
    score = side * side / len(xs)
    return score, side


def angle_grid(step: float) -> np.ndarray:
    step = max(step, 1e-6)
    return np.arange(0.0, 180.0 + step / 2.0, step, dtype=np.float64)


def find_best_global_rotation(points_xy: np.ndarray, coarse_step: float, fine_step: float, fine_window: float) -> float:
    best_angle = 0.0
    best_side = math.inf
    for ang in angle_grid(coarse_step):
        side = compute_side_from_points(rotate_points_xy(points_xy, float(ang)))
        if side < best_side:
            best_side = side
            best_angle = float(ang)

    if fine_step > 0.0:
        start = max(0.0, best_angle - fine_window)
        end = min(180.0, best_angle + fine_window)
        for ang in angle_grid(fine_step):
            if ang < start or ang > end:
                continue
            side = compute_side_from_points(rotate_points_xy(points_xy, float(ang)))
            if side < best_side:
                best_side = side
                best_angle = float(ang)
    return best_angle


def rotate_centers(xs: np.ndarray, ys: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ang = math.radians(angle_deg)
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    new_xs = xs * cos_a - ys * sin_a
    new_ys = xs * sin_a + ys * cos_a
    return new_xs, new_ys


def make_allowed_orientations(step_deg: float) -> list[float]:
    step_deg = float(step_deg)
    if step_deg <= 0:
        raise ValueError("--orient-step は正の値が必要です")
    vals = np.arange(0.0, 360.0, step_deg, dtype=np.float64)
    # 例: step=7 の場合に 360 を超えないようにする
    return [float(v) for v in vals]


def run_sparrow_cli(
    sparrow_cli: str,
    input_json: Path,
    output_json: Path,
    timeout_s: int,
    workers: int,
    seed: int | None,
    early_termination: bool,
    verbose: bool,
) -> None:
    cmd = [sparrow_cli, "--input", str(input_json), "--output", str(output_json), "--timeout", str(timeout_s)]
    cmd += ["--workers", str(workers)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if early_termination:
        cmd += ["--early-termination"]
    if verbose:
        cmd += ["--verbose"]

    subprocess.run(cmd, check=True, capture_output=not verbose, text=True)


def optimize_one_group_with_sparrow(
    n: int,
    sparrow_cli: str,
    strip_height: float,
    sparrow_scale: float,
    orient_step: float,
    timeout_s: int,
    workers: int,
    seed: int | None,
    early_termination: bool,
    verbose_cli: bool,
    coarse_rot_step: float,
    fine_rot_step: float,
    fine_rot_window: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    # sparrow-cli 用の形状は「左下が (0,0)」になるように平行移動したものを渡す
    min_x = float(TREE_POINTS[:, 0].min())
    min_y = float(TREE_POINTS[:, 1].min())
    shift = np.array([-min_x, -min_y], dtype=np.float64)

    if sparrow_scale <= 0:
        raise ValueError("--sparrow-scale は正の値が必要です")

    scale = float(sparrow_scale)
    poly = make_ccw_closed((TREE_POINTS + shift) * scale)

    instance = {
        "name": f"group_{n:03d}",
        "items": [
            {
                "id": 0,
                "demand": n,
                "allowed_orientations": make_allowed_orientations(orient_step),
                "shape": {"type": "simple_polygon", "data": poly.tolist()},
            }
        ],
        "strip_height": float(strip_height) * scale,
    }

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_path = td_path / "input.json"
        out_path = td_path / "output.json"
        in_path.write_text(json.dumps(instance), encoding="utf-8")

        run_sparrow_cli(
            sparrow_cli=sparrow_cli,
            input_json=in_path,
            output_json=out_path,
            timeout_s=timeout_s,
            workers=workers,
            seed=seed,
            early_termination=early_termination,
            verbose=verbose_cli,
        )

        result = json.loads(out_path.read_text(encoding="utf-8"))

    layouts = result.get("layouts") or []
    status = result.get("status")
    if status != "complete" or len(layouts) != n:
        raise RuntimeError(f"group={n}: 配置に失敗しました（status={status}, placed={len(layouts)}/{n}）")

    xs_raw = np.array([float(it["position_x"]) for it in layouts], dtype=np.float64) / scale
    ys_raw = np.array([float(it["position_y"]) for it in layouts], dtype=np.float64) / scale
    degs = np.array([float(it["rotation_degrees"]) for it in layouts], dtype=np.float64) % 360.0

    # 形状を shift して渡しているので、評価系の原点（幹の付け根中央）へ戻す
    ang = np.deg2rad(degs)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    shift_x = shift[0] * cos_a - shift[1] * sin_a
    shift_y = shift[0] * sin_a + shift[1] * cos_a
    xs = xs_raw + shift_x
    ys = ys_raw + shift_y

    # 追加で「全体回転」を入れて正方形側長を下げる（Kaggleの目的関数に寄せる）
    pts = build_group_points(xs, ys, degs)
    best_ang = find_best_global_rotation(pts, coarse_rot_step, fine_rot_step, fine_rot_window)
    if abs(best_ang) > 1e-12:
        xs, ys = rotate_centers(xs, ys, best_ang)
        degs = (degs + best_ang) % 360.0
        pts = rotate_points_xy(pts, best_ang)

    side = compute_side_from_points(pts)
    score = side * side / n
    return xs, ys, degs, score, side


def compute_block20_totals(df: pd.DataFrame) -> list[float]:
    totals = []
    for start in range(1, 201, 20):
        end = start + 19
        total = 0.0
        for n in range(start, end + 1):
            prefix = f"{n:03d}_"
            g = df[df["id"].str.startswith(prefix)].sort_values("id")
            xs = parse_prefixed(g["x"])
            ys = parse_prefixed(g["y"])
            degs = parse_prefixed(g["deg"])
            score, _ = group_score(xs, ys, degs)
            total += score
        totals.append(total)
    return totals


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="sparroWASM（sparrow-cli）で提出CSVの一部グループを最適化します。")
    parser.add_argument("--input", default="submissions/submission.csv", help="入力CSV")
    parser.add_argument("--output", required=True, help="出力CSV")
    parser.add_argument("--sparrow-cli", required=True, help="sparrow-cli 実行ファイルのパス")
    parser.add_argument("--groups", default="1-200", help="対象グループ（例: 21-100, 5,7,9）")
    parser.add_argument("--timeout", type=int, default=10, help="1グループあたりの制限時間（秒）")
    parser.add_argument("--workers", type=int, default=4, help="sparrow-cli の worker 数")
    parser.add_argument("--seed", type=int, default=None, help="sparrow-cli の乱数 seed（省略可）")
    parser.add_argument("--early-termination", action="store_true", help="sparrow-cli の早期停止を有効化")
    parser.add_argument("--orient-step", type=float, default=5.0, help="許容回転角の刻み（度）")
    parser.add_argument(
        "--sparrow-scale",
        type=float,
        default=1000.0,
        help="sparrow-cli に渡す座標のスケール倍率（小さすぎると探索が崩れる場合がある）",
    )
    parser.add_argument("--strip-height-scale", type=float, default=1.0, help="strip_height = 現状side * scale")
    parser.add_argument(
        "--strip-height-scales",
        default="",
        help="strip_height の scale を複数試す（例: 0.9,0.95,1.0）。未指定なら --strip-height-scale のみ。",
    )
    parser.add_argument("--trials", type=int, default=1, help="各 scale で何回試すか（seed を変えて複数回）")
    parser.add_argument("--accept-eps", type=float, default=1e-12, help="改善判定の許容値（小さいほど厳密）")

    parser.add_argument("--rot-coarse-step", type=float, default=2.0, help="全体回転探索（粗）ステップ")
    parser.add_argument("--rot-fine-step", type=float, default=0.2, help="全体回転探索（細）ステップ")
    parser.add_argument("--rot-fine-window", type=float, default=2.0, help="全体回転探索（細）範囲")

    parser.add_argument("--verbose-cli", action="store_true", help="sparrow-cli のログをそのまま表示")
    parser.add_argument("--print-best", action="store_true", help="改善しなかった場合もベスト試行を表示")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {input_path}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sparrow_cli = str(Path(args.sparrow_cli))
    if not Path(sparrow_cli).exists():
        raise FileNotFoundError(f"sparrow-cli が見つかりません: {sparrow_cli}")

    df = pd.read_csv(input_path)
    groups = parse_groups(args.groups)
    if not groups:
        raise ValueError("--groups の結果が空です（1..200 の範囲で指定してください）")

    print(f"入力: {input_path}")
    print(f"出力: {out_path}")
    print(f"sparrow-cli: {sparrow_cli}")
    print(f"対象グループ数: {len(groups)}（例: {groups[:5]}...）")
    print(
        "パラメータ: "
        f"timeout={args.timeout}s workers={args.workers} orient_step={args.orient_step} "
        f"sparrow_scale={args.sparrow_scale} strip_height_scale={args.strip_height_scale}"
    )

    improved = 0
    failed = 0

    scales = parse_float_list(args.strip_height_scales)
    if not scales:
        scales = [float(args.strip_height_scale)]

    if args.trials <= 0:
        raise ValueError("--trials は 1 以上を指定してください")

    for n in tqdm(groups, desc="最適化", unit="group"):
        prefix = f"{n:03d}_"
        group_df = df[df["id"].str.startswith(prefix)].sort_values("id")
        if len(group_df) != n:
            print(f"[WARN] group={n}: 行数が不正です（期待={n}, 実際={len(group_df)}） -> スキップ")
            continue

        xs0 = parse_prefixed(group_df["x"])
        ys0 = parse_prefixed(group_df["y"])
        degs0 = parse_prefixed(group_df["deg"])
        base_score, base_side = group_score(xs0, ys0, degs0)

        best = None
        for scale_idx, scale in enumerate(scales):
            strip_h = base_side * float(scale)
            for trial in range(int(args.trials)):
                trial_seed = None
                if args.seed is not None:
                    trial_seed = int(args.seed) + scale_idx * 1000 + trial

                try:
                    xs1, ys1, degs1, new_score, new_side = optimize_one_group_with_sparrow(
                        n=n,
                        sparrow_cli=sparrow_cli,
                        strip_height=strip_h,
                        sparrow_scale=args.sparrow_scale,
                        orient_step=args.orient_step,
                        timeout_s=args.timeout,
                        workers=args.workers,
                        seed=trial_seed,
                        early_termination=args.early_termination,
                        verbose_cli=args.verbose_cli,
                        coarse_rot_step=args.rot_coarse_step,
                        fine_rot_step=args.rot_fine_step,
                        fine_rot_window=args.rot_fine_window,
                    )
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    if args.print_best:
                        print(f"[FAIL] group={n} scale={scale} trial={trial}: {e}")
                    continue

                cand = (new_score, new_side, scale, trial_seed, xs1, ys1, degs1)
                if best is None or cand[0] < best[0]:
                    best = cand

        if best is None:
            continue

        new_score, new_side, best_scale, best_seed, xs1, ys1, degs1 = best

        if new_score + float(args.accept_eps) < base_score:
            improved += 1
            df.loc[group_df.index, "x"] = format_prefixed(xs1)
            df.loc[group_df.index, "y"] = format_prefixed(ys1)
            df.loc[group_df.index, "deg"] = format_prefixed(degs1)
            print(
                f"[OK] group={n}: score {base_score:.6f} -> {new_score:.6f} "
                f"(side {base_side:.4f} -> {new_side:.4f}) scale={best_scale} seed={best_seed}"
            )
        elif args.print_best:
            print(
                f"[NG] group={n}: best {new_score:.6f} (side {new_side:.4f}) / base {base_score:.6f} "
                f"scale={best_scale} seed={best_seed}"
            )

    df.to_csv(out_path, index=False)

    totals = compute_block20_totals(df)
    total_score = float(sum(totals))
    print("\n==== 20グループ刻み集計 ====")
    for i, val in enumerate(totals, start=0):
        start = 1 + 20 * i
        end = start + 19
        print(f"{start:03d}-{end:03d}: {val:.6f}")
    print(f"合計: {total_score:.6f}")
    print(f"\n改善: {improved} / {len(groups)}（失敗: {failed}）")


if __name__ == "__main__":
    main()
