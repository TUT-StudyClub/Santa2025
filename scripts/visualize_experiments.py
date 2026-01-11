import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

TABLE_COLUMNS = [
    "Date",
    "Author",
    "Branch/PR",
    "Change",
    "Seed",
    "CV",
    "LB",
    "Command/Config",
    "Note",
]


DEFAULT_CATEGORY_RULES = [
    ("特徴量追加", ["特徴量", "feature", "feat"]),
    ("モデル変更", ["モデル", "model", "architecture", "アーキテクチャ"]),
    ("ハイパラ調整", ["探索範囲", "range", "param", "tuning", "movement", "強化"]),
    ("探索/アルゴリズム", ["search", "greedy", "hill", "reheat", "SA", "anneal", "packing"]),
    ("前処理/後処理", ["shrink", "rotate", "shear", "recenter", "baseline"]),
]


DEFAULT_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#8c564b",
    "#17becf",
    "#7f7f7f",
]


FONT_CANDIDATES = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Yu Gothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "IPAPGothic",
]


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


TREE_X_MIN = float(TREE_POINTS[:, 0].min())
TREE_X_MAX = float(TREE_POINTS[:, 0].max())
TREE_Y_MIN = float(TREE_POINTS[:, 1].min())
TREE_Y_MAX = float(TREE_POINTS[:, 1].max())


@dataclass
class ExperimentRow:
    row_index: int
    date: str
    author: str
    branch: str
    change: str
    seed: str
    cv: str
    lb: str
    command: str
    note: str
    score: float | None
    score_source: str
    exp_name: str
    category: str
    matched_submissions: list[str]


def load_category_rules(path: Path | None) -> tuple[list[tuple[str, list[str]]], str]:
    if path is None:
        return DEFAULT_CATEGORY_RULES, "その他"
    data = yaml.safe_load(path.read_text()) or {}
    rules = []
    default_category = data.get("default", "その他")
    for key, keywords in data.items():
        if key == "default":
            continue
        rules.append((key, list(keywords)))
    return (rules or DEFAULT_CATEGORY_RULES), default_category


def setup_fonts(preferred_font: str | None) -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = []
    if preferred_font:
        candidates.append(preferred_font)
    candidates.extend(FONT_CANDIDATES)
    selected = [name for name in candidates if name in available]
    if not selected:
        return None
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    return selected[0]


def extract_score(text: str) -> float | None:
    if not text or text.strip() == "-":
        return None
    match = re.search(r"score\s*=\s*([0-9.]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"([0-9]+\.[0-9]+)", text)
    return float(match.group(1)) if match else None


def extract_exp_name(*fields: str) -> str:
    for field in fields:
        match = re.search(r"(exp\d{3,4}_[A-Za-z0-9_]+)", field)
        if match:
            return match.group(1)
    return "unknown"


def classify_category(text: str, rules: list[tuple[str, list[str]]], default_category: str) -> str:
    lowered = text.lower()
    for category, keywords in rules:
        for keyword in keywords:
            if keyword.lower() in lowered:
                return category
    return default_category


def find_matched_submissions(text: str, submission_names: list[str]) -> list[str]:
    matches = []
    for name in submission_names:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(name)
    return matches


def parse_experiments_markdown(
    path: Path,
    category_rules: list[tuple[str, list[str]]],
    default_category: str,
    submissions_dir: Path,
) -> list[ExperimentRow]:
    lines = path.read_text().splitlines()
    header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| Date |"):
            header_index = i
            break
    if header_index is None:
        raise ValueError("experiments.md のテーブルが見つかりません。")

    submission_names = cast(list[str], sorted({p.name for p in submissions_dir.rglob("*.csv")}, key=len, reverse=True))

    rows = []
    row_index = 0
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < len(TABLE_COLUMNS):
            continue
        data = dict(zip(TABLE_COLUMNS, cells, strict=False))
        lb_score = extract_score(data["LB"])
        cv_score = extract_score(data["CV"])
        score = lb_score if lb_score is not None else cv_score
        score_source = "LB" if lb_score is not None else ("CV" if cv_score is not None else "NA")
        exp_name = extract_exp_name(data["Change"], data["Command/Config"])
        text_blob = " ".join([data["Change"], data["Note"], data["Command/Config"]])
        category = classify_category(text_blob, category_rules, default_category)
        matched = find_matched_submissions(text_blob, submission_names)
        rows.append(
            ExperimentRow(
                row_index=row_index,
                date=data["Date"],
                author=data["Author"],
                branch=data["Branch/PR"],
                change=data["Change"],
                seed=data["Seed"],
                cv=data["CV"],
                lb=data["LB"],
                command=data["Command/Config"],
                note=data["Note"],
                score=score,
                score_source=score_source,
                exp_name=exp_name,
                category=category,
                matched_submissions=matched,
            )
        )
        row_index += 1
    return rows


def to_dataframe(rows: list[ExperimentRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "row_index": row.row_index,
                "date": row.date,
                "author": row.author,
                "branch": row.branch,
                "change": row.change,
                "seed": row.seed,
                "cv": row.cv,
                "lb": row.lb,
                "command": row.command,
                "note": row.note,
                "score": row.score,
                "score_source": row.score_source,
                "exp_name": row.exp_name,
                "category": row.category,
                "matched_submissions": ";".join(row.matched_submissions),
            }
        )
    return pd.DataFrame.from_records(records)


def add_best_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["score"].notna()].reset_index(drop=True)
    df["sequence"] = np.arange(1, len(df) + 1)
    best_scores = []
    best = math.inf
    for score in df["score"]:
        if score < best:
            best = score
            best_scores.append(True)
        else:
            best_scores.append(False)
    df["is_new_best"] = best_scores
    df["best_so_far"] = df["score"].cummin()
    return df


def plot_scores(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("スコアが見つからないため、スコア可視化はスキップします。")
        return

    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    categories = df["category"].unique().tolist()
    palette = {cat: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i, cat in enumerate(categories)}

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)

    # 折れ線（時系列）
    ax = axes[0]
    ax.plot(df["date_dt"], df["score"], color="#444444", linewidth=1.6, zorder=1)
    for category in categories:
        subset = df[df["category"] == category]
        ax.scatter(
            subset["date_dt"],
            subset["score"],
            s=60,
            color=palette[category],
            label=category,
            alpha=0.9,
            zorder=2,
        )
    best_rows = df[df["is_new_best"]]
    ax.scatter(
        best_rows["date_dt"],
        best_rows["score"],
        s=150,
        marker="*",
        color="#f4b400",
        edgecolor="#333333",
        linewidth=0.6,
        label="ベスト更新",
        zorder=3,
    )
    ax.set_title("スコア推移（時系列）")
    ax.set_xlabel("日付")
    ax.set_ylabel("スコア（低いほど良い）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # 散布図（試行回数 × スコア）
    ax = axes[1]
    for category in categories:
        subset = df[df["category"] == category]
        matched = subset[subset["matched_submissions"] != ""]
        unmatched = subset[subset["matched_submissions"] == ""]
        ax.scatter(
            unmatched["sequence"],
            unmatched["score"],
            s=60,
            color=palette[category],
            alpha=0.8,
        )
        ax.scatter(
            matched["sequence"],
            matched["score"],
            s=70,
            color=palette[category],
            edgecolor="#111111",
            linewidth=0.8,
            marker="D",
        )
    ax.scatter(
        best_rows["sequence"],
        best_rows["score"],
        s=150,
        marker="*",
        color="#f4b400",
        edgecolor="#333333",
        linewidth=0.6,
        zorder=3,
    )
    ax.set_title("試行回数 × スコア")
    ax.set_xlabel("試行回数")
    ax.set_ylabel("スコア（低いほど良い）")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "提出ファイル一致は◆マーカー",
        xy=(0.99, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color="#333333",
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"スコア可視化を出力しました: {output_path}")


def parse_submission_values(series: pd.Series) -> np.ndarray:
    values = []
    for value in series:
        v = value
        if isinstance(v, str):
            v = v[1:] if v.startswith("s") else v
        values.append(float(v))
    return np.array(values, dtype=np.float64)


def load_submission_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def calculate_total_score_from_df(df: pd.DataFrame) -> float:
    total = 0.0
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group_df = df[df["id"].str.startswith(prefix)].sort_values("id")
        if group_df.empty:
            continue
        xs = parse_submission_values(group_df["x"])
        ys = parse_submission_values(group_df["y"])
        degs = parse_submission_values(group_df["deg"])
        polygons = [build_tree_vertices(x, y, deg) for x, y, deg in zip(xs, ys, degs, strict=False)]
        side, _ = compute_side_and_bounds(polygons)
        total += side * side / len(polygons)
    return total


def build_tree_vertices(cx: float, cy: float, angle_deg: float) -> np.ndarray:
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return TREE_POINTS @ rot.T + np.array([cx, cy])


def compute_side_and_bounds(
    polygons: list[np.ndarray],
) -> tuple[float, tuple[float, float, float, float]]:
    min_x, min_y, max_x, max_y = compute_bounds_from_vertices(polygons)
    width = max_x - min_x
    height = max_y - min_y
    side = max(width, height)
    return side, (min_x, min_y, max_x, max_y)


def compute_bounds_from_vertices(polygons: list[np.ndarray]) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for verts in polygons:
        min_x = min(min_x, float(verts[:, 0].min()))
        max_x = max(max_x, float(verts[:, 0].max()))
        min_y = min(min_y, float(verts[:, 1].min()))
        max_y = max(max_y, float(verts[:, 1].max()))
    return min_x, min_y, max_x, max_y


def compute_bounds_from_center_ranges(
    x_low: float, x_high: float, y_low: float, y_high: float
) -> tuple[float, float, float, float]:
    min_x = x_low + TREE_X_MIN
    max_x = x_high + TREE_X_MAX
    min_y = y_low + TREE_Y_MIN
    max_y = y_high + TREE_Y_MAX
    dx = max_x - min_x
    dy = max_y - min_y
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0
    margin = 0.02
    return (
        min_x - dx * margin,
        min_y - dy * margin,
        max_x + dx * margin,
        max_y + dy * margin,
    )


def plot_group_layout(  # noqa: PLR0913
    submission_path: Path,
    output_path: Path,
    group_id: int,
    show_labels: bool,
    overlap_eps: float,
    center_layout: bool,
    verbose: bool,
    total_score: float | None,
    score_mode: str,
) -> None:
    if not submission_path.exists():
        print(f"提出ファイルが見つからないため、グループ可視化はスキップします: {submission_path}")
        return

    df = load_submission_df(submission_path)
    plot_group_layout_from_df(
        df,
        output_path,
        group_id,
        show_labels,
        overlap_eps,
        center_layout,
        verbose,
        total_score,
        score_mode,
    )


def plot_group_layout_from_df(  # noqa: PLR0913, PLR0912, PLR0915
    df: pd.DataFrame,
    output_path: Path,
    group_id: int,
    show_labels: bool,
    overlap_eps: float,
    center_layout: bool,
    verbose: bool,
    total_score: float | None,
    score_mode: str,
) -> None:
    prefix = f"{group_id:03d}_"
    group_df = df[df["id"].str.startswith(prefix)].sort_values("id")
    if group_df.empty:
        if verbose:
            print(f"グループ {group_id:03d} が見つかりませんでした。")
        return

    xs = parse_submission_values(group_df["x"])
    ys = parse_submission_values(group_df["y"])
    degs = parse_submission_values(group_df["deg"])
    polygons = [build_tree_vertices(x, y, deg) for x, y, deg in zip(xs, ys, degs, strict=False)]

    overlaps = set()
    overlap_pairs = []
    try:
        from shapely.geometry import Polygon  # noqa: PLC0415
    except ImportError:
        Polygon = None  # type: ignore  # noqa: N806
        print("shapely が見つからないため、重なり判定をスキップします。")

    if Polygon is not None:
        poly_objs = [Polygon(verts) for verts in polygons]
        for i in range(len(poly_objs)):
            for j in range(i + 1, len(poly_objs)):
                inter = poly_objs[i].intersection(poly_objs[j])
                if inter.area > overlap_eps:
                    overlaps.update([i, j])
                    overlap_pairs.append((i, j))

    side, bounds = compute_side_and_bounds(polygons)
    group_score = side * side / len(polygons)

    facecolors = []
    for idx in range(len(polygons)):
        if idx in overlaps:
            facecolors.append("#f4a261")
        else:
            facecolors.append("#dbeeff")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    collection = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors="black",
        linewidths=1.0,
        alpha=0.6,
    )
    ax.add_collection(collection)

    min_x, min_y, max_x, max_y = bounds
    if center_layout:
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        polygons = [verts - np.array([center_x, center_y]) for verts in polygons]
        xs = xs - center_x
        ys = ys - center_y
        min_x -= center_x
        max_x -= center_x
        min_y -= center_y
        max_y -= center_y
    width = max_x - min_x
    height = max_y - min_y
    pad_x = (side - width) / 2.0
    pad_y = (side - height) / 2.0
    square_min_x = min_x - pad_x
    square_max_x = max_x + pad_x
    square_min_y = min_y - pad_y
    square_max_y = max_y + pad_y
    pad = side * 0.02

    ax.add_patch(
        Rectangle(
            (square_min_x, square_min_y),
            side,
            side,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
        )
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(square_min_x - pad, square_max_x + pad)
    ax.set_ylim(square_min_y - pad, square_max_y + pad)

    title = f"Group {group_id:03d} placement (overlaps highlighted)"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    info_lines = []
    if score_mode in {"total", "both"}:
        if total_score is None:
            raise ValueError("total_score が必要です。")
        info_lines.append(f"Score: {total_score:.12f}")
    if score_mode in {"group", "both"}:
        label = "Group score" if score_mode == "both" else "Score"
        info_lines.append(f"{label}: {group_score:.12f}")
    info_lines.append(f"Side: {side:.12f}")
    info = "\n".join(info_lines)
    fig.text(0.02, 0.98, info, ha="left", va="top")
    fig.subplots_adjust(top=0.88)

    if show_labels:
        for label, x, y in zip(group_df["id"], xs, ys, strict=False):
            ax.text(x, y, label, fontsize=8, ha="center", va="center", color="black")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    if verbose:
        print(f"グループ可視化を出力しました: {output_path}")
        if overlap_pairs:
            print(f"重なりペア数: {len(overlap_pairs)}")
        else:
            print("重なりは検出されませんでした。")


def plot_tree_axes(
    ax: plt.Axes,
    polygons: list[np.ndarray],
    bounds: tuple[float, float, float, float],
    title: str,
    outlier_centers: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    collection = PolyCollection(
        polygons,
        facecolors="none",
        edgecolors="#2f4f4f",
        linewidths=0.2,
        alpha=0.6,
        rasterized=True,
    )
    ax.add_collection(collection)
    if outlier_centers is not None:
        ax.scatter(
            outlier_centers[0],
            outlier_centers[1],
            s=8,
            color="#e63946",
            alpha=0.8,
            marker="x",
            linewidth=0.6,
        )
    min_x, min_y, max_x, max_y = bounds
    ax.add_patch(
        Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            fill=False,
            edgecolor="#e76f51",
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)


def plot_metrics_by_group(df: pd.DataFrame, output_path: Path) -> None:
    """グループごとの指標（辺の長さ、スコア寄与度）を可視化します。"""
    if df.empty:
        return

    # データの前処理
    group_ids = []
    sides = []
    scores = []

    # 既存の計算結果があるか確認、なければ計算
    # ここでは submission.csv の内容から計算するため、一度全ポリゴンを生成する
    prefix_set = sorted(list({value.split("_")[0] for value in df["id"] if "_" in value}), key=lambda x: int(x))

    for prefix in prefix_set:
        gid = int(prefix)
        group_df = df[df["id"].str.startswith(prefix + "_")]
        xs = parse_submission_values(group_df["x"])
        ys = parse_submission_values(group_df["y"])
        degs = parse_submission_values(group_df["deg"])
        polygons = [build_tree_vertices(x, y, deg) for x, y, deg in zip(xs, ys, degs, strict=False)]

        side, _ = compute_side_and_bounds(polygons)
        score_contribution = (side * side) / gid

        group_ids.append(gid)
        sides.append(side)
        scores.append(score_contribution)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # 辺の長さの推移
    ax = axes[0]
    ax.plot(group_ids, sides, marker="o", markersize=3, linestyle="-", linewidth=1, color="#1f77b4")
    ax.set_title("Group Size (Side Length) vs N")
    ax.set_xlabel("N (Number of Trees)")
    ax.set_ylabel("Side Length")
    ax.grid(True, alpha=0.3)

    # スコア寄与度の推移
    ax = axes[1]
    ax.plot(group_ids, scores, marker="o", markersize=3, linestyle="-", linewidth=1, color="#d62728")
    ax.set_title("Score Contribution (S^2 / N) vs N")
    ax.set_xlabel("N (Number of Trees)")
    ax.set_ylabel("Score Contribution")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"メトリクス可視化を出力しました: {output_path}")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:  # noqa: PLR0915, PLR0912
    parser = argparse.ArgumentParser(description="実験ログと提出ファイルの可視化を行います。")
    parser.add_argument("--experiments-path", default="docs/experiments.md", help="experiments.md のパス")
    parser.add_argument("--submissions-dir", default="submissions", help="提出ファイルのルート")
    parser.add_argument("--output-dir", default="outputs/visualizations", help="出力先ディレクトリ")
    parser.add_argument("--submission-path", default=None, help="ツリー配置図に使う提出ファイル")
    parser.add_argument("--category-config", default=None, help="カテゴリ分類用の YAML")
    parser.add_argument("--font", default=None, help="日本語フォント名（例: Hiragino Sans）")
    parser.add_argument("--skip-scores", action="store_true", help="実験スコア履歴の可視化をスキップ")
    parser.add_argument("--skip-metrics", action="store_true", help="提出ファイルのメトリクス可視化をスキップ")
    parser.add_argument("--group", type=int, default=None, help="特定のグループ番号(1-200)のみ出力")
    parser.add_argument("--group-all", action="store_true", help="全グループの配置図を出力")
    parser.add_argument("--group-labels", action="store_true", help="グループ図にラベルを表示")
    parser.add_argument("--group-overlap-eps", type=float, default=1e-8, help="重なり判定の面積閾値")
    parser.add_argument(
        "--group-score-mode",
        choices=["total", "group", "both"],
        default="group",
        help="グループ図のスコア表示形式",
    )
    parser.add_argument(
        "--group-no-center",
        action="store_true",
        help="グループ図を中心合わせせずに描画",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    # 1. 実験スコア履歴の可視化
    config_path = Path(args.category_config) if args.category_config else None
    category_rules, default_category = load_category_rules(config_path)
    font_name = setup_fonts(args.font)

    if font_name:
        print(f"フォント設定: {font_name}")
    else:
        print("日本語フォントが見つからないため、警告が出る可能性があります。")

    if not args.skip_scores:
        try:
            rows = parse_experiments_markdown(
                Path(args.experiments_path),
                category_rules,
                default_category,
                Path(args.submissions_dir),
            )
            df = to_dataframe(rows)
            df = df.iloc[::-1].reset_index(drop=True)
            df_scores = add_best_flags(df)

            scores_csv = output_dir / "experiment_scores.csv"
            df_scores.to_csv(scores_csv, index=False)

            matches = df_scores[df_scores["matched_submissions"] != ""]
            matches_csv = output_dir / "submission_matches.csv"
            matches.to_csv(matches_csv, index=False)

            plot_scores(df_scores, output_dir / "score_overview.png")
            print(f"実験スコア履歴を出力しました: {scores_csv}")
        except Exception as e:
            print(f"実験スコア履歴の処理中にエラーが発生しました: {e}")

    # 2. 提出ファイルの可視化
    submission_path = None
    if args.submission_path:
        submission_path = Path(args.submission_path)
    else:
        default_submission = Path(args.submissions_dir) / "submission.csv"
        fallback_submission = Path(args.submissions_dir) / "baseline.csv"
        submission_path = default_submission if default_submission.exists() else fallback_submission

    if not submission_path or not submission_path.exists():
        print("可視化対象の提出ファイルが見つかりません。")
        return

    print(f"提出ファイルを読み込んでいます: {submission_path}")
    sub_df = load_submission_df(submission_path)

    # メトリクス可視化（各Nごとのスコアなど）
    if not args.skip_metrics:
        plot_metrics_by_group(sub_df, output_dir / "metrics_by_group.png")

    # グループ配置図の出力
    target_groups = []
    if args.group is not None:
        target_groups = [args.group]
    elif args.group_all:
        group_ids = sorted({int(value.split("_")[0]) for value in sub_df["id"].astype(str) if "_" in value})
        target_groups = group_ids
        print(f"全{len(target_groups)}グループを出力します。時間がかかる場合があります。")
    else:
        # デフォルト動作: N=200 のみ出力
        target_groups = [200]
        print("デフォルトで N=200 の配置図を出力します。")

    groups_output_dir = output_dir / "groups"
    if args.group_all:  # 複数出力時はフォルダを掘る
        ensure_output_dir(groups_output_dir)

    total_score = None
    if args.group_score_mode in {"total", "both"}:
        total_score = calculate_total_score_from_df(sub_df)

    for idx, group_id in enumerate(target_groups):
        if args.group_all:
            group_output = groups_output_dir / f"group_{group_id:03d}.png"
        else:
            # 単一出力時は直下に置く
            group_output = output_dir / f"group_{group_id:03d}.png"

        plot_group_layout_from_df(
            sub_df,
            group_output,
            group_id,
            args.group_labels,
            args.group_overlap_eps,
            not args.group_no_center,
            verbose=True,
            total_score=total_score,
            score_mode=args.group_score_mode,
        )
        if args.group_all and (idx % 20 == 0 or idx == len(target_groups) - 1):
            print(f"進捗: {idx + 1}/{len(target_groups)}")

    print("完了しました。")


if __name__ == "__main__":
    main()
