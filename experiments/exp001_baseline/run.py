from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from tqdm import tqdm
from utils.env import EnvConfig
from utils.logger import get_logger
from utils.timing import trace

load_dotenv()


@dataclass
class ExpConfig:
    debug: bool = False
    debug_max_trees: int = 30
    seed: int = 42
    max_trees: int = 200
    attempts_per_tree: int = 10
    start_radius: float = 20.0
    step_in: float = 0.5
    step_out: float = 0.05
    angle_weighted: bool = True
    plot_every: int = 0
    decimal_precision: int = 25
    scale_factor: str = "1e15"
    xy_limit: float = 100.0
    compact_iters: int = 0
    compact_every: int = 1
    compact_step_in: float = 0.2
    reinsert_iters: int = 0
    reinsert_attempts: int = 10
    reinsert_every: int = 1
    reinsert_rotate: bool = True
    recenter_every: int = 0
    anneal_iters: int = 0
    anneal_every: int = 1
    anneal_move: float = 0.2
    anneal_rotate_deg: float = 5.0
    anneal_start_temp: float = 0.05
    anneal_end_temp: float = 0.005
    beam_width: int = 1
    beam_expansions: int = 1
    beam_start_n: int = 1
    progress: bool = True
    note: str = ""


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)


@dataclass
class ChristmasTree:
    center_x: Decimal
    center_y: Decimal
    angle: Decimal
    polygon: Polygon


cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def init_output_dir(cfg: Config) -> Path:
    major = Path(__file__).resolve().parent.name
    minor = HydraConfig.get().runtime.choices.exp  # type: ignore
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = repo_root() / "outputs" / "runs" / major / minor / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.env.output_dir = output_dir
    cfg.env.exp_name = f"{major}/{minor}"
    return output_dir


def dump_config(cfg: Config, output_dir: Path) -> None:
    (output_dir / "config_exp.yaml").write_text(
        OmegaConf.to_yaml(cfg.exp, resolve=True),
        encoding="utf-8",
    )
    (output_dir / "config_env.yaml").write_text(
        OmegaConf.to_yaml(cfg.env, resolve=True),
        encoding="utf-8",
    )


def build_submission_index(max_trees: int) -> list[str]:
    return [f"{n:03d}_{t}" for n in range(1, max_trees + 1) for t in range(n)]


def build_tree_base_polygon(scale_factor: Decimal) -> Polygon:
    trunk_w = Decimal("0.15")
    trunk_h = Decimal("0.2")
    base_w = Decimal("0.7")
    mid_w = Decimal("0.4")
    top_w = Decimal("0.25")
    tip_y = Decimal("0.8")
    tier_1_y = Decimal("0.5")
    tier_2_y = Decimal("0.25")
    base_y = Decimal("0.0")
    trunk_bottom_y = -trunk_h

    return Polygon(
        [
            (Decimal("0.0") * scale_factor, tip_y * scale_factor),
            (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
        ]
    )


def clone_trees(trees: list[ChristmasTree]) -> list[ChristmasTree]:
    return [
        ChristmasTree(
            center_x=tree.center_x,
            center_y=tree.center_y,
            angle=tree.angle,
            polygon=tree.polygon,
        )
        for tree in trees
    ]


def compute_bounds_from_polygons(polygons: list[Polygon]) -> tuple[float, float, float, float]:
    if not polygons:
        return 0.0, 0.0, 0.0, 0.0
    minx, miny, maxx, maxy = polygons[0].bounds
    for poly in polygons[1:]:
        b_minx, b_miny, b_maxx, b_maxy = poly.bounds
        minx = min(minx, b_minx)
        miny = min(miny, b_miny)
        maxx = max(maxx, b_maxx)
        maxy = max(maxy, b_maxy)
    return minx, miny, maxx, maxy


def compute_bounds(placed_trees: list[ChristmasTree]) -> tuple[float, float, float, float]:
    return compute_bounds_from_polygons([t.polygon for t in placed_trees])


def merge_bounds(
    base_bounds: tuple[float, float, float, float] | None,
    add_bounds: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if base_bounds is None:
        return add_bounds if add_bounds is not None else (0.0, 0.0, 0.0, 0.0)
    if add_bounds is None:
        return base_bounds
    minx = min(base_bounds[0], add_bounds[0])
    miny = min(base_bounds[1], add_bounds[1])
    maxx = max(base_bounds[2], add_bounds[2])
    maxy = max(base_bounds[3], add_bounds[3])
    return minx, miny, maxx, maxy


def compute_bounds_excluding(
    polygons: list[Polygon],
    skip_idx: int,
) -> tuple[float, float, float, float] | None:
    bounds = None
    for idx, poly in enumerate(polygons):
        if idx == skip_idx:
            continue
        bounds = merge_bounds(bounds, poly.bounds)
    return bounds


def side_length_from_bounds(
    bounds: tuple[float, float, float, float],
    scale_factor: Decimal,
) -> Decimal:
    width = Decimal(bounds[2] - bounds[0]) / scale_factor
    height = Decimal(bounds[3] - bounds[1]) / scale_factor
    return max(width, height)


def compute_side_length(placed_trees: list[ChristmasTree], scale_factor: Decimal) -> Decimal:
    return side_length_from_bounds(compute_bounds(placed_trees), scale_factor)


def compute_bounds_center(
    placed_trees: list[ChristmasTree],
    scale_factor: Decimal,
) -> tuple[Decimal, Decimal]:
    minx, miny, maxx, maxy = compute_bounds(placed_trees)
    center_x = (Decimal(minx) + Decimal(maxx)) / scale_factor / Decimal("2")
    center_y = (Decimal(miny) + Decimal(maxy)) / scale_factor / Decimal("2")
    return center_x, center_y


def collides(candidate: Polygon, polygons: list[Polygon], skip_idx: int) -> bool:
    for idx, poly in enumerate(polygons):
        if idx == skip_idx:
            continue
        if candidate.intersects(poly) and not candidate.touches(poly):
            return True
    return False


def collides_with_any(
    candidate: Polygon,
    polygons: list[Polygon],
    tree_index: STRtree | None,
) -> bool:
    if not polygons:
        return False
    if tree_index is None:
        indices = range(len(polygons))
    else:
        indices = tree_index.query(candidate)
    for idx in indices:
        if candidate.intersects(polygons[idx]) and not candidate.touches(polygons[idx]):
            return True
    return False


def find_position_along_vector(  # noqa: PLR0913
    polygon: Polygon,
    *,
    direction_angle: float,
    scale_factor: Decimal,
    start_radius: Decimal,
    step_in: Decimal,
    step_out: Decimal,
    placed_polygons: list[Polygon],
    tree_index: STRtree | None,
) -> tuple[Decimal, Decimal, Polygon]:
    vx = Decimal(str(math.cos(direction_angle)))
    vy = Decimal(str(math.sin(direction_angle)))
    radius = start_radius
    collision_found = False

    while radius >= 0:
        px = radius * vx
        py = radius * vy

        candidate = affinity.translate(
            polygon,
            xoff=float(px * scale_factor),
            yoff=float(py * scale_factor),
        )
        if collides_with_any(candidate, placed_polygons, tree_index):
            collision_found = True
            break
        radius -= step_in

    if collision_found:
        while True:
            radius += step_out
            px = radius * vx
            py = radius * vy
            candidate = affinity.translate(
                polygon,
                xoff=float(px * scale_factor),
                yoff=float(py * scale_factor),
            )
            if not collides_with_any(candidate, placed_polygons, tree_index):
                break
    else:
        radius = Decimal("0")
        px = Decimal("0")
        py = Decimal("0")
        candidate = affinity.translate(
            polygon,
            xoff=float(px * scale_factor),
            yoff=float(py * scale_factor),
        )

    return px, py, candidate


def generate_weighted_angle(rng: random.Random) -> float:
    while True:
        angle = rng.uniform(0, 2 * math.pi)
        if rng.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def initialize_trees(  # noqa: PLR0913
    num_trees: int,
    *,
    rng: random.Random,
    base_polygon: Polygon,
    scale_factor: Decimal,
    cfg: ExpConfig,
    existing_trees: list[ChristmasTree] | None = None,
) -> tuple[list[ChristmasTree], Decimal]:
    if num_trees == 0:
        return [], Decimal("0")

    placed_trees = list(existing_trees) if existing_trees else []
    num_to_add = num_trees - len(placed_trees)

    start_radius = Decimal(str(cfg.start_radius))
    step_in = Decimal(str(cfg.step_in))
    step_out = Decimal(str(cfg.step_out))

    placed_polygons = [t.polygon for t in placed_trees]
    current_bounds = compute_bounds_from_polygons(placed_polygons) if placed_polygons else None

    if num_to_add > 0:
        for _ in range(num_to_add):
            tree_index = STRtree(placed_polygons) if placed_polygons else None

            best_candidate: tuple[Decimal, Decimal, float, Polygon] | None = None
            best_bounds: tuple[float, float, float, float] | None = None
            best_side: Decimal | None = None

            for _ in range(max(cfg.attempts_per_tree, 1)):
                angle_deg = rng.uniform(0, 360)
                rotated = affinity.rotate(base_polygon, float(angle_deg), origin=(0, 0))
                direction_angle = generate_weighted_angle(rng) if cfg.angle_weighted else rng.uniform(0, 2 * math.pi)

                px, py, candidate_poly = find_position_along_vector(
                    rotated,
                    direction_angle=direction_angle,
                    scale_factor=scale_factor,
                    start_radius=start_radius,
                    step_in=step_in,
                    step_out=step_out,
                    placed_polygons=placed_polygons,
                    tree_index=tree_index,
                )

                merged_bounds = merge_bounds(current_bounds, candidate_poly.bounds)
                side_length = side_length_from_bounds(merged_bounds, scale_factor)

                if best_side is None or side_length < best_side:
                    best_side = side_length
                    best_bounds = merged_bounds
                    best_candidate = (px, py, angle_deg, candidate_poly)

            if best_candidate is None:
                angle_deg = rng.uniform(0, 360)
                rotated = affinity.rotate(base_polygon, float(angle_deg), origin=(0, 0))
                px = Decimal("0")
                py = Decimal("0")
                candidate_poly = affinity.translate(
                    rotated,
                    xoff=float(px * scale_factor),
                    yoff=float(py * scale_factor),
                )
                best_bounds = merge_bounds(current_bounds, candidate_poly.bounds)
                best_candidate = (px, py, angle_deg, candidate_poly)

            px, py, angle_deg, candidate_poly = best_candidate
            placed_trees.append(
                ChristmasTree(
                    center_x=px,
                    center_y=py,
                    angle=Decimal(str(angle_deg)),
                    polygon=candidate_poly,
                )
            )
            placed_polygons.append(candidate_poly)
            current_bounds = best_bounds

    side_length = compute_side_length(placed_trees, scale_factor)

    return placed_trees, side_length


def compact_trees(
    placed_trees: list[ChristmasTree],
    *,
    rng: random.Random,
    scale_factor: Decimal,
    cfg: ExpConfig,
) -> None:
    if cfg.compact_iters <= 0 or len(placed_trees) < 2:  # noqa: PLR2004
        return

    step_in = Decimal(str(cfg.compact_step_in))
    polygons = [t.polygon for t in placed_trees]

    for _ in range(cfg.compact_iters):
        center_x, center_y = compute_bounds_center(placed_trees, scale_factor)
        order = list(range(len(placed_trees)))
        rng.shuffle(order)

        for idx in order:
            tree = placed_trees[idx]
            dx = center_x - tree.center_x
            dy = center_y - tree.center_y
            dist = math.hypot(float(dx), float(dy))
            if dist < 1e-9:  # noqa: PLR2004
                continue

            vx = Decimal(str(float(dx) / dist))
            vy = Decimal(str(float(dy) / dist))
            max_radius = Decimal(str(dist))

            best_radius = Decimal("0")
            radius = step_in
            while radius <= max_radius:
                cand_x = tree.center_x + vx * radius
                cand_y = tree.center_y + vy * radius
                candidate_poly = affinity.translate(
                    tree.polygon,
                    xoff=float((cand_x - tree.center_x) * scale_factor),
                    yoff=float((cand_y - tree.center_y) * scale_factor),
                )

                if collides(candidate_poly, polygons, idx):
                    break
                best_radius = radius
                radius += step_in

            if best_radius > 0:
                cand_x = tree.center_x + vx * best_radius
                cand_y = tree.center_y + vy * best_radius
                candidate_poly = affinity.translate(
                    tree.polygon,
                    xoff=float((cand_x - tree.center_x) * scale_factor),
                    yoff=float((cand_y - tree.center_y) * scale_factor),
                )
                tree.center_x = cand_x
                tree.center_y = cand_y
                tree.polygon = candidate_poly
                polygons[idx] = candidate_poly


def reinsert_trees(
    placed_trees: list[ChristmasTree],
    *,
    rng: random.Random,
    base_polygon: Polygon,
    scale_factor: Decimal,
    cfg: ExpConfig,
) -> None:
    if cfg.reinsert_iters <= 0 or len(placed_trees) < 2:  # noqa: PLR2004
        return

    start_radius = Decimal(str(cfg.start_radius))
    step_in = Decimal(str(cfg.step_in))
    step_out = Decimal(str(cfg.step_out))
    attempts = max(cfg.reinsert_attempts, 1)

    for _ in range(cfg.reinsert_iters):
        order = list(range(len(placed_trees)))
        rng.shuffle(order)

        for idx in order:
            other_polygons = [tree.polygon for i, tree in enumerate(placed_trees) if i != idx]
            tree_index = STRtree(other_polygons) if other_polygons else None
            current_bounds = compute_bounds_from_polygons(other_polygons) if other_polygons else None

            baseline_bounds = merge_bounds(current_bounds, placed_trees[idx].polygon.bounds)
            best_bounds = baseline_bounds
            best_side = side_length_from_bounds(best_bounds, scale_factor)
            best_candidate: tuple[Decimal, Decimal, float, Polygon] | None = None

            for _ in range(attempts):
                if cfg.reinsert_rotate:
                    angle_deg = rng.uniform(0, 360)
                else:
                    angle_deg = float(placed_trees[idx].angle)

                rotated = affinity.rotate(base_polygon, float(angle_deg), origin=(0, 0))
                direction_angle = generate_weighted_angle(rng) if cfg.angle_weighted else rng.uniform(0, 2 * math.pi)
                px, py, candidate_poly = find_position_along_vector(
                    rotated,
                    direction_angle=direction_angle,
                    scale_factor=scale_factor,
                    start_radius=start_radius,
                    step_in=step_in,
                    step_out=step_out,
                    placed_polygons=other_polygons,
                    tree_index=tree_index,
                )

                merged_bounds = merge_bounds(current_bounds, candidate_poly.bounds)
                side_length = side_length_from_bounds(merged_bounds, scale_factor)

                if side_length < best_side:
                    best_side = side_length
                    best_bounds = merged_bounds
                    best_candidate = (px, py, angle_deg, candidate_poly)

            if best_candidate is not None:
                px, py, angle_deg, candidate_poly = best_candidate
                placed_trees[idx].center_x = px
                placed_trees[idx].center_y = py
                placed_trees[idx].angle = Decimal(str(angle_deg))
                placed_trees[idx].polygon = candidate_poly


def anneal_trees(
    placed_trees: list[ChristmasTree],
    *,
    rng: random.Random,
    base_polygon: Polygon,
    scale_factor: Decimal,
    cfg: ExpConfig,
) -> None:
    if cfg.anneal_iters <= 0 or len(placed_trees) < 2:  # noqa: PLR2004
        return

    polygons = [tree.polygon for tree in placed_trees]
    xy_limit = Decimal(str(cfg.xy_limit))
    total_iters = max(cfg.anneal_iters, 1)
    temp_start = cfg.anneal_start_temp
    temp_end = cfg.anneal_end_temp

    for step in range(total_iters):
        idx = rng.randrange(len(placed_trees))
        tree = placed_trees[idx]

        t_ratio = step / max(total_iters - 1, 1)
        temp = temp_start + (temp_end - temp_start) * t_ratio
        move_scale = cfg.anneal_move * max((temp / temp_start) if temp_start > 0 else 1.0, 0.2)

        dx = Decimal(str(rng.uniform(-move_scale, move_scale)))
        dy = Decimal(str(rng.uniform(-move_scale, move_scale)))
        cand_x = tree.center_x + dx
        cand_y = tree.center_y + dy

        if abs(cand_x) > xy_limit or abs(cand_y) > xy_limit:
            continue

        angle_deg = float(tree.angle)
        if cfg.anneal_rotate_deg > 0:
            angle_deg += rng.uniform(-cfg.anneal_rotate_deg, cfg.anneal_rotate_deg)

        rotated = affinity.rotate(base_polygon, angle_deg, origin=(0, 0))
        candidate_poly = affinity.translate(
            rotated,
            xoff=float(cand_x * scale_factor),
            yoff=float(cand_y * scale_factor),
        )

        if collides(candidate_poly, polygons, idx):
            continue

        other_bounds = compute_bounds_excluding(polygons, idx)
        current_bounds = merge_bounds(other_bounds, polygons[idx].bounds)
        candidate_bounds = merge_bounds(other_bounds, candidate_poly.bounds)

        current_side = side_length_from_bounds(current_bounds, scale_factor)
        candidate_side = side_length_from_bounds(candidate_bounds, scale_factor)
        delta = float(candidate_side - current_side)

        if delta <= 0:
            accept = True
        else:
            accept = rng.random() < math.exp(-delta / max(temp, 1e-9))

        if accept:
            tree.center_x = cand_x
            tree.center_y = cand_y
            tree.angle = Decimal(str(angle_deg))
            tree.polygon = candidate_poly
            polygons[idx] = candidate_poly


def recenter_trees(
    placed_trees: list[ChristmasTree],
    *,
    scale_factor: Decimal,
    xy_limit: float,
) -> None:
    if not placed_trees:
        return

    center_x, center_y = compute_bounds_center(placed_trees, scale_factor)
    if center_x == 0 and center_y == 0:
        return

    limit = Decimal(str(xy_limit))
    max_after_shift = max(max(abs(tree.center_x - center_x), abs(tree.center_y - center_y)) for tree in placed_trees)
    if max_after_shift > limit:
        return

    dx = -center_x * scale_factor
    dy = -center_y * scale_factor

    for tree in placed_trees:
        tree.center_x -= center_x
        tree.center_y -= center_y
        tree.polygon = affinity.translate(tree.polygon, xoff=float(dx), yoff=float(dy))


def plot_results(
    side_length: Decimal,
    placed_trees: list[ChristmasTree],
    num_trees: int,
    scale_factor: Decimal,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import Rectangle  # noqa: PLC0415

    _, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])  # type: ignore

    bounds = compute_bounds(placed_trees)

    for i, tree in enumerate(placed_trees):
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(val) / scale_factor for val in x_scaled]
        y = [Decimal(val) / scale_factor for val in y_scaled]
        ax.plot(x, y, color=colors[i])  # type: ignore
        ax.fill(x, y, alpha=0.5, color=colors[i])

    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny

    square_x = minx if width >= height else minx - (side_length - width) / 2
    square_y = miny if height >= width else miny - (side_length - height) / 2
    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor="red",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(bounding_square)

    padding = Decimal("0.5")
    ax.set_xlim(float(square_x - padding), float(square_x + side_length + padding))
    ax.set_ylim(float(square_y - padding), float(square_y + side_length + padding))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(f"{num_trees} Trees: {side_length:.12f}")
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def write_submission(
    tree_data: list[list[Decimal]],
    index: list[str],
    output_path: Path,
) -> None:
    cols = ["x", "y", "deg"]
    submission = pd.DataFrame(index=index, columns=cols, data=tree_data).rename_axis("id")

    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)

    for col in submission.columns:
        submission[col] = "s" + submission[col].astype("string")

    submission.to_csv(output_path)


def compute_score(side_lengths: list[float]) -> float:
    return float(sum((length**2) / (idx + 1) for idx, length in enumerate(side_lengths)))


def validate_xy_bounds(trees: list[ChristmasTree], xy_limit: float) -> None:
    if not trees:
        return
    limit = Decimal(str(xy_limit))
    max_val = max(max(abs(tree.center_x), abs(tree.center_y)) for tree in trees)
    if max_val > limit:
        raise ValueError(f"center coordinate out of bounds: max_abs={max_val} limit={limit}")


def summarize_metrics(
    side_lengths: Iterable[float],
    cfg: ExpConfig,
) -> dict[str, float | int | str | list[float]]:
    lengths = list(side_lengths)
    return {
        "n_trees": len(lengths),
        "side_lengths": lengths,
        "side_sum": float(np.sum(lengths)) if lengths else 0.0,
        "side_mean": float(np.mean(lengths)) if lengths else 0.0,
        "side_std": float(np.std(lengths)) if lengths else 0.0,
        "score": compute_score(lengths) if lengths else 0.0,
        "note": cfg.note,
    }


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:  # noqa: PLR0915
    output_dir = init_output_dir(cfg)
    logger = get_logger(__name__, output_dir)

    logger.info("exp_name=%s", cfg.env.exp_name)
    logger.info("output_dir=%s", str(output_dir))
    logger.info("overrides=%s", ", ".join(HydraConfig.get().overrides.task))
    logger.info("cfg.exp=%s", OmegaConf.to_container(cfg.exp, resolve=True))

    set_seed(cfg.exp.seed)
    dump_config(cfg, output_dir)

    getcontext().prec = cfg.exp.decimal_precision
    scale_factor = Decimal(cfg.exp.scale_factor)
    base_polygon = build_tree_base_polygon(scale_factor)

    rng = random.Random(cfg.exp.seed)
    max_trees = cfg.exp.max_trees
    if cfg.exp.debug:
        max_trees = min(max_trees, cfg.exp.debug_max_trees)
        logger.info("debug mode: using %d trees", max_trees)

    submission_index = build_submission_index(max_trees)
    tree_data: list[list[Decimal]] = []
    candidates: list[list[ChristmasTree]] = [[]]
    side_lengths: list[float] = []
    beam_width = max(cfg.exp.beam_width, 1)
    beam_expansions = max(cfg.exp.beam_expansions, 1)
    beam_start_n = max(cfg.exp.beam_start_n, 1)

    with trace("packing"):
        for n_trees in tqdm(
            range(1, max_trees + 1),
            desc="packing",
            disable=not cfg.exp.progress,
        ):
            use_beam = beam_width > 1 and n_trees >= beam_start_n
            base_candidates = candidates if use_beam else [candidates[0]]
            expansions = beam_expansions if use_beam else 1

            new_candidates: list[tuple[list[ChristmasTree], Decimal]] = []

            for cand in base_candidates:
                for _ in range(expansions):
                    placed_trees = clone_trees(cand)
                    placed_trees, _ = initialize_trees(
                        n_trees,
                        rng=rng,
                        base_polygon=base_polygon,
                        scale_factor=scale_factor,
                        cfg=cfg.exp,
                        existing_trees=placed_trees,
                    )

                    if cfg.exp.compact_iters > 0 and n_trees % max(cfg.exp.compact_every, 1) == 0:
                        compact_trees(
                            placed_trees,
                            rng=rng,
                            scale_factor=scale_factor,
                            cfg=cfg.exp,
                        )
                    if cfg.exp.reinsert_iters > 0 and n_trees % max(cfg.exp.reinsert_every, 1) == 0:
                        reinsert_trees(
                            placed_trees,
                            rng=rng,
                            base_polygon=base_polygon,
                            scale_factor=scale_factor,
                            cfg=cfg.exp,
                        )

                    if cfg.exp.anneal_iters > 0 and n_trees % max(cfg.exp.anneal_every, 1) == 0:
                        anneal_trees(
                            placed_trees,
                            rng=rng,
                            base_polygon=base_polygon,
                            scale_factor=scale_factor,
                            cfg=cfg.exp,
                        )

                    if cfg.exp.recenter_every > 0 and n_trees % max(cfg.exp.recenter_every, 1) == 0:
                        recenter_trees(
                            placed_trees,
                            scale_factor=scale_factor,
                            xy_limit=cfg.exp.xy_limit,
                        )

                    side_length = compute_side_length(placed_trees, scale_factor)
                    validate_xy_bounds(placed_trees, cfg.exp.xy_limit)
                    new_candidates.append((placed_trees, side_length))

            new_candidates.sort(key=lambda item: item[1])
            best_trees, best_side = new_candidates[0]
            side_lengths.append(float(best_side))

            if cfg.exp.plot_every > 0 and n_trees % cfg.exp.plot_every == 0:
                plot_path = output_dir / f"plot_{n_trees:03d}.png"
                plot_results(best_side, best_trees, n_trees, scale_factor, plot_path)

            for tree in best_trees:
                tree_data.append([tree.center_x, tree.center_y, tree.angle])

            candidates = [trees for trees, _ in new_candidates[:beam_width]] if use_beam else [best_trees]

    metrics = summarize_metrics(side_lengths, cfg.exp)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    submission_path = output_dir / "submission.csv"
    write_submission(tree_data, submission_index, submission_path)
    logger.info("submission saved: %s", submission_path)


if __name__ == "__main__":
    main()
