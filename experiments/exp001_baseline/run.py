from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Iterable

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

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
    minor = HydraConfig.get().runtime.choices.exp
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


def create_tree(angle_deg: float, base_polygon: Polygon) -> ChristmasTree:
    angle = Decimal(str(angle_deg))
    rotated = affinity.rotate(base_polygon, float(angle), origin=(0, 0))
    return ChristmasTree(
        center_x=Decimal("0"),
        center_y=Decimal("0"),
        angle=angle,
        polygon=rotated,
    )


def generate_weighted_angle(rng: random.Random) -> float:
    while True:
        angle = rng.uniform(0, 2 * math.pi)
        if rng.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def initialize_trees(
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

    if num_to_add > 0:
        unplaced_trees = [create_tree(rng.uniform(0, 360), base_polygon) for _ in range(num_to_add)]
        if not placed_trees:
            placed_trees.append(unplaced_trees.pop(0))

        for tree_to_place in unplaced_trees:
            placed_polygons = [p.polygon for p in placed_trees]
            tree_index = STRtree(placed_polygons)

            best_px = None
            best_py = None
            min_radius = Decimal("Infinity")

            for _ in range(max(cfg.attempts_per_tree, 1)):
                angle = (
                    generate_weighted_angle(rng)
                    if cfg.angle_weighted
                    else rng.uniform(0, 2 * math.pi)
                )
                vx = Decimal(str(math.cos(angle)))
                vy = Decimal(str(math.sin(angle)))

                radius = start_radius
                collision_found = False
                while radius >= 0:
                    px = radius * vx
                    py = radius * vy

                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * scale_factor),
                        yoff=float(py * scale_factor),
                    )

                    possible_indices = tree_index.query(candidate_poly)
                    if any(
                        candidate_poly.intersects(placed_polygons[i])
                        and not candidate_poly.touches(placed_polygons[i])
                        for i in possible_indices
                    ):
                        collision_found = True
                        break
                    radius -= step_in

                if collision_found:
                    while True:
                        radius += step_out
                        px = radius * vx
                        py = radius * vy

                        candidate_poly = affinity.translate(
                            tree_to_place.polygon,
                            xoff=float(px * scale_factor),
                            yoff=float(py * scale_factor),
                        )

                        possible_indices = tree_index.query(candidate_poly)
                        if not any(
                            candidate_poly.intersects(placed_polygons[i])
                            and not candidate_poly.touches(placed_polygons[i])
                            for i in possible_indices
                        ):
                            break
                else:
                    radius = Decimal("0")
                    px = Decimal("0")
                    py = Decimal("0")

                if radius < min_radius:
                    min_radius = radius
                    best_px = px
                    best_py = py

            tree_to_place.center_x = best_px
            tree_to_place.center_y = best_py
            tree_to_place.polygon = affinity.translate(
                tree_to_place.polygon,
                xoff=float(tree_to_place.center_x * scale_factor),
                yoff=float(tree_to_place.center_y * scale_factor),
            )
            placed_trees.append(tree_to_place)

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)

    return placed_trees, side_length


def plot_results(
    side_length: Decimal,
    placed_trees: list[ChristmasTree],
    num_trees: int,
    scale_factor: Decimal,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    for i, tree in enumerate(placed_trees):
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(val) / scale_factor for val in x_scaled]
        y = [Decimal(val) / scale_factor for val in y_scaled]
        ax.plot(x, y, color=colors[i])
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
        raise ValueError(
            f"center coordinate out of bounds: max_abs={max_val} limit={limit}"
        )


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
def main(cfg: Config) -> None:
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
    placed_trees: list[ChristmasTree] = []
    side_lengths: list[float] = []

    with trace("packing"):
        for n_trees in range(1, max_trees + 1):
            placed_trees, side_length = initialize_trees(
                n_trees,
                rng=rng,
                base_polygon=base_polygon,
                scale_factor=scale_factor,
                cfg=cfg.exp,
                existing_trees=placed_trees,
            )
            side_lengths.append(float(side_length))

            if cfg.exp.plot_every > 0 and n_trees % cfg.exp.plot_every == 0:
                plot_path = output_dir / f"plot_{n_trees:03d}.png"
                plot_results(side_length, placed_trees, n_trees, scale_factor, plot_path)

            for tree in placed_trees:
                tree_data.append([tree.center_x, tree.center_y, tree.angle])

    validate_xy_bounds(placed_trees, cfg.exp.xy_limit)

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
