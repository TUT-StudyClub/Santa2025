import math
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import yaml
from numba import njit
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
def resolve_config_path() -> str:
    config_name = "000"
    for arg in sys.argv[1:]:
        if arg.startswith("exp="):
            config_name = arg.split("=", 1)[1]
        elif arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
    return os.path.join("experiments", "exp007_intensive", "exp", f"{config_name}.yaml")


def merge_defaults(defaults: dict, overrides: dict | None) -> dict:
    merged = defaults.copy()
    if overrides:
        merged.update(overrides)
    return merged


CONFIG_PATH = resolve_config_path()
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

MULTI_START_DEFAULT = {
    "restarts": 1,
    "seed_stride": 1000,
    "jitter_xy": 0.0,
    "jitter_deg": 0.0,
    "jitter_ab": 0.0,
    "jitter_shear": 0.0,
}
POLISH_DEFAULT = {"enabled": False}
HILL_DEFAULT = {"enabled": False}
ROTATE_DEFAULT = {"enabled": False}
SHRINK_ANISO_DEFAULT = {"enabled": False}
SHRINK_DEFAULT = {"enabled": False}
MULTI_START = merge_defaults(MULTI_START_DEFAULT, CONFIG.get("multi_start"))
POLISH = merge_defaults(POLISH_DEFAULT, CONFIG.get("polish"))
HILL = merge_defaults(HILL_DEFAULT, CONFIG.get("hill"))
ROTATE = merge_defaults(ROTATE_DEFAULT, CONFIG.get("rotate"))
SHRINK_ANISO = merge_defaults(SHRINK_ANISO_DEFAULT, CONFIG.get("shrink_aniso"))
SHRINK = merge_defaults(SHRINK_DEFAULT, CONFIG.get("shrink"))

TREE_CFG = CONFIG["tree_shape"]
TRUNK_W = float(TREE_CFG["trunk_w"])
TRUNK_H = float(TREE_CFG["trunk_h"])
BASE_W = float(TREE_CFG["base_w"])
MID_W = float(TREE_CFG["mid_w"])
TOP_W = float(TREE_CFG["top_w"])
TIP_Y = float(TREE_CFG["tip_y"])
TIER_1_Y = float(TREE_CFG["tier_1_y"])
TIER_2_Y = float(TREE_CFG["tier_2_y"])
BASE_Y = float(TREE_CFG["base_y"])
TRUNK_BOTTOM_Y = float(TREE_CFG["trunk_bottom_y"])


# -----------------------------------------------------------------------------
# Geometry Utils (Numba)
# -----------------------------------------------------------------------------
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
def polygon_bounds(vertices: np.ndarray) -> tuple[float, float, float, float]:
    min_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_x = vertices[0, 0]
    max_y = vertices[0, 1]
    for i in range(1, vertices.shape[0]):
        x = vertices[i, 0]
        y = vertices[i, 1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def point_in_polygon(px: float, py: float, vertices: np.ndarray) -> bool:
    n = vertices.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i, 0], vertices[i, 1]
        xj, yj = vertices[j, 0], vertices[j, 1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@njit(cache=True)
def segments_intersect(
    p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float, p4x: float, p4y: float
) -> bool:
    d1x = p2x - p1x
    d1y = p2y - p1y
    d2x = p4x - p3x
    d2y = p4y - p3y
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-10:
        return False
    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / det
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / det
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


@njit(cache=True)
def polygons_overlap(verts1: np.ndarray, verts2: np.ndarray) -> bool:
    min_x1, min_y1, max_x1, max_y1 = polygon_bounds(verts1)
    min_x2, min_y2, max_x2, max_y2 = polygon_bounds(verts2)

    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False

    for i in range(verts1.shape[0]):
        if point_in_polygon(verts1[i, 0], verts1[i, 1], verts2):
            return True
    for i in range(verts2.shape[0]):
        if point_in_polygon(verts2[i, 0], verts2[i, 1], verts1):
            return True

    n1 = verts1.shape[0]
    n2 = verts2.shape[0]
    for i in range(n1):
        j = (i + 1) % n1
        p1x, p1y = verts1[i, 0], verts1[i, 1]
        p2x, p2y = verts1[j, 0], verts1[j, 1]
        for k in range(n2):
            m = (k + 1) % n2
            p3x, p3y = verts2[k, 0], verts2[k, 1]
            p4x, p4y = verts2[m, 0], verts2[m, 1]
            if segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
                return True
    return False


@njit(cache=True)
def has_any_overlap(all_vertices: list[np.ndarray]) -> bool:
    n = len(all_vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(all_vertices[i], all_vertices[j]):
                return True
    return False


@njit(cache=True)
def compute_bounding_box(all_vertices: list[np.ndarray]) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for verts in all_vertices:
        x1, y1, x2, y2 = polygon_bounds(verts)
        if x1 < min_x:
            min_x = x1
        if y1 < min_y:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def get_side_length(all_vertices: list[np.ndarray]) -> float:
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True)
def calculate_score_numba(all_vertices: list[np.ndarray]) -> float:
    side = get_side_length(all_vertices)
    return side * side / len(all_vertices)


# -----------------------------------------------------------------------------
# Grid Generation
# -----------------------------------------------------------------------------
@njit(cache=True)
def create_grid_vertices_extended(
    seed_xs: np.ndarray,
    seed_ys: np.ndarray,
    seed_degs: np.ndarray,
    a: float,
    b: float,
    shear_x: float,
    shear_y: float,
    ncols: int,
    nrows: int,
    append_x: bool,
    append_y: bool,
) -> list[np.ndarray]:
    n_seeds = len(seed_xs)
    all_vertices = []

    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                cx = seed_xs[s] + col * a + row * shear_x
                cy = seed_ys[s] + row * b + col * shear_y
                all_vertices.append(get_tree_vertices(cx, cy, seed_degs[s]))

    if append_x and n_seeds > 1:
        for row in range(nrows):
            cx = seed_xs[1] + ncols * a + row * shear_x
            cy = seed_ys[1] + row * b + ncols * shear_y
            all_vertices.append(get_tree_vertices(cx, cy, seed_degs[1]))

    if append_y and n_seeds > 1:
        for col in range(ncols):
            cx = seed_xs[1] + col * a + nrows * shear_x
            cy = seed_ys[1] + nrows * b + col * shear_y
            all_vertices.append(get_tree_vertices(cx, cy, seed_degs[1]))

    return all_vertices


@njit(cache=True)
def get_initial_translations(
    seed_xs: np.ndarray, seed_ys: np.ndarray, seed_degs: np.ndarray
) -> tuple[float, float]:
    seed_vertices = [
        get_tree_vertices(seed_xs[i], seed_ys[i], seed_degs[i]) for i in range(len(seed_xs))
    ]
    min_x, min_y, max_x, max_y = compute_bounding_box(seed_vertices)
    return max_x - min_x, max_y - min_y


# -----------------------------------------------------------------------------
# Optimization Logic (Simulated Annealing)
# -----------------------------------------------------------------------------
@njit(cache=True)
def sa_optimize_improved(
    seed_xs_init: np.ndarray,
    seed_ys_init: np.ndarray,
    seed_degs_init: np.ndarray,
    a_init: float,
    b_init: float,
    shear_x_init: float,
    shear_y_init: float,
    ncols: int,
    nrows: int,
    append_x: bool,
    append_y: bool,
    Tmax: float,
    Tmin: float,
    nsteps: int,
    nsteps_per_T: int,
    position_delta: float,
    angle_delta: float,
    angle_delta2: float,
    delta_t: float,
    shear_delta: float,
    random_seed: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    np.random.seed(random_seed)
    n_seeds = len(seed_xs_init)

    seed_xs = seed_xs_init.copy()
    seed_ys = seed_ys_init.copy()
    seed_degs = seed_degs_init.copy()
    a, b = a_init, b_init
    shear_x, shear_y = shear_x_init, shear_y_init

    all_vertices = create_grid_vertices_extended(
        seed_xs, seed_ys, seed_degs, a, b, shear_x, shear_y, ncols, nrows, append_x, append_y
    )
    if has_any_overlap(all_vertices):
        a_test, b_test = get_initial_translations(seed_xs, seed_ys, seed_degs)
        a = max(a, a_test * 1.5)
        b = max(b, b_test * 1.5)
        all_vertices = create_grid_vertices_extended(
            seed_xs, seed_ys, seed_degs, a, b, shear_x, shear_y, ncols, nrows, append_x, append_y
        )

    current_score = calculate_score_numba(all_vertices)
    best_score = current_score
    best_xs, best_ys, best_degs = seed_xs.copy(), seed_ys.copy(), seed_degs.copy()
    best_a, best_b = a, b
    best_shear_x, best_shear_y = shear_x, shear_y

    T = Tmax
    Tfactor = -math.log(Tmax / Tmin)
    n_move_types = n_seeds + 2

    old_x, old_y, old_deg = 0.0, 0.0, 0.0
    old_a, old_b = 0.0, 0.0
    old_shear_x, old_shear_y = 0.0, 0.0

    patience = nsteps // 5
    no_improve_count = 0
    last_best_score = best_score

    for step in range(nsteps):
        progress = step / nsteps
        decay = 1.0 - 0.9 * progress
        cur_pos_delta = position_delta * decay
        cur_ang_delta = angle_delta * decay
        cur_shear_delta = shear_delta * decay

        for _ in range(nsteps_per_T):
            move_type = np.random.randint(0, n_move_types)

            if move_type < n_seeds:
                i = move_type
                old_x, old_y, old_deg = seed_xs[i], seed_ys[i], seed_degs[i]

                dx = (np.random.random() * 2.0 - 1.0) * cur_pos_delta
                dy = (np.random.random() * 2.0 - 1.0) * cur_pos_delta
                ddeg = (np.random.random() * 2.0 - 1.0) * cur_ang_delta

                seed_xs[i] += dx
                seed_ys[i] += dy
                seed_degs[i] = (seed_degs[i] + ddeg) % 360.0

            elif move_type == n_seeds:
                old_a, old_b = a, b
                old_shear_x, old_shear_y = shear_x, shear_y
                da = (np.random.random() * 2.0 - 1.0) * delta_t
                db = (np.random.random() * 2.0 - 1.0) * delta_t
                dsx = (np.random.random() * 2.0 - 1.0) * cur_shear_delta
                dsy = (np.random.random() * 2.0 - 1.0) * cur_shear_delta
                a += a * da
                b += b * db
                shear_x += b * dsx
                shear_y += a * dsy

            else:
                old_degs_array = seed_degs.copy()
                ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta2
                for k in range(n_seeds):
                    seed_degs[k] = (seed_degs[k] + ddeg) % 360.0

            test_vertices = create_grid_vertices_extended(
                seed_xs, seed_ys, seed_degs, a, b, shear_x, shear_y, 2, 2, False, False
            )
            if has_any_overlap(test_vertices):
                if move_type < n_seeds:
                    seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = (
                        old_x,
                        old_y,
                        old_deg,
                    )
                elif move_type == n_seeds:
                    a, b = old_a, old_b
                    shear_x, shear_y = old_shear_x, old_shear_y
                else:
                    seed_degs[:] = old_degs_array[:]
                continue

            new_vertices = create_grid_vertices_extended(
                seed_xs,
                seed_ys,
                seed_degs,
                a,
                b,
                shear_x,
                shear_y,
                ncols,
                nrows,
                append_x,
                append_y,
            )
            if has_any_overlap(new_vertices):
                if move_type < n_seeds:
                    seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = (
                        old_x,
                        old_y,
                        old_deg,
                    )
                elif move_type == n_seeds:
                    a, b = old_a, old_b
                    shear_x, shear_y = old_shear_x, old_shear_y
                else:
                    seed_degs[:] = old_degs_array[:]
                continue

            new_score = calculate_score_numba(new_vertices)
            delta = new_score - current_score
            accept = False

            if delta < 0:
                accept = True
            elif T > 1e-10 and np.random.random() < math.exp(-delta / T):
                accept = True

            if accept:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_xs, best_ys, best_degs = seed_xs.copy(), seed_ys.copy(), seed_degs.copy()
                    best_a, best_b = a, b
                    best_shear_x, best_shear_y = shear_x, shear_y
            else:
                if move_type < n_seeds:
                    seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = (
                        old_x,
                        old_y,
                        old_deg,
                    )
                elif move_type == n_seeds:
                    a, b = old_a, old_b
                    shear_x, shear_y = old_shear_x, old_shear_y
                else:
                    seed_degs[:] = old_degs_array[:]

        if last_best_score - best_score > 1e-9:
            no_improve_count = 0
            last_best_score = best_score
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

        T = Tmax * math.exp(Tfactor * (step + 1) / nsteps)

    return best_score, best_xs, best_ys, best_degs, best_a, best_b, best_shear_x, best_shear_y


@njit(cache=True)
def hill_refine(
    seed_xs: np.ndarray,
    seed_ys: np.ndarray,
    seed_degs: np.ndarray,
    a: float,
    b: float,
    shear_x: float,
    shear_y: float,
    ncols: int,
    nrows: int,
    append_x: bool,
    append_y: bool,
    position_delta: float,
    angle_delta: float,
    angle_delta2: float,
    delta_t: float,
    shear_delta: float,
    n_steps: int,
    step_scale: float,
    random_seed: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    if n_steps <= 0 or step_scale <= 0.0:
        all_vertices = create_grid_vertices_extended(
            seed_xs, seed_ys, seed_degs, a, b, shear_x, shear_y, ncols, nrows, append_x, append_y
        )
        score = calculate_score_numba(all_vertices)
        return score, seed_xs, seed_ys, seed_degs, a, b, shear_x, shear_y

    np.random.seed(random_seed)
    n_seeds = len(seed_xs)
    n_move_types = n_seeds + 2

    all_vertices = create_grid_vertices_extended(
        seed_xs, seed_ys, seed_degs, a, b, shear_x, shear_y, ncols, nrows, append_x, append_y
    )
    if has_any_overlap(all_vertices):
        score = calculate_score_numba(all_vertices)
        return score, seed_xs, seed_ys, seed_degs, a, b, shear_x, shear_y

    current_score = calculate_score_numba(all_vertices)

    old_x, old_y, old_deg = 0.0, 0.0, 0.0
    old_a, old_b = 0.0, 0.0
    old_shear_x, old_shear_y = 0.0, 0.0
    old_degs_array = seed_degs.copy()

    for _ in range(n_steps):
        move_type = np.random.randint(0, n_move_types)

        if move_type < n_seeds:
            i = move_type
            old_x, old_y, old_deg = seed_xs[i], seed_ys[i], seed_degs[i]
            dx = (np.random.random() * 2.0 - 1.0) * position_delta * step_scale
            dy = (np.random.random() * 2.0 - 1.0) * position_delta * step_scale
            ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta * step_scale
            seed_xs[i] += dx
            seed_ys[i] += dy
            seed_degs[i] = (seed_degs[i] + ddeg) % 360.0

        elif move_type == n_seeds:
            old_a, old_b = a, b
            old_shear_x, old_shear_y = shear_x, shear_y
            da = (np.random.random() * 2.0 - 1.0) * delta_t * step_scale
            db = (np.random.random() * 2.0 - 1.0) * delta_t * step_scale
            dsx = (np.random.random() * 2.0 - 1.0) * shear_delta * step_scale
            dsy = (np.random.random() * 2.0 - 1.0) * shear_delta * step_scale
            a += a * da
            b += b * db
            shear_x += b * dsx
            shear_y += a * dsy

        else:
            old_degs_array[:] = seed_degs[:]
            ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta2 * step_scale
            for k in range(n_seeds):
                seed_degs[k] = (seed_degs[k] + ddeg) % 360.0

        new_vertices = create_grid_vertices_extended(
            seed_xs,
            seed_ys,
            seed_degs,
            a,
            b,
            shear_x,
            shear_y,
            ncols,
            nrows,
            append_x,
            append_y,
        )
        if has_any_overlap(new_vertices):
            if move_type < n_seeds:
                seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = (
                    old_x,
                    old_y,
                    old_deg,
                )
            elif move_type == n_seeds:
                a, b = old_a, old_b
                shear_x, shear_y = old_shear_x, old_shear_y
            else:
                seed_degs[:] = old_degs_array[:]
            continue

        new_score = calculate_score_numba(new_vertices)
        if new_score < current_score - 1e-9:
            current_score = new_score
        else:
            if move_type < n_seeds:
                seed_xs[move_type], seed_ys[move_type], seed_degs[move_type] = (
                    old_x,
                    old_y,
                    old_deg,
                )
            elif move_type == n_seeds:
                a, b = old_a, old_b
                shear_x, shear_y = old_shear_x, old_shear_y
            else:
                seed_degs[:] = old_degs_array[:]

    return current_score, seed_xs, seed_ys, seed_degs, a, b, shear_x, shear_y


@njit(cache=True)
def get_final_grid_positions_extended(
    seed_xs: np.ndarray,
    seed_ys: np.ndarray,
    seed_degs: np.ndarray,
    a: float,
    b: float,
    shear_x: float,
    shear_y: float,
    ncols: int,
    nrows: int,
    append_x: bool,
    append_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_seeds = len(seed_xs)
    n_base = n_seeds * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_total = n_base + n_append_x + n_append_y

    xs = np.empty(n_total, dtype=np.float64)
    ys = np.empty(n_total, dtype=np.float64)
    degs = np.empty(n_total, dtype=np.float64)

    idx = 0
    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                xs[idx] = seed_xs[s] + col * a + row * shear_x
                ys[idx] = seed_ys[s] + row * b + col * shear_y
                degs[idx] = seed_degs[s]
                idx += 1

    if append_x and n_seeds > 1:
        for row in range(nrows):
            xs[idx] = seed_xs[1] + ncols * a + row * shear_x
            ys[idx] = seed_ys[1] + row * b + ncols * shear_y
            degs[idx] = seed_degs[1]
            idx += 1

    if append_y and n_seeds > 1:
        for col in range(ncols):
            xs[idx] = seed_xs[1] + col * a + nrows * shear_x
            ys[idx] = seed_ys[1] + nrows * b + col * shear_y
            degs[idx] = seed_degs[1]
            idx += 1

    return xs, ys, degs


@njit(cache=True)
def deletion_cascade_numba(
    all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_start = np.zeros(201, dtype=np.int64)
    for n in range(1, 201):
        group_start[n] = group_start[n - 1] + (n - 1) if n > 1 else 0

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    side_lengths = np.zeros(201, dtype=np.float64)
    for n in range(1, 201):
        start = group_start[n]
        vertices = [
            get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) for i in range(start, start + n)
        ]
        side_lengths[n] = get_side_length(vertices)

    for n in range(200, 1, -1):
        start_n = group_start[n]
        start_prev = group_start[n - 1]
        best_side = side_lengths[n - 1]
        best_delete_idx = -1

        for del_idx in range(n):
            vertices = []
            for i in range(n):
                if i != del_idx:
                    idx = start_n + i
                    vertices.append(get_tree_vertices(new_xs[idx], new_ys[idx], new_degs[idx]))

            candidate_side = get_side_length(vertices)
            if candidate_side < best_side:
                best_side = candidate_side
                best_delete_idx = del_idx

        if best_delete_idx >= 0:
            out_idx = start_prev
            for i in range(n):
                if i != best_delete_idx:
                    in_idx = start_n + i
                    new_xs[out_idx] = new_xs[in_idx]
                    new_ys[out_idx] = new_ys[in_idx]
                    new_degs[out_idx] = new_degs[in_idx]
                    out_idx += 1
            side_lengths[n - 1] = best_side

    return new_xs, new_ys, new_degs, side_lengths


# -----------------------------------------------------------------------------
# Data & Worker Handlers
# -----------------------------------------------------------------------------
def build_polish_params(sa_params: dict, polish_cfg: dict) -> dict | None:
    if not polish_cfg.get("enabled", False):
        return None
    params = sa_params.copy()
    for key, value in polish_cfg.items():
        if key != "enabled":
            params[key] = value
    return params


def build_hill_params(sa_params: dict, hill_cfg: dict) -> dict | None:
    if not hill_cfg.get("enabled", False):
        return None
    steps = int(hill_cfg.get("steps", 0))
    steps_ratio = float(hill_cfg.get("steps_ratio", 0.05))
    if steps <= 0:
        steps = max(1, int(sa_params["nsteps"] * steps_ratio))
    decay = float(hill_cfg.get("decay", 0.1))
    seed_offset = int(hill_cfg.get("seed_offset", 7777))
    return {"steps": steps, "decay": decay, "seed_offset": seed_offset}


def apply_jitter(
    seed_xs: np.ndarray,
    seed_ys: np.ndarray,
    seed_degs: np.ndarray,
    a_init: float,
    b_init: float,
    shear_x: float,
    shear_y: float,
    jitter_xy: float,
    jitter_deg: float,
    jitter_ab: float,
    jitter_shear: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    xs = seed_xs.copy()
    ys = seed_ys.copy()
    degs = seed_degs.copy()

    if jitter_xy > 0.0:
        xs += rng.uniform(-jitter_xy, jitter_xy, size=xs.shape)
        ys += rng.uniform(-jitter_xy, jitter_xy, size=ys.shape)
    if jitter_deg > 0.0:
        degs = (degs + rng.uniform(-jitter_deg, jitter_deg, size=degs.shape)) % 360.0
    if jitter_ab > 0.0:
        a_init *= 1.0 + rng.uniform(-jitter_ab, jitter_ab)
        b_init *= 1.0 + rng.uniform(-jitter_ab, jitter_ab)
        if a_init <= 0.0:
            a_init = 1e-6
        if b_init <= 0.0:
            b_init = 1e-6
    if jitter_shear > 0.0:
        shear_x += b_init * rng.uniform(-jitter_shear, jitter_shear)
        shear_y += a_init * rng.uniform(-jitter_shear, jitter_shear)

    return xs, ys, degs, a_init, b_init, shear_x, shear_y


def optimize_grid_config(args: tuple) -> tuple[int, float, list[tuple[float, float, float]]]:
    (
        ncols,
        nrows,
        append_x,
        append_y,
        initial_seeds,
        a_init,
        b_init,
        shear_x_init,
        shear_y_init,
        params,
        multi_start,
        polish_params,
        hill_params,
        seed,
    ) = args

    seed_xs_base = np.array([s[0] for s in initial_seeds], dtype=np.float64)
    seed_ys_base = np.array([s[1] for s in initial_seeds], dtype=np.float64)
    seed_degs_base = np.array([s[2] for s in initial_seeds], dtype=np.float64)

    n_trees = (
        len(initial_seeds) * ncols * nrows + (nrows if append_x else 0) + (ncols if append_y else 0)
    )

    restarts = max(1, int(multi_start.get("restarts", 1)))
    seed_stride = int(multi_start.get("seed_stride", 1000))
    jitter_xy = float(multi_start.get("jitter_xy", 0.0))
    jitter_deg = float(multi_start.get("jitter_deg", 0.0))
    jitter_ab = float(multi_start.get("jitter_ab", 0.0))
    jitter_shear = float(multi_start.get("jitter_shear", 0.0))

    best_score = math.inf
    best_xs = seed_xs_base
    best_ys = seed_ys_base
    best_degs = seed_degs_base
    best_a = a_init
    best_b = b_init
    best_shear_x = shear_x_init
    best_shear_y = shear_y_init

    for restart in range(restarts):
        run_seed = seed + restart * seed_stride
        rng = np.random.default_rng(run_seed)
        seed_xs, seed_ys, seed_degs, a_jit, b_jit, shear_x_jit, shear_y_jit = apply_jitter(
            seed_xs_base,
            seed_ys_base,
            seed_degs_base,
            a_init,
            b_init,
            shear_x_init,
            shear_y_init,
            jitter_xy,
            jitter_deg,
            jitter_ab,
            jitter_shear,
            rng,
        )

        score, xs, ys, degs, a, b, shear_x, shear_y = sa_optimize_improved(
            seed_xs,
            seed_ys,
            seed_degs,
            a_jit,
            b_jit,
            shear_x_jit,
            shear_y_jit,
            ncols,
            nrows,
            append_x,
            append_y,
            params["Tmax"],
            params["Tmin"],
            params["nsteps"],
            params["nsteps_per_T"],
            params["position_delta"],
            params["angle_delta"],
            params["angle_delta2"],
            params["delta_t"],
            params.get("shear_delta", 0.0),
            run_seed,
        )

        if polish_params is not None:
            polish_seed = run_seed + 17
            score_p, xs_p, ys_p, degs_p, a_p, b_p, shear_x_p, shear_y_p = sa_optimize_improved(
                xs,
                ys,
                degs,
                a,
                b,
                shear_x,
                shear_y,
                ncols,
                nrows,
                append_x,
                append_y,
                polish_params["Tmax"],
                polish_params["Tmin"],
                polish_params["nsteps"],
                polish_params["nsteps_per_T"],
                polish_params["position_delta"],
                polish_params["angle_delta"],
                polish_params["angle_delta2"],
                polish_params["delta_t"],
                polish_params.get("shear_delta", 0.0),
                polish_seed,
            )
            if score_p < score:
                score, xs, ys, degs, a, b, shear_x, shear_y = (
                    score_p,
                    xs_p,
                    ys_p,
                    degs_p,
                    a_p,
                    b_p,
                    shear_x_p,
                    shear_y_p,
                )

        if score < best_score:
            best_score = score
            best_xs, best_ys, best_degs = xs, ys, degs
            best_a, best_b = a, b
            best_shear_x, best_shear_y = shear_x, shear_y

    if hill_params is not None:
        hill_steps = int(hill_params.get("steps", 0))
        hill_decay = float(hill_params.get("decay", 0.0))
        if hill_steps > 0 and hill_decay > 0.0:
            hill_seed = seed + int(hill_params.get("seed_offset", 0))
            score_h, xs_h, ys_h, degs_h, a_h, b_h, shear_x_h, shear_y_h = hill_refine(
                best_xs.copy(),
                best_ys.copy(),
                best_degs.copy(),
                best_a,
                best_b,
                best_shear_x,
                best_shear_y,
                ncols,
                nrows,
                append_x,
                append_y,
                params["position_delta"],
                params["angle_delta"],
                params["angle_delta2"],
                params["delta_t"],
                params.get("shear_delta", 0.0),
                hill_steps,
                hill_decay,
                hill_seed,
            )
            if score_h < best_score:
                best_score = score_h
                best_xs, best_ys, best_degs = xs_h, ys_h, degs_h
                best_a, best_b = a_h, b_h
                best_shear_x, best_shear_y = shear_x_h, shear_y_h

    final_xs, final_ys, final_degs = get_final_grid_positions_extended(
        best_xs,
        best_ys,
        best_degs,
        best_a,
        best_b,
        best_shear_x,
        best_shear_y,
        ncols,
        nrows,
        append_x,
        append_y,
    )

    tree_data = [(final_xs[i], final_ys[i], final_degs[i]) for i in range(len(final_xs))]
    return n_trees, best_score, tree_data


def save_submission(
    filepath: str, all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray
) -> None:
    rows = []
    idx = 0
    for n in range(1, 201):
        for t in range(n):
            rows.append(
                {
                    "id": f"{n:03d}_{t}",
                    "x": f"s{all_xs[idx]}",
                    "y": f"s{all_ys[idx]}",
                    "deg": f"s{all_degs[idx]}",
                }
            )
            idx += 1
    pd.DataFrame(rows).to_csv(filepath, index=False)


def generate_baseline_positions(tree_cfg: dict, max_trees: int = 200) -> tuple[np.ndarray, ...]:
    base_w = float(tree_cfg["base_w"])
    height = float(tree_cfg["tip_y"]) - float(tree_cfg["trunk_bottom_y"])
    spacing = max(base_w, height) * 1.05

    xs: list[float] = []
    ys: list[float] = []
    degs: list[float] = []

    for n in range(1, max_trees + 1):
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        x0 = -((cols - 1) / 2.0) * spacing
        y0 = -((rows - 1) / 2.0) * spacing
        for i in range(n):
            col = i % cols
            row = i // cols
            xs.append(x0 + col * spacing)
            ys.append(y0 + row * spacing)
            degs.append(0.0)

    return np.array(xs), np.array(ys), np.array(degs)


@njit(cache=True)
def compute_center_bounds(
    all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray, start_idx: int, n: int
) -> tuple[float, float]:
    vertices = []
    for i in range(n):
        idx = start_idx + i
        vertices.append(get_tree_vertices(all_xs[idx], all_ys[idx], all_degs[idx]))
    min_x, min_y, max_x, max_y = compute_bounding_box(vertices)
    return (min_x + max_x) * 0.5, (min_y + max_y) * 0.5


@njit(cache=True)
def can_scale_group(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_degs: np.ndarray,
    start_idx: int,
    n: int,
    center_x: float,
    center_y: float,
    scale: float,
) -> bool:
    vertices = []
    for i in range(n):
        idx = start_idx + i
        cx = center_x + scale * (all_xs[idx] - center_x)
        cy = center_y + scale * (all_ys[idx] - center_y)
        vertices.append(get_tree_vertices(cx, cy, all_degs[idx]))
    return not has_any_overlap(vertices)


@njit(cache=True)
def can_scale_group_aniso(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_degs: np.ndarray,
    start_idx: int,
    n: int,
    center_x: float,
    center_y: float,
    scale_x: float,
    scale_y: float,
) -> bool:
    vertices = []
    for i in range(n):
        idx = start_idx + i
        cx = center_x + scale_x * (all_xs[idx] - center_x)
        cy = center_y + scale_y * (all_ys[idx] - center_y)
        vertices.append(get_tree_vertices(cx, cy, all_degs[idx]))
    return not has_any_overlap(vertices)


@njit(cache=True)
def apply_group_scale_aniso(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    start_idx: int,
    n: int,
    center_x: float,
    center_y: float,
    scale_x: float,
    scale_y: float,
) -> None:
    for i in range(n):
        idx = start_idx + i
        all_xs[idx] = center_x + scale_x * (all_xs[idx] - center_x)
        all_ys[idx] = center_y + scale_y * (all_ys[idx] - center_y)


def load_submission_data(
    filepath: str, tree_cfg: dict, fallback_path: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(filepath):
        if fallback_path is None:
            fallback_path = os.path.join("submissions", "baseline_autogen.csv")
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        xs, ys, degs = generate_baseline_positions(tree_cfg)
        save_submission(fallback_path, xs, ys, degs)
        print(f"Baseline not found. Generated fallback at {fallback_path}")
        return xs, ys, degs

    df = pd.read_csv(filepath)
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


def shrink_positions(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_degs: np.ndarray,
    shrink_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    enabled = bool(shrink_cfg.get("enabled", False))
    if not enabled:
        return all_xs, all_ys

    min_scale = float(shrink_cfg.get("min_scale", 0.92))
    iters = int(shrink_cfg.get("iters", 24))
    n_min = int(shrink_cfg.get("n_min", 1))
    n_max = int(shrink_cfg.get("n_max", 200))
    center_mode = str(shrink_cfg.get("center", "bounds"))

    if min_scale <= 0.0:
        min_scale = 0.5

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()

    group_start = np.zeros(201, dtype=np.int64)
    for n in range(1, 201):
        group_start[n] = group_start[n - 1] + (n - 1) if n > 1 else 0

    for n in range(n_min, min(n_max, 200) + 1):
        if n <= 1:
            continue
        start = int(group_start[n])
        if center_mode == "mean":
            center_x = float(new_xs[start : start + n].mean())
            center_y = float(new_ys[start : start + n].mean())
        else:
            center_x, center_y = compute_center_bounds(new_xs, new_ys, all_degs, start, n)

        low = min_scale
        high = 1.0
        best = 1.0
        for _ in range(iters):
            mid = (low + high) * 0.5
            if can_scale_group(new_xs, new_ys, all_degs, start, n, center_x, center_y, mid):
                best = mid
                high = mid
            else:
                low = mid

        if best < 1.0 - 1e-6:
            for i in range(n):
                idx = start + i
                new_xs[idx] = center_x + best * (new_xs[idx] - center_x)
                new_ys[idx] = center_y + best * (new_ys[idx] - center_y)

    return new_xs, new_ys


def shrink_positions_aniso(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_degs: np.ndarray,
    shrink_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    enabled = bool(shrink_cfg.get("enabled", False))
    if not enabled:
        return all_xs, all_ys

    min_scale_x = float(shrink_cfg.get("min_scale_x", 0.92))
    min_scale_y = float(shrink_cfg.get("min_scale_y", 0.92))
    iters = int(shrink_cfg.get("iters", 24))
    n_min = int(shrink_cfg.get("n_min", 1))
    n_max = int(shrink_cfg.get("n_max", 200))
    center_mode = str(shrink_cfg.get("center", "bounds"))

    if min_scale_x <= 0.0:
        min_scale_x = 0.5
    if min_scale_y <= 0.0:
        min_scale_y = 0.5
    if iters < 1:
        iters = 1

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()

    group_start = np.zeros(201, dtype=np.int64)
    for n in range(1, 201):
        group_start[n] = group_start[n - 1] + (n - 1) if n > 1 else 0

    for n in range(n_min, min(n_max, 200) + 1):
        if n <= 1:
            continue
        start = int(group_start[n])
        if center_mode == "mean":
            center_x = float(new_xs[start : start + n].mean())
            center_y = float(new_ys[start : start + n].mean())
        else:
            center_x, center_y = compute_center_bounds(new_xs, new_ys, all_degs, start, n)

        low = min_scale_x
        high = 1.0
        best = 1.0
        for _ in range(iters):
            mid = (low + high) * 0.5
            if can_scale_group_aniso(
                new_xs, new_ys, all_degs, start, n, center_x, center_y, mid, 1.0
            ):
                best = mid
                high = mid
            else:
                low = mid

        if best < 1.0 - 1e-6:
            apply_group_scale_aniso(new_xs, new_ys, start, n, center_x, center_y, best, 1.0)

        if center_mode == "mean":
            center_x = float(new_xs[start : start + n].mean())
            center_y = float(new_ys[start : start + n].mean())
        else:
            center_x, center_y = compute_center_bounds(new_xs, new_ys, all_degs, start, n)

        low = min_scale_y
        high = 1.0
        best = 1.0
        for _ in range(iters):
            mid = (low + high) * 0.5
            if can_scale_group_aniso(
                new_xs, new_ys, all_degs, start, n, center_x, center_y, 1.0, mid
            ):
                best = mid
                high = mid
            else:
                low = mid

        if best < 1.0 - 1e-6:
            apply_group_scale_aniso(new_xs, new_ys, start, n, center_x, center_y, 1.0, best)

    return new_xs, new_ys


@njit(cache=True)
def group_side_length_rotated(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_degs: np.ndarray,
    start_idx: int,
    n: int,
    center_x: float,
    center_y: float,
    angle_rad: float,
) -> float:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    angle_deg = angle_rad * 180.0 / math.pi

    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for i in range(n):
        idx = start_idx + i
        dx = all_xs[idx] - center_x
        dy = all_ys[idx] - center_y
        cx = center_x + dx * cos_a - dy * sin_a
        cy = center_y + dx * sin_a + dy * cos_a
        deg = (all_degs[idx] + angle_deg) % 360.0
        verts = get_tree_vertices(cx, cy, deg)
        x1, y1, x2, y2 = polygon_bounds(verts)
        if x1 < min_x:
            min_x = x1
        if y1 < min_y:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2

    return max(max_x - min_x, max_y - min_y)


@njit(cache=True)
def find_best_rotation_angle(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_degs: np.ndarray,
    start_idx: int,
    n: int,
    center_x: float,
    center_y: float,
    angles_rad: np.ndarray,
) -> tuple[float, float]:
    best_angle = 0.0
    best_side = math.inf

    for k in range(angles_rad.shape[0]):
        angle = angles_rad[k]
        side = group_side_length_rotated(
            all_xs, all_ys, all_degs, start_idx, n, center_x, center_y, angle
        )
        if side < best_side:
            best_side = side
            best_angle = angle

    return best_angle, best_side


@njit(cache=True)
def apply_group_rotation(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_degs: np.ndarray,
    start_idx: int,
    n: int,
    center_x: float,
    center_y: float,
    angle_rad: float,
) -> None:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    angle_deg = angle_rad * 180.0 / math.pi

    for i in range(n):
        idx = start_idx + i
        dx = all_xs[idx] - center_x
        dy = all_ys[idx] - center_y
        all_xs[idx] = center_x + dx * cos_a - dy * sin_a
        all_ys[idx] = center_y + dx * sin_a + dy * cos_a
        all_degs[idx] = (all_degs[idx] + angle_deg) % 360.0


def rotate_positions(
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    all_degs: np.ndarray,
    rotate_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enabled = bool(rotate_cfg.get("enabled", False))
    if not enabled:
        return all_xs, all_ys, all_degs

    n_min = int(rotate_cfg.get("n_min", 1))
    n_max = int(rotate_cfg.get("n_max", 200))
    center_mode = str(rotate_cfg.get("center", "bounds"))
    coarse_step_deg = float(rotate_cfg.get("coarse_step_deg", 5.0))
    fine_step_deg = float(rotate_cfg.get("fine_step_deg", 0.5))
    fine_window_deg = float(rotate_cfg.get("fine_window_deg", 4.0))
    max_angle_deg = float(rotate_cfg.get("max_angle_deg", 90.0))

    if coarse_step_deg <= 0.0:
        coarse_step_deg = 5.0
    if fine_step_deg < 0.0:
        fine_step_deg = 0.0
    if fine_window_deg < 0.0:
        fine_window_deg = 0.0
    if max_angle_deg <= 0.0:
        max_angle_deg = 90.0
    if max_angle_deg > 90.0:
        max_angle_deg = 90.0

    angles_coarse = np.deg2rad(
        np.arange(0.0, max_angle_deg + 1e-9, coarse_step_deg, dtype=np.float64)
    )

    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    group_start = np.zeros(201, dtype=np.int64)
    for n in range(1, 201):
        group_start[n] = group_start[n - 1] + (n - 1) if n > 1 else 0

    for n in range(n_min, min(n_max, 200) + 1):
        if n <= 1:
            continue
        start = int(group_start[n])
        if center_mode == "mean":
            center_x = float(new_xs[start : start + n].mean())
            center_y = float(new_ys[start : start + n].mean())
        else:
            center_x, center_y = compute_center_bounds(new_xs, new_ys, new_degs, start, n)

        best_angle, _ = find_best_rotation_angle(
            new_xs, new_ys, new_degs, start, n, center_x, center_y, angles_coarse
        )

        if fine_step_deg > 0.0 and fine_window_deg > 0.0:
            best_deg = best_angle * 180.0 / math.pi
            fine_start = max(0.0, best_deg - fine_window_deg)
            fine_end = min(max_angle_deg, best_deg + fine_window_deg)
            if fine_end - fine_start >= fine_step_deg:
                angles_fine = np.deg2rad(
                    np.arange(fine_start, fine_end + 1e-9, fine_step_deg, dtype=np.float64)
                )
                best_angle, _ = find_best_rotation_angle(
                    new_xs, new_ys, new_degs, start, n, center_x, center_y, angles_fine
                )

        if abs(best_angle) > 1e-12:
            apply_group_rotation(new_xs, new_ys, new_degs, start, n, center_x, center_y, best_angle)

    return new_xs, new_ys, new_degs


def calculate_total_score(all_xs: np.ndarray, all_ys: np.ndarray, all_degs: np.ndarray) -> float:
    total = 0.0
    idx = 0
    for n in range(1, 201):
        vertices = [
            get_tree_vertices(all_xs[idx + i], all_ys[idx + i], all_degs[idx + i]) for i in range(n)
        ]
        total += calculate_score_numba(vertices)
        idx += n
    return total


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Intensive SA Optimizer (exp007_intensive)")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")

    baseline_path = CONFIG["paths"]["baseline"]
    print(f"\nBaseline: {baseline_path}")

    baseline_xs, baseline_ys, baseline_degs = load_submission_data(
        baseline_path,
        CONFIG["tree_shape"],
        CONFIG.get("paths", {}).get("baseline_fallback"),
    )
    baseline_total = calculate_total_score(baseline_xs, baseline_ys, baseline_degs)
    print(f"Baseline total score: {baseline_total:.6f}")

    # Generate Grid Configurations
    grid_cfg = CONFIG["grid_search"]
    grid_configs = []

    n_seeds = len(CONFIG["initial_state"]["seeds"])
    for ncols in range(grid_cfg["col_min"], grid_cfg["col_max"]):
        for nrows in range(ncols, grid_cfg["row_max_limit"]):
            n_trees = n_seeds * ncols * nrows
            max_t = grid_cfg["max_trees"]

            if 20 <= n_trees <= max_t:
                if (ncols, nrows, False, False) not in grid_configs:
                    grid_configs.append((ncols, nrows, False, False))
                if n_seeds > 1 and n_trees + ncols <= max_t:
                    grid_configs.append((ncols, nrows, False, True))
                if n_seeds > 1 and n_trees + nrows <= max_t:
                    grid_configs.append((ncols, nrows, True, False))
                if n_seeds > 1 and n_trees + nrows + ncols <= max_t:
                    grid_configs.append((ncols, nrows, True, True))

    grid_configs = sorted(
        list(set(grid_configs)),
        key=lambda x: (n_seeds * x[0] * x[1] + (x[1] if x[2] else 0) + (x[0] if x[3] else 0)),
    )
    print(f"Generated {len(grid_configs)} grid configurations")

    # Prepare Tasks
    tasks = []
    init_state = CONFIG["initial_state"]
    seeds = init_state["seeds"]
    a_init = init_state["translation_a"]
    b_init = init_state["translation_b"]
    shear_x_init = float(init_state.get("shear_x", 0.0))
    shear_y_init = float(init_state.get("shear_y", 0.0))
    sa_params = CONFIG["sa_params"]
    multi_start = MULTI_START
    polish_params = build_polish_params(sa_params, POLISH)
    hill_params = build_hill_params(sa_params, HILL)

    n_seeds = len(seeds)
    for i, (ncols, nrows, append_x, append_y) in enumerate(grid_configs):
        n_trees = n_seeds * ncols * nrows + (nrows if append_x else 0) + (ncols if append_y else 0)
        if n_trees > 200:
            continue
        seed = sa_params["random_seed_base"] + i * 1000
        tasks.append(
            (
                ncols,
                nrows,
                append_x,
                append_y,
                seeds,
                a_init,
                b_init,
                shear_x_init,
                shear_y_init,
                sa_params,
                multi_start,
                polish_params,
                hill_params,
                seed,
            )
        )

    # Execute Parallel Optimization
    print(f"Running SA optimization on {len(tasks)} configurations...")
    num_workers = min(cpu_count(), len(tasks))
    t0 = time.time()
    progress = bool(CONFIG.get("progress", True))

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(optimize_grid_config, tasks),
            total=len(tasks),
            desc="Optimizing",
            disable=not progress,
        ):
            results.append(result)

    print(f"SA optimization completed in {time.time() - t0:.1f}s")

    # Process Results
    new_trees = {}
    improved_count = 0
    for n_trees, score, tree_data in results:
        idx = sum(range(1, n_trees))
        baseline_vertices = [
            get_tree_vertices(baseline_xs[idx + i], baseline_ys[idx + i], baseline_degs[idx + i])
            for i in range(n_trees)
        ]
        baseline_score = calculate_score_numba(baseline_vertices)

        if score < baseline_score:
            new_trees[n_trees] = tree_data
            if baseline_score - score > 1e-6:
                improved_count += 1
                print(f"  n={n_trees}: {score:.6f} (baseline: {baseline_score:.6f})")

    # Merge with Baseline
    print("Merging with baseline...")
    merged_xs = baseline_xs.copy()
    merged_ys = baseline_ys.copy()
    merged_degs = baseline_degs.copy()

    for n_trees, tree_data in new_trees.items():
        idx = sum(range(1, n_trees))
        for i in range(n_trees):
            merged_xs[idx + i] = tree_data[i][0]
            merged_ys[idx + i] = tree_data[i][1]
            merged_degs[idx + i] = tree_data[i][2]

    # Deletion Cascade
    print("Applying tree deletion cascade...")
    final_xs, final_ys, final_degs, _ = deletion_cascade_numba(merged_xs, merged_ys, merged_degs)

    # Shrink step
    shrink_cfg = SHRINK
    if shrink_cfg.get("enabled", False):
        print("Applying shrink optimization...")
        final_xs, final_ys = shrink_positions(final_xs, final_ys, final_degs, shrink_cfg)

    shrink_aniso_cfg = SHRINK_ANISO
    if shrink_aniso_cfg.get("enabled", False):
        print("Applying anisotropic shrink optimization...")
        final_xs, final_ys = shrink_positions_aniso(
            final_xs, final_ys, final_degs, shrink_aniso_cfg
        )

    rotate_cfg = ROTATE
    if rotate_cfg.get("enabled", False):
        print("Applying rotation optimization...")
        final_xs, final_ys, final_degs = rotate_positions(
            final_xs, final_ys, final_degs, rotate_cfg
        )

    # Second pass of shrink after rotation
    if shrink_aniso_cfg.get("enabled", False):
        print("Applying second pass of anisotropic shrink after rotation...")
        final_xs, final_ys = shrink_positions_aniso(
            final_xs, final_ys, final_degs, shrink_aniso_cfg
        )

    final_score = calculate_total_score(final_xs, final_ys, final_degs)
    print("=" * 80)
    print(f"  Baseline total:      {baseline_total:.6f}")
    print(f"  After cascade:       {final_score:.6f}")
    print(f"  Total improvement:   {baseline_total - final_score:+.6f}")
    print("=" * 80)

    # Save Results
    if final_score < baseline_total:
        out_path = CONFIG["paths"]["output"]
        save_submission(out_path, final_xs, final_ys, final_degs)
        print(f"Saved to {out_path}")

        script_path = CONFIG["paths"].get("overlap_script", "")
        if script_path and os.path.exists(script_path):
            cmd = f"python {script_path} {baseline_path} {out_path}"
            os.system(cmd)
    else:
        print("No improvement - keeping baseline")
