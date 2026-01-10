"""すべてのグループの重なりをチェックし、修正する"""

import math
import pandas as pd
import numpy as np

TRUNK_W = 0.15
BASE_W = 0.7
MID_W = 0.4
TOP_W = 0.25
TIP_Y = 0.8
TIER_1_Y = 0.5
TIER_2_Y = 0.25
BASE_Y = 0.0
TRUNK_BOTTOM_Y = -0.2


def get_tree_vertices(cx, cy, angle_deg):
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    pts = [
        [0.0, TIP_Y],
        [TOP_W / 2.0, TIER_1_Y], [TOP_W / 4.0, TIER_1_Y],
        [MID_W / 2.0, TIER_2_Y], [MID_W / 4.0, TIER_2_Y],
        [BASE_W / 2.0, BASE_Y], [TRUNK_W / 2.0, BASE_Y],
        [TRUNK_W / 2.0, TRUNK_BOTTOM_Y], [-TRUNK_W / 2.0, TRUNK_BOTTOM_Y],
        [-TRUNK_W / 2.0, BASE_Y], [-BASE_W / 2.0, BASE_Y],
        [-MID_W / 4.0, TIER_2_Y], [-MID_W / 2.0, TIER_2_Y],
        [-TOP_W / 4.0, TIER_1_Y], [-TOP_W / 2.0, TIER_1_Y],
    ]
    
    vertices = []
    for px, py in pts:
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        vertices.append([rx + cx, ry + cy])
    return np.array(vertices)


def point_in_polygon(px, py, vertices):
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def segments_intersect(p1, p2, p3, p4):
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-10:
        return False
    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / det
    u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / det
    return 0.0 < t < 1.0 and 0.0 < u < 1.0  # 厳密な交差のみ


def polygons_overlap(verts1, verts2, margin=0.001):
    """マージンを考慮した重なりチェック"""
    # Bounding box check with margin
    min_x1, max_x1 = min(v[0] for v in verts1) - margin, max(v[0] for v in verts1) + margin
    min_y1, max_y1 = min(v[1] for v in verts1) - margin, max(v[1] for v in verts1) + margin
    min_x2, max_x2 = min(v[0] for v in verts2) - margin, max(v[0] for v in verts2) + margin
    min_y2, max_y2 = min(v[1] for v in verts2) - margin, max(v[1] for v in verts2) + margin
    
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False
    
    # Point in polygon
    for v in verts1:
        if point_in_polygon(v[0], v[1], verts2):
            return True
    for v in verts2:
        if point_in_polygon(v[0], v[1], verts1):
            return True
    
    # Edge intersection
    n1, n2 = len(verts1), len(verts2)
    for i in range(n1):
        j = (i + 1) % n1
        for k in range(n2):
            m = (k + 1) % n2
            if segments_intersect(verts1[i], verts1[j], verts2[k], verts2[m]):
                return True
    return False


def check_group_overlap(xs, ys, degs):
    """グループ内の重なりをチェック"""
    n = len(xs)
    vertices = [get_tree_vertices(xs[i], ys[i], degs[i]) for i in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(vertices[i], vertices[j]):
                return True, (i, j)
    return False, None


def generate_safe_placement(n):
    """安全な配置を生成（広いグリッド）"""
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    spacing = 1.5  # 十分広い間隔
    
    xs = []
    ys = []
    degs = []
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            xs.append(c * spacing)
            ys.append(r * spacing)
            degs.append(0.0)
            idx += 1
    
    # 中心を原点に
    xs = np.array(xs)
    ys = np.array(ys)
    xs -= np.mean(xs)
    ys -= np.mean(ys)
    
    return xs, ys, np.array(degs)


def main():
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "submissions/baseline.csv"
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Checking all groups for overlaps...")
    
    problem_groups = []
    
    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")
        
        xs, ys, degs = [], [], []
        for _, row in group.iterrows():
            x = float(row["x"][1:]) if isinstance(row["x"], str) else float(row["x"])
            y = float(row["y"][1:]) if isinstance(row["y"], str) else float(row["y"])
            deg = float(row["deg"][1:]) if isinstance(row["deg"], str) else float(row["deg"])
            xs.append(x)
            ys.append(y)
            degs.append(deg)
        
        xs, ys, degs = np.array(xs), np.array(ys), np.array(degs)
        
        has_overlap, pair = check_group_overlap(xs, ys, degs)
        if has_overlap:
            problem_groups.append(n)
            print(f"  Group {n}: OVERLAP between trees {pair[0]} and {pair[1]}")
    
    if not problem_groups:
        print("No overlaps found in baseline!")
        df.to_csv("submissions/submission_verified.csv", index=False)
        print("Saved baseline as submissions/submission_verified.csv")
        return
    
    print(f"\nFound {len(problem_groups)} groups with overlaps: {problem_groups}")
    print("Fixing...")
    
    # 問題のあるグループを修正
    for n in problem_groups:
        prefix = f"{n:03d}_"
        
        # 安全な配置を生成
        safe_xs, safe_ys, safe_degs = generate_safe_placement(n)
        
        # DataFrameを更新
        group_ids = [f"{n:03d}_{i}" for i in range(n)]
        for i, tree_id in enumerate(group_ids):
            idx = df[df["id"] == tree_id].index[0]
            df.at[idx, "x"] = f"s{safe_xs[i]}"
            df.at[idx, "y"] = f"s{safe_ys[i]}"
            df.at[idx, "deg"] = f"s{safe_degs[i]}"
        
        print(f"  Fixed Group {n}")
    
    # 再度チェック
    print("\nVerifying fixes...")
    all_ok = True
    for n in problem_groups:
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")
        
        xs, ys, degs = [], [], []
        for _, row in group.iterrows():
            x = float(row["x"][1:]) if isinstance(row["x"], str) else float(row["x"])
            y = float(row["y"][1:]) if isinstance(row["y"], str) else float(row["y"])
            deg = float(row["deg"][1:]) if isinstance(row["deg"], str) else float(row["deg"])
            xs.append(x)
            ys.append(y)
            degs.append(deg)
        
        xs, ys, degs = np.array(xs), np.array(ys), np.array(degs)
        
        has_overlap, pair = check_group_overlap(xs, ys, degs)
        if has_overlap:
            print(f"  Group {n}: STILL HAS OVERLAP!")
            all_ok = False
        else:
            print(f"  Group {n}: OK")
    
    if all_ok:
        df.to_csv("submissions/submission_verified.csv", index=False)
        print("\nSaved to submissions/submission_verified.csv")
    else:
        print("\nSome groups still have overlaps!")


if __name__ == "__main__":
    main()

