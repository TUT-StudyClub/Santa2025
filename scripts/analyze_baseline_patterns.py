"""ベースラインの配置パターンを分析"""

import math
import numpy as np
import pandas as pd


def load_submission_data(filepath: str):
    df = pd.read_csv(filepath)
    groups = {}
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
        groups[n] = (np.array(xs), np.array(ys), np.array(degs))
    return groups


def analyze_group(n: int, xs: np.ndarray, ys: np.ndarray, degs: np.ndarray):
    """グループの配置パターンを分析"""
    if n == 1:
        return {"n": 1, "pattern": "single"}
    
    # 角度の分布
    unique_angles = np.unique(np.round(degs, 1))
    angle_counts = {a: np.sum(np.abs(degs - a) < 1.0) for a in unique_angles}
    
    # 位置の分布
    x_range = np.max(xs) - np.min(xs)
    y_range = np.max(ys) - np.min(ys)
    
    # 木同士の距離
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            distances.append(d)
    
    min_dist = np.min(distances) if distances else 0
    avg_dist = np.mean(distances) if distances else 0
    
    # グリッドパターンかチェック
    x_unique = np.unique(np.round(xs, 2))
    y_unique = np.unique(np.round(ys, 2))
    is_grid_like = len(x_unique) * len(y_unique) >= n * 0.8
    
    return {
        "n": n,
        "x_range": x_range,
        "y_range": y_range,
        "aspect_ratio": x_range / y_range if y_range > 0 else 0,
        "unique_angles": len(unique_angles),
        "dominant_angle": max(angle_counts, key=angle_counts.get),
        "min_distance": min_dist,
        "avg_distance": avg_dist,
        "x_levels": len(x_unique),
        "y_levels": len(y_unique),
        "is_grid_like": is_grid_like,
    }


if __name__ == "__main__":
    print("Loading baseline...")
    groups = load_submission_data("submissions/baseline.csv")
    
    print("\n" + "=" * 100)
    print("Baseline Pattern Analysis")
    print("=" * 100)
    
    print(f"\n{'n':>4} | {'X Range':>8} | {'Y Range':>8} | {'Aspect':>7} | {'Angles':>7} | {'Dom Ang':>8} | {'Min Dist':>8} | {'Grid?':>6}")
    print("-" * 100)
    
    for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 200]:
        xs, ys, degs = groups[n]
        analysis = analyze_group(n, xs, ys, degs)
        
        print(f"{n:>4} | {analysis['x_range']:>8.4f} | {analysis['y_range']:>8.4f} | "
              f"{analysis['aspect_ratio']:>7.3f} | {analysis['unique_angles']:>7} | "
              f"{analysis['dominant_angle']:>8.1f} | {analysis['min_distance']:>8.4f} | "
              f"{'Yes' if analysis['is_grid_like'] else 'No':>6}")
    
    # 小さいグループの詳細
    print("\n" + "=" * 100)
    print("Small Groups Detail (n=2 to 10)")
    print("=" * 100)
    
    for n in range(2, 11):
        xs, ys, degs = groups[n]
        print(f"\nGroup {n}:")
        print(f"  Positions: {list(zip(np.round(xs, 4), np.round(ys, 4)))}")
        print(f"  Angles: {list(np.round(degs, 2))}")
        
        # 木の配置を視覚化（簡易）
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        print(f"  Bounding box: ({x_min:.4f}, {y_min:.4f}) to ({x_max:.4f}, {y_max:.4f})")


