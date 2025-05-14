import numpy as np
import pandas as pd
from farthest_insertion import farthest_insertion
from utils import tour_length
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def load_distance(csv_path):
    df = pd.read_csv(csv_path, header=None)
    xs, ys = df[0].values, df[1].values
    return xs, ys


def build_D(xs, ys):
    n = len(xs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
    return D


def or_opt(route, D, k):
    """
    Perform one pass of OR-opt-k: remove any segment of length k and reinsert.
    Return improved route and gain; if no improvement, returns original and gain=0.
    """
    n = len(route)
    best_gain = 0.0
    best_route = list(route)
    base = tour_length(route, D)
    arr = np.array(route, dtype=int)
    for i in range(n - k + 1):
        seg = arr[i:i+k]
        rest = np.concatenate((arr[:i], arr[i+k:]))
        for j in range(len(rest) + 1):
            cand = np.concatenate((rest[:j], seg, rest[j:]))
            cand_list = cand.astype(int).tolist()
            gain = base - tour_length(cand_list, D)
            if gain > best_gain:
                best_gain = gain
                best_route = cand_list
    return best_route, best_gain


def local_search(route, D, max_k=5):
    """
    Apply OR-opt-k for k=1..max_k until no improvement.
    """
    current = list(route)
    while True:
        improved = False
        for k in range(1, max_k+1):
            new_route, gain = or_opt(current, D, k)
            if gain > 0:
                current = new_route
                improved = True
                break
        if not improved:
            return current


if __name__ == "__main__":
    CSV = "data/xy48.csv"
    xs, ys = load_distance(CSV)
    D = build_D(xs, ys)

    # 初始解：farthest insertion (不含 city 0)
    init = farthest_insertion(D)
    length_init = tour_length(init, D)
    print(f"Initial farthest insertion length: {length_init:.2f}")

    # OR-opt 本地搜尋
    refined = local_search(init, D, max_k=3)
    length_refined = tour_length(refined, D)
    print(f"Refined length after OR-opt1..5: {length_refined:.2f}")
    print("Route:", [0] + refined + [0])

    # 視覺化結果設定 (使用指定字型)
    font = FontProperties(fname="NotoSansTC-Regular.otf")
    # 根據 final_route 取得對應的座標
    x_route = [ xs[i] for i in [0] + refined + [0] ]
    y_route = [ ys[i] for i in [0] + refined + [0] ]
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_route, y_route, marker='o', linestyle='-', color='blue')
    plt.title("TSP 路徑視覺化", fontproperties=font)
    plt.xlabel("X 坐標", fontproperties=font)
    plt.ylabel("Y 坐標", fontproperties=font)
    plt.grid(True)
    plt.show()