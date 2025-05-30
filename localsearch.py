import pandas as pd
import numpy as np
from utils import tour_length          # ← 跟主程式同一支長度函式
# def load_distance(csv_path):
#     data = pd.read_csv(csv_path, header=None).values
#     return data[:,0], data[:,1]

# def build_D(csv_path):
#     xs, ys = load_distance(csv_path)
#     m = len(xs)
#     D = np.zeros((m, m))
#     for i in range(m):
#         for j in range(m):
#             D[i,j] = np.hypot(xs[i]-xs[j], ys[i]-ys[j])
#     return D



# D = build_D("C:/Users/eric3/OneDrive/桌面/tsp_pso/data/xy48.csv")
# localsearch.py

def _or_once(route, k, D):
    n   = len(route)
    # 一開始也把 route 轉成整數 numpy array
    route = np.array(route, dtype=int)
    base  = tour_length(route, D)
    best_gain = 0.0
    best      = None

    for i in range(n - k + 1):
        seg  = route[i : i+k]
        rest = np.concatenate((route[:i], route[i+k:]))
        for j in range(len(rest) + 1):
            cand = np.concatenate((rest[:j], seg, rest[j:]))
            # 確保 cand 是整數
            cand = cand.astype(int)
            gain = base - tour_length(cand, D)
            if gain > best_gain:
                best_gain, best = gain, cand.copy()

    if best is not None:
        return best.tolist(), best_gain
    else:
        # 返回的 route 也最好是 list
        return route.tolist(), 0.0

def local_search_full(route, D):
    # 不要傳 numpy.float64 進來，保證都是純整數 list
    cur = list(map(int, route))
    while True:
        for k in (1, 2, 3):
            nxt, gain = _or_once(cur, k, D)
            if gain > 0:
                cur = nxt   # 已經是 list[int]
                break
        else:
            return cur
# print("localsearch.py",local_search_full( [15, 21, 2, 33, 40, 28, 1, 41, 25, 3, 34, 44, 9, 23, 4, 47, 38, 31, 20, 46, 12, 24, 13, 22, 10, 11, 39, 14, 45, 32, 19, 29, 42, 16, 26, 18, 36, 5, 27, 35, 6, 17, 43, 30, 37, 8, 7],D))