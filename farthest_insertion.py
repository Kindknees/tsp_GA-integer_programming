
import numpy as np
import pandas as pd

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

def farthest_insertion(D):
    """
    Farthest Insertion heuristic for TSP.
    D: n×n numpy array of distances (包含城市0…n-1)
    回傳一條不含0的染色體列表（城市編號1..n-1）。
    """
    n = D.shape[0]
    unvis = set(range(1, n))
    # 1. 初始：找最遠的一對 (不含0)
    a, b = max(
        ((i,j) for i in unvis for j in unvis if i<j),
        key=lambda p: D[p[0], p[1]]
    )
    # 我們把0也插進去作起點，種子路線為 [0, a, b]
    tour = [0, a, b]
    # 從未訪集合移除 a, b
    unvis -= {a, b}

    # 2. 重複直到 unvis 空
    while unvis:
        # (a) 找出「最遠城市」：對每個 c，計算 min(D[c][v] for v in tour)，取最大的 c
        far_c = max(unvis, key=lambda c: min(D[c, v] for v in tour))
        
        # (b) 找這個 far_c 插入 tour 的最佳位置
        best_inc = float('inf')
        best_pos = None
        for i in range(len(tour)):
            j = tour[(i+1) % len(tour)]
            inc = D[tour[i], far_c] + D[far_c, j] - D[tour[i], j]
            if inc < best_inc:
                best_inc, best_pos = inc, i+1
        
        # (c) 插入並從未訪中移除
        tour.insert(best_pos, far_c)
        unvis.remove(far_c)

    # 回傳去掉開頭0的部分，作為不含0的染色體
    return tour[1:]
# print("Farthest Insertion:", farthest_insertion(D))