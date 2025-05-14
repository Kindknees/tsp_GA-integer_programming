# clustering.py
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from farthest_insertion import farthest_insertion
def load_distance(csv_path):
    data = pd.read_csv(csv_path, header=None).values
    return data[:,0], data[:,1]

def perturb_tour(tour, num_swaps=3, rng=None):
    """
    對給定的 tour 做隨機交換擾動。
    num_swaps: 交換次數
    rng: numpy 隨機生成器
    """
    if rng is None:
        rng = np.random.default_rng()
    pert = tour.copy()
    n = len(pert)
    for _ in range(num_swaps):
        i, j = rng.integers(0, n, size=2)
        pert[i], pert[j] = pert[j], pert[i]
    return pert

def fcm_initial_tours(csv_path, k=10, pop_size=100,a_ratio=0.45,b_ratio=0.45, seed=None):
    """回傳 pop_size 條路徑 (list[int])，只用 FCM 隨機法"""
    rng = np.random.default_rng(seed)
    xs, ys = load_distance(csv_path)
    X = np.vstack([xs, ys]).T
    cntr, u, *_ = fuzz.cluster.cmeans(X.T, c=k, m=2.5, error=1e-5,
                                      maxiter=1000)
    c, n = u.shape
    tours = []
    # 建距離矩陣 D
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
     # FCM 解數量
    na = int(pop_size * a_ratio)

    # 純隨機解數量
    nb = int(pop_size * b_ratio)

    # FI 解數量
    nc = pop_size - na - nb
   
    for _ in range(na):
        # 1) 隨機根據 membership 指派群
        clusters = {i: [] for i in range(c)}
        for j in range(1, n):
            lab = rng.choice(c, p=u[:, j]/u[:, j].sum())
            clusters[lab].append(j)
        # 2) 隨機決定群順序 (權重 = 該群最大 membership)
        weights = np.array([u[i,1:].max() for i in range(c)])
        order = rng.choice(c, size=c, replace=False,
                           p=weights/weights.sum())
        # 3) 產生路徑
        tour = []
        for lab in order:
            rng.shuffle(clusters[lab])
            tour.extend(clusters[lab])
        # 對 FCM 生成的路徑做擾動
        perturbed = perturb_tour(tour, num_swaps=5, rng=rng)

        
        tours.append(perturbed)

    # —— 方法 B：全域純隨機 ——
    for _ in range(nb):
        perm = list(range(1, n))
        rng.shuffle(perm)
        tours.append(perm)
    
    # # —— 方法 C：Farthest Insertion ——
    # for _ in range(nc):
    #     r=rng.uniform(0,1)
    #     base_tour = farthest_insertion(D)
    #     k= max(int(10*r),5)
    #     # 對 FI 解做小規模交換擾動
    #     perturbed = perturb_tour(base_tour, num_swaps=k, rng=rng)
    #     tours.append(perturbed)
    #     # tours.append(farthest_insertion(D))


    return tours
