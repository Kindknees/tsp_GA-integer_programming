import numpy as np, random, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from initial import new_initial
from utils   import tour_length2
from your_ga_module import two_opt2
from localsearch import local_search_full,depot_insert_best    # 你寫的 OR-opt2/3
import time
# ---------- 參數 ----------
CSV_PATH       = "data/xy48.csv"
K_CLUSTER      = 6
POP_SIZE       = 100
A_RATIO, B_RATIO = 0.4, 0.6
# SEED           = 42
LOCAL_P        = 0.10     # two-opt 機率
MUTATE_P       = 0.15     # mutate 機率
P_TOUR         = 0.8     # p-binary tournament
MAX_GEN        = 50
OPTIMAL        = 33551    # ATT48 最優

# ---------- 讀距離矩陣 ----------
D = pd.read_csv("data/USA_distance.csv", header=None).values
# class Distance():
#     def __init__(self, filename):
#         data = pd.read_csv(filename, header=None)
#         self.x = data[0].values  # 第一欄作為 x 座標
#         self.y = data[1].values  # 第二欄作為 y 座標
#     def get_distance(self):
#         # 計算距離矩陣
#         n = len(self.x)
#         distance = np.zeros((n, n))
#         for i in range(n):
#             for j in range(n):
#                 distance[i][j] = np.round(np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2))
#         # # 四捨五入
#         # distance = np.round(distance)
#         # 轉換為 DataFrame
#         distance = pd.DataFrame(distance)
#         distance.index = [f'City {i+1}' for i in range(n)]
#         distance.columns = [f'City {i+1}' for i in range(n)]
#         return distance
# d= Distance('data/xy48.csv')
# distance = d.get_distance()
# distance.head()
# D = distance.to_numpy() #轉成numpy array，我會比較開心


N = D.shape[0]            # =48

# ---------- 初始化 ----------
def init_population():
    return new_initial(
        CSV_PATH, k=K_CLUSTER, pop_size=POP_SIZE,
        a_ratio=A_RATIO, b_ratio=B_RATIO, seed=None
    )

# ---------- 適應度 ----------
def fitness(route):               # 越大越好
    return -tour_length2(route, D) # = −distance

# ---------- selection：p-binary tournament ----------
def selection(pop, fit, p, rng):
    n = len(pop)
    sel = []
    for _ in range(n):
        i, j = rng.choice(n, 2, replace=False)
        better = i if fit[i] >= fit[j] else j
        worse  = j if better == i else i
        winner = better if rng.random() < p else worse
        sel.append(pop[winner])
    return sel

# ---------- crossover：ERX ----------
def crossover(p1, p2, rng=None):
    rng  = rng or np.random.default_rng()
    size = len(p1)
    # 1) 鄰接表
    adj = {v: set() for v in p1}
    for parent in (p1, p2):
        for i in range(size):
            a = parent[i]
            pred  = parent[(i-1) % size]    # 前驅（尾→頭）
            succ  = parent[(i+1) % size]    # 後繼（頭→尾）
            adj[a].update({pred, succ})
    # 2) 組裝子代
    current = rng.choice(p1)
    child   = [current]
    while len(child) < size:
        for s in adj.values():
            s.discard(current)
        if adj[current]:
            current = min(adj[current], key=lambda c: (len(adj[c]), rng.random()))
        else:
            remaining = [c for c in p1 if c not in child]
            current   = rng.choice(remaining)
        child.append(current)
    return child
# ---------- mutate (六算子) ----------
def mutate(route, rng):
    r = rng.random(); n = len(route)
    if r < 1/6:
        i,j = sorted(rng.choice(n, 2, replace=False))
        seg = route[i:j+1]
        rem = route[:i] + route[j+1:]
        k   = rng.integers(0, len(rem)+1)
        route = rem[:k] + seg + rem[k:]
    elif r < 2/6:
        i = rng.integers(n)
        city = route.pop(i)
        k = rng.integers(len(route)+1)
        route.insert(k, city)
    elif r < 3/6:
        i,j = sorted(rng.choice(n, 2, replace=False))
        seg = route[i:j+1][::-1]
        rem = route[:i] + route[j+1:]
        k   = rng.integers(len(rem)+1)
        route = rem[:k] + seg + rem[k:]
    elif r < 4/6:
        i,j = rng.choice(n, 2, replace=False)
        route[i], route[j] = route[j], route[i]
    elif r < 5/6:
        i,j = sorted(rng.choice(n, 2, replace=False))
        if j == i+1: j = (j+1)%n
        a,b = route[i], route[(i+1)%n]
        c,d = route[j], route[(j+1)%n]
        if D[a][b]+D[c][d] > D[a][c]+D[b][d]:
            route[i+1:j+1] = reversed(route[i+1:j+1])
    else:
        # 6. 把最佳插入點 k 插進 a,b 之間
        idx  = rng.integers(n)
        city = route.pop(idx)          # 先拔掉自己
        # 拿到左右鄰居 a,b
        a = route[idx-1]
        b = route[idx % len(route)]
        # 在所有剩餘城市找 k，使 a→k→b 最短
        k = min(route, key=lambda x: D[a][x] + D[x][b])
        j = route.index(k)
        route.insert(j+1, city)        # 把 city 插到 k 後
    return route

# ---------- GA 主程式 ----------
def GA_TSP():
    rng = np.random.default_rng()

    # 1) 初始化：產生初始族群
    population = init_population()  # 回傳 100 條路徑
    # 1-a) 直接計算 fitness = -tour_length2
    pop_fit = [ -tour_length2(route, D) for route in population ]

    # 找出一開始的 best
    best_fit   = max(pop_fit)
    best_route = population[ pop_fit.index(best_fit) ].copy()

    best_outputs = []
    mean_outputs = []

    for gen in range(1, MAX_GEN + 1):
        # 紀錄 mean/best fitness
        mean_outputs.append(np.mean(pop_fit))
        best_outputs.append(best_fit)

        # 2) 選擇：p‐binary tournament（用 fitness 做比較）
        mating_pool = selection(population, pop_fit, P_TOUR, rng)
        #   ※ 下面 selection 就改成比較 pop_fit 而非重算 tour_length2

        # 3) 交配、產生 offspring
        offspring     = []
        offspring_fit = []

        rng.shuffle(mating_pool)
        for i in range(0, POP_SIZE, 2):
            p1 = mating_pool[i]
            p2 = mating_pool[i+1]
            child = crossover(p1, p2, rng)

            # 交配後立刻計算 fitness
            f = -tour_length2(child, D)
            offspring.append(child)
            offspring_fit.append(f)

        # 4) 突變／two-opt：只要做了路徑改動就重新計算 fitness
        for idx in range(len(offspring)):
            route = offspring[idx]

            # 突變
            if rng.random() < MUTATE_P:
                route = mutate(route, rng)
                offspring[idx] = route
                offspring_fit[idx] = -tour_length2(route, D)

            # 2-opt
            if rng.random() < LOCAL_P:
                route = two_opt2(route, D)
                offspring[idx] = route
                offspring_fit[idx] = -tour_length2(route, D)

        # 5) 合併：父母 + 子代
        combined_pop  = population   + offspring
        combined_fit  = pop_fit      + offspring_fit

        # 6) 排序：直接用 fitness 由大排到小，取前 POP_SIZE
        zipped = list(zip(combined_fit, combined_pop))
        # zipped[i] = (fitness_i, route_i)
        # sort by fitness descending
        zipped.sort(key=lambda x: x[0], reverse=True)

        # 7) 保留前 100 條
        top = zipped[:POP_SIZE]
        pop_fit    = [f for (f, r) in top]
        population = [r for (f, r) in top]

        # 8) 更新 best
        if pop_fit[0] > best_fit:
            best_fit   = pop_fit[0]
            best_route = population[0].copy()
        if -best_fit <= OPTIMAL:  # cost = -best_fit
            break

    # 回傳最終 best route & cost
    return best_route, -best_fit, best_outputs, mean_outputs


if __name__ == "__main__":
    NUM_RUN = 100         # ← 要跑幾次
    results = []

    
    total_time=0
    for run in range(NUM_RUN):
        # 計時開始depot_insert_best
        t0 = time.perf_counter()
        route, cost, best_outputs, mean_outputs = GA_TSP()      # ← 不用改 GA_TSP 內容
        total_time = total_time + time.perf_counter() - t0
        results.append(cost)
        print(f"Run {run:3d} | best = {cost:.0f}")

    
    print(f"\n★ GA 總執行時間：{total_time:.2f} 秒")
    print(f"★ 平均每次執行時間：{(total_time/NUM_RUN):.4f} 秒")

    # 以下為原本的統計與繪圖程式…
    res = np.array(results)


    # # 畫圖 (new)
    # import matplotlib.pyplot
    # matplotlib.pyplot.plot(best_outputs)
    # matplotlib.pyplot.plot(mean_outputs)
    # matplotlib.pyplot.xlabel("Iteration")
    # matplotlib.pyplot.ylabel("Fitness")
    # matplotlib.pyplot.show()
    # ---- 統計 ----
    res = np.array(results)
    mean, median, std = res.mean(), np.median(res), res.std(ddof=1)
    q1, q3            = np.percentile(res, [25, 75])
    opt_hits          = np.sum(res <= 33551)
    below_thr         = np.sum(res <= 34148)

    print("\n=== 100 次統計 ===")
    print(f"平均    : {mean:.0f}")
    print(f"中位數  : {median:.0f}")
    print(f"標準差  : {std:.0f}")
    print(f"Q1 / Q3 : {q1:.0f}  /  {q3:.0f}")
    print(f"最小 / 最大 : {res.min():.0f}  /  {res.max():.0f}")
    print(f"命中最佳 (<=33551) 次數 : {opt_hits}/{NUM_RUN}")
    print(f"< 34148 次數           : {below_thr}/{NUM_RUN}")

    # ---- box plot (with mean) ----
    plt.figure(figsize=(7, 4))

    plt.boxplot(
        res,
        showfliers=False,
        showmeans=True,                 # ← 顯示平均
        meanprops=dict(marker="D",      #   D = 菱形
                    markerfacecolor="blue",
                    markersize=6),
        boxprops=dict(color="steelblue", linewidth=2),
        medianprops=dict(color="red", linewidth=2)
    )

    plt.ylabel("Best distance", fontsize=11)
    plt.title("GA best distance distribution (100 runs)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # # 畫路徑（按城市序號折線，僅示意）
    # font = FontProperties(fname="NotoSansTC-Regular.otf")
        
    # # --- 讀取座標 ----------------------------------------------------
    # csv_path = "data/xy48.csv"   # ⇠ 路徑依你的專案結構調整
    # coords   = pd.read_csv(csv_path, header=None, names=["x", "y"])


    # # 把 depot(0) 加到首尾形成完整巡迴
    # full_route = [0] + route + [0]

    # # 把路徑轉成座標序列
    # points = coords.iloc[full_route]

    # # --- 畫圖 --------------------------------------------------------
    # plt.figure(figsize=(8, 6))
    # plt.plot(points["x"], points["y"], "-o", markersize=4)
    # plt.scatter(points["x"].iloc[0], points["y"].iloc[0], c="red", s=80, label="Depot (0)")
    # plt.title("ATT48 Route in Geographic Coordinates")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()
