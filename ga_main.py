import numpy as np, random, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from initial import fcm_initial_tours
from utils   import tour_length, two_opt,three_opt
from localsearch import local_search_full    # 你寫的 OR-opt2/3

# ---------- 參數 ----------
CSV_PATH       = "data/xy48.csv"
K_CLUSTER      = 10
POP_SIZE       = 100
A_RATIO, B_RATIO = 0.4, 0.6
# SEED           = 42
LOCAL_P        = 0.30     # two-opt 機率
MUTATE_P       = 0.16     # mutate 機率
P_TOUR         = 0.85     # p-binary tournament
MAX_GEN        = 750
OPTIMAL        = 33551    # ATT48 最優

# ---------- 讀距離矩陣 ----------
D = pd.read_csv("data/USA_distance.csv", header=None).values
N = D.shape[0]            # =48

# ---------- 初始化 ----------
def init_population():
    return fcm_initial_tours(
        CSV_PATH, k=K_CLUSTER, pop_size=POP_SIZE,
        a_ratio=A_RATIO, b_ratio=B_RATIO, seed=None
    )

# ---------- 適應度 ----------
def fitness(route):               # 越大越好
    return -tour_length(route, D) # = −distance

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
            adj[a].update({parent[i-1], parent[(i+1)%size]})
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
        idx = rng.integers(n)
        city = route[idx]
        nearest = min(range(1,N), key=lambda x: np.inf if x==city else D[city][x])
        j = route.index(nearest)
        if abs(j-idx) not in (1, n-1):
            route.pop(idx)
            j = route.index(nearest)
            route.insert(j+1, city)
    return route

# ---------- GA 主程式 ----------
def GA_TSP():
    rng        = np.random.default_rng()
    population = init_population()
    best_route = min(population, key=lambda t: tour_length(t, D))
    best_cost  = tour_length(best_route, D)
    best_outputs = []                           # 用此變數來紀錄每一個迴圈的最佳解 (new)
            # 存下初始群體的最佳解 (new)

    mean_outputs = []                           # 用此變數來紀錄每一個迴圈的平均解 (new)
    
    for gen in range(1, MAX_GEN+1):
        # 選擇
        fit   = [fitness(t) for t in population]
        mean_outputs.append(np.average(fit))        # 存下初始群體的最佳解 (new)
        best_outputs.append(np.max(fit))
        mating_pool = selection(population, fit, P_TOUR, rng)
        # 交配
        offspring = []
        rng.shuffle(mating_pool)
        for i in range(0, POP_SIZE, 2):
            c1 = crossover(mating_pool[i], mating_pool[i+1], rng)
            c2 = crossover(mating_pool[i+1], mating_pool[i], rng)
            offspring.extend([c1, c2])

        # two-opt (30 %)
        for k in range(len(offspring)):
            if  rng.random() < LOCAL_P:
                # if rng.random() <  * LOCAL_P:
                offspring[k] = two_opt(offspring[k],D)
                # else:
                #     offspring[k] = three_opt(offspring[k],D)

        # 突變 (5 %)
        for k in range(len(offspring)):
            if rng.random() < MUTATE_P:
                offspring[k] = mutate(offspring[k], rng)
                offspring[k] = mutate(offspring[k], rng)



        # 生存者篩選 (父母+子女取前 POP_SIZE)
        population = sorted(population+offspring, key=lambda t: tour_length(t,D))[:POP_SIZE]

        # 更新最優
        if tour_length(population[0], D) < best_cost:
            best_route, best_cost = population[0].copy(), tour_length(population[0], D)
        if best_cost <= OPTIMAL:
            break

    # 最終 OR-opt2/3 強化
    best_route = local_search_full(best_route, D)
    # best_route= three_opt(best_route, D)  # 可選擇性使用 three_opt
    best_cost  = tour_length(best_route, D)
    return best_route, best_cost,best_outputs, mean_outputs

# ---------- 執行 ----------
if __name__ == "__main__":
    NUM_RUN = 100         # ← 要跑幾次
    results = []

    for run in range(NUM_RUN):
        # # 每回合換一顆 seed，避免族群重覆
        # SEED = 42 + run
        route, cost,best_outputs,mean_outputs = GA_TSP()      # ← 不用改 GA_TSP 內容
        results.append(cost)
        print(f"Run {run:3d} | best = {cost:.0f}")


    # # 畫圖 (new)
    # import matplotlib.pyplot
    # matplotlib.pyplot.plot(best_outputs)
    # matplotlib.pyplot.plot(mean_outputs)
    # matplotlib.pyplot.xlabel("Iteration")
    # matplotlib.pyplot.ylabel("Fitness")
    #matplotlib.pyplot.show()
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

    # route, cost = GA_TSP()
    # print("Best length :", cost)
    # print("Best route  :", [0]+route+[0])

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