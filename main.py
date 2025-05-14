# main.py
import numpy as np
from utils import tour_length
from initial import fcm_initial_tours, load_distance
from pso_tsp import PSO_TSP
from localsearch import local_search_full
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 建立 FontProperties 物件
myfont = FontProperties(fname="NotoSansTC-Regular.otf")

def build_D(csv_path):
    xs, ys = load_distance(csv_path)
    m = len(xs)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            D[i,j] = np.hypot(xs[i]-xs[j], ys[i]-ys[j])
    return D

if __name__ == "__main__":
    CSV = "data/xy48.csv"
    POP = 100
    ITERS = 650

    D = build_D(CSV)
    o=[]
    p=[]
    oi=[]
    pi=[]
    for _ in range(100):
        init_tours = fcm_initial_tours(CSV, k=10, pop_size=POP,a_ratio=0.5,b_ratio=0.5)

        solver = PSO_TSP(D, pop_size=POP, w=.9, c1=1.5, c2=0.85,
                        local_opt=True
                        )
        solver.init_population(init_tours)
        best_len, best_route = solver.run(ITERS, verbose=True)

        # print("\nBest length :", best_len)
        # print("Best route  :", best_route)
        o.append(best_len)
        p.append(best_route)

        # ---- 追加最終 OR-opt local search ----
        refined = local_search_full(best_route[1:-1], D)     # 去掉兩端 0
        refined_len = tour_length(refined, D)
        if refined_len < best_len:
            best_len   = refined_len
            best_route = [0] + refined + [0]
        # --------------------------------------

        print("\nRefined best length :", best_len)
        print("Refined best route  :", best_route)
        oi.append(best_len)
        pi.append(best_route)
print("Best length :", min(oi))



def print_stats(data, label):
    data_np = np.array(data)
    mean = np.mean(data_np)
    std = np.std(data_np)
    q1, median, q3 = np.percentile(data_np, [25, 50, 75])
    lower = np.min(data_np)
    upper = np.max(data_np)
    print(f"\n{label} 統計數據:")
    print(f"平均: {mean:.2f}")
    print(f"標準差: {std:.2f}")
    print(f"四分位數: Q1: {q1:.2f}, 中位數: {median:.2f}, Q3: {q3:.2f}")
    print(f"上下界: {lower:.2f} ~ {upper:.2f}")
    
print_stats(o, "PSO最佳")
print_stats(oi, "Refined最佳")
# 繪製箱型圖
plt.figure(figsize=(10, 5))
plt.boxplot([o, oi], tick_labels=["PSO最佳", "Refined最佳"], showfliers=False)
plt.title("PSO vs Refined 箱型圖 (不顯示離群值)", fontproperties=myfont)
plt.ylabel("路徑長度", fontproperties=myfont)
plt.xticks(fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.show()

plt.figure(figsize=(10, 5))
plt.boxplot([o, oi], tick_labels=["PSO最佳", "Refined最佳"], showfliers=True)
plt.title("PSO vs Refined 箱型圖 (不顯示離群值)", fontproperties=myfont)
plt.ylabel("路徑長度", fontproperties=myfont)
plt.xticks(fontproperties=myfont)
plt.yticks(fontproperties=myfont)
plt.show()
print("Best distance :",o)
print("Best route    :",p)
print("Refined distance :",oi)
print("Refined route    :",pi)