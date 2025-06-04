import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# distance = pd.read_csv('USA_distance.csv', header=None) 
# distance.head()
class Distance():
    def __init__(self, filename):
        data = pd.read_csv(filename, header=None)
        self.x = data[0].values  # 第一欄作為 x 座標
        self.y = data[1].values  # 第二欄作為 y 座標
    def get_distance(self):
        # 計算距離矩陣
        n = len(self.x)
        distance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance[i][j] = np.ceil(np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2))
        # # 四捨五入
        # distance = np.round(distance)
        # 轉換為 DataFrame
        distance = pd.DataFrame(distance)
        distance.index = [f'City {i+1}' for i in range(n)]
        distance.columns = [f'City {i+1}' for i in range(n)]
        return distance
from gurobipy import *

# ---- (保持不變) 讀檔、算距離 --------------------------------------------------
d = Distance('data/xy48.csv')
distance = d.get_distance()
D = distance.to_numpy()
n = D.shape[0]
V = range(n)

# ---- (NEW) STSP 模型 ---------------------------------------------------------
m = Model('STSP_USA')

# 1. 只建 i<j 的邊 (半三角形)
E = [(i, j) for i in V for j in V if i < j]
y = m.addVars(E, vtype=GRB.BINARY, name='y')

# 2. degree = 2  (∑_j y_ij = 2)
m.addConstrs(
    quicksum(y[min(i, j), max(i, j)] for j in V if j != i) == 2
    for i in V)

# 3. 目標：每條邊只算一次
m.setObjective(
    quicksum(D[i, j] * y[i, j] for (i, j) in E),
    GRB.MINIMIZE)

# ---- subtour callback (undirected) ------------------------------------------
def subtour(edges):
    """回傳目前解中最短子迴圈 (list)。"""
    visited = [False] * n
    cycles = []
    for i in V:
        if not visited[i]:
            this_cycle = []
            stack = [i]
            while stack:
                curr = stack.pop()
                if not visited[curr]:
                    visited[curr] = True
                    this_cycle.append(curr)
                    # 走鄰接點（無向邊）
                    neigh = [j for j in V
                             if (min(curr, j), max(curr, j)) in edges
                             and edges[min(curr, j), max(curr, j)] > 0.5]
                    stack.extend(neigh)
            cycles.append(this_cycle)
    return min(cycles, key=len)

def callback(model, where):
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._y)
        edges = {(i, j): vals[i, j] for (i, j) in E}
        C = subtour(edges)
        if len(C) < n:
            # Lazy cut：∑_{i,j∈C} y_ij ≤ |C| − 1
            model.cbLazy(
                quicksum(model._y[min(i, j), max(i, j)]
                         for i in C for j in C if i < j)
                <= len(C) - 1)

m.Params.LazyConstraints = 1
m._y = y

m.optimize(callback)

# ---- 還原巡迴順序 ------------------------------------------------------------
print(f"Optimal tour length: {m.objVal:.2f}")

sol = m.getAttr('x', y)
edges = [(i, j) for (i, j), v in sol.items() if v > 0.5]

tour = [0]
while len(tour) < n:
    last = tour[-1]
    # 找跟 last 相鄰、且還沒走過的城市
    next_city = next(j if last == i else i
                     for (i, j) in edges
                     if last in (i, j) and (j if last == i else i) not in tour)
    tour.append(next_city)
tour.append(0)

print("Tour order:", tour)
