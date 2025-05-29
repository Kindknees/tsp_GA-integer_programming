# iam_tsp48.py ───────────────────────────────────────────
import pandas as pd
from math import hypot
from itertools import permutations
from typing import List, Tuple

City  = Tuple[float, float]
Route = List[int]


# ────── 工具函式 ────────────────────────────────────────
def euclidean(p: City, q: City) -> float:
    return hypot(p[0] - q[0], p[1] - q[1])


def tour_cost(route: Route, coords: List[City]) -> float:
    return sum(
        euclidean(coords[route[i]], coords[route[(i + 1) % len(route)]])
        for i in range(len(route))
    )


def best_insertion(route: Route, city: int, coords: List[City]) -> Route:
    """將 `city` 以最小增量插入 route，回傳 *新* 路線"""
    best_route, best_delta = None, float("inf")
    m = len(route)
    for i in range(m):  # 夾在 route[i-1] 與 route[i] 之間
        a, b = route[i - 1], route[i]
        delta = (
            euclidean(coords[a], coords[city])
            + euclidean(coords[city], coords[b])
            - euclidean(coords[a], coords[b])
        )
        if delta < best_delta:
            best_delta = delta
            best_route = route[:i] + [city] + route[i:]
    return best_route


# ────── Algorithm 1 ︱ IAM-TSP ──────────────────────────
def iam_tsp(coords: List[City]) -> Route:
    n = len(coords)
    xs, ys = zip(*coords)

    east  = max(range(n), key=lambda i: coords[i][0])
    west  = min(range(n), key=lambda i: coords[i][0])
    north = max(range(n), key=lambda i: coords[i][1])
    south = min(range(n), key=lambda i: coords[i][1])

    route: Route = [east, north, west, south]
    visited      = [False] * n
    for c in route:
        visited[c] = True

    open_set = [i for i in range(n) if not visited[i]]

    while open_set:
        # 試著把 open_set 中每個城市插進 route；保留最省的那一個
        best_cost, best_city, best_route = float("inf"), None, None
        for c in open_set:
            candidate = best_insertion(route, c, coords)
            c_cost    = tour_cost(candidate, coords)
            if c_cost < best_cost:
                best_cost, best_city, best_route = c_cost, c, candidate
        route    = best_route
        open_set.remove(best_city)

    return route


# ────── Algorithm 2 ︱ IAM-TSP+ ─────────────────────────
def iam_tsp_plus(coords: List[City], k: int = 5) -> Route:
    """對 IAM-TSP 路線做長度 k 視窗的 k! 全排列改良"""
    route = iam_tsp(coords)
    n     = len(route)
    if k < 2 or k > n:
        return route

    for i in range(n - k):
        window    = route[i : i + k]
        best_cost = tour_cost(route, coords)
        best_route = route
        for perm in permutations(window):
            cand = route[:i] + list(perm) + route[i + k :]
            c_cost = tour_cost(cand, coords)
            if c_cost < best_cost:
                best_cost, best_route = c_cost, cand
        route = best_route

    return route


# ────── 主程式 ─────────────────────────────────────────
if __name__ == "__main__":
    # 讀取你的座標檔
    df = pd.read_csv(r"data/xy48.csv", header=None, names=["x", "y"])
    coords: List[City] = list(df.itertuples(index=False, name=None))
    print(f"共載入 {len(coords)} 座城市\n")

    # IAM-TSP
    r0 = iam_tsp(coords)
    c0 = tour_cost(r0, coords)
    print(f"IAM-TSP  路徑長度 : {c0:,.2f}")

    # IAM-TSP+  (k = 5)
    r1 = iam_tsp_plus(coords, k=5)
    c1 = tour_cost(r1, coords)
    print(f"IAM-TSP+ 路徑長度 : {c1:,.2f}   (k = 5)")

    # 如果想看完整順序，把下面兩行解除註解
    # print("\nIAM-TSP  路線 :", r0)
    # print("IAM-TSP+ 路線 :", r1)
