import random
from utils import tour_length


def roulette_select(population,D):
    """
    輪盤選擇：根據 fitness = 1/length 選出一條路徑
    """
    fitness = [1.0 / tour_length(route) for route in population]
    total = sum(fitness)
    pick = random.random() * total
    current = 0.0
    for route, fit in zip(population, fitness):
        current += fit
        if current >= pick:
            return route
    return population[-1]


def edge_recombination(p1, p2):
    """
    邊緣重組交叉 (ERX)：保留父母邊列表的鄰接關係
    """
    size = len(p1)
    # 建立鄰接表
    neighbors = {v: set() for v in p1}
    for parent in (p1, p2):
        for i in range(size):
            v = parent[i]
            left = parent[i - 1]
            right = parent[(i + 1) % size]
            neighbors[v].update({left, right})

    # 開始構建子路徑
    current = p1[0]
    child = [current]
    unused = set(p1)
    unused.remove(current)
    while unused:
        # remove current from all neighbor lists
        for nbrs in neighbors.values():
            nbrs.discard(current)
        # 選下一個
        if neighbors[current]:
            # pick neighbor with fewest remaining neighbors
            next_city = min(neighbors[current], key=lambda x: len(neighbors[x]))
        else:
            next_city = random.choice(list(unused))
        child.append(next_city)
        unused.remove(next_city)
        current = next_city
    return child


def two_opt(route):
    """
    基本 2-opt 鄰域搜尋：嘗試交換任意兩條邊，直到無改進
    """
    best = route[:]
    best_cost = tour_length(best)
    size = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(size - 1):
            for j in range(i + 2, size):
                # skip if edges share a node in circular tour
                if i == 0 and j == size - 1:
                    continue
                new_route = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                cost = tour_length(new_route)
                if cost < best_cost:
                    best, best_cost = new_route, cost
                    improved = True
                    break
            if improved:
                break
    return best


def mutate(route):
    """
    六種突變算子之一 (隨機選擇):
    1. DM 段移動
    2. ISM 單點插入
    3. IVM 段反轉插入
    4. EM 兩點交換
    5. 隨機 2-opt
    6. 鄰近吸引 (交換至相鄰)
    """
    r = random.random()
    n = len(route)
    new_route = route[:]
    if r < 1/6:
        # DM: segment move
        i, j = sorted(random.sample(range(n), 2))
        seg = new_route[i:j+1]
        rem = new_route[:i] + new_route[j+1:]
        k = random.randrange(len(rem) + 1)
        new_route = rem[:k] + seg + rem[k:]
    elif r < 2/6:
        # ISM: single insertion
        i = random.randrange(n)
        city = new_route.pop(i)
        k = random.randrange(len(new_route) + 1)
        new_route.insert(k, city)
    elif r < 3/6:
        # IVM: segment reverse + insert
        i, j = sorted(random.sample(range(n), 2))
        seg = new_route[i:j+1][::-1]
        rem = new_route[:i] + new_route[j+1:]
        k = random.randrange(len(rem) + 1)
        new_route = rem[:k] + seg + rem[k:]
    elif r < 4/6:
        # EM: swap two points
        i, j = random.sample(range(n), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
    elif r < 5/6:
        # random 2-opt mutation
        i, j = sorted(random.sample(range(n), 2))
        if j - i > 1:
            new_route[i+1:j+1] = reversed(new_route[i+1:j+1])
    else:
        # neighbor attraction: swap with nearest neighbor in tour
        i = random.randrange(n)
        j = (i+1) % n
        new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def two_opt2 (route, D):
    """對路徑應用2-opt鄰域優化，返回可能改進後的路徑"""
    n = len(route)
    improved = True
    while improved:  # 持續嘗試2-opt直到無改善
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # 修改：考慮閉合路徑的情況，跳過首尾相鄰邊組合，以免移除兩條共用節點的邊
                if i == 0 and j == n - 1:
                    continue  # 略過移除 route[-1]->route[0] 和 route[0]->route[1] 這種相鄰邊情況
                # 定義當前考慮移除的兩條邊：
                a, b = route[i], route[i+1]
                c, d = route[j], route[(j+1) % n]  # (j+1)%n 確保考慮到末節點與首節點的邊
                # 計算移除前後的距離變化
                old_dist = D[a][b] + D[c][d]
                new_dist = D[a][c] + D[b][d]
                if new_dist < old_dist:
                    # 如有改善，執行2-opt交換：反轉 route[i+1...j] 段
                    route[i+1:j+1] = route[i+1:j+1][::-1]
                    improved = True
                    break  # 執行一次改善後跳出內層迴圈，重新開始掃描
            if improved:
                break
    return route


