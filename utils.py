# utils.py
import numpy as np
import random

def tour_length(tour, D):
    """tour = list[int] (city1…city47)；回傳完整 (0-…-0) 距離"""
    n = len(tour)
    dist = D[0, tour[0]] + D[tour[-1], 0]
    for i in range(n-1):
        dist += D[tour[i], tour[i+1]]
    return dist

def two_opt(tour, D):
    """簡易兩段翻轉；貪婪只做一次改進"""
    best = tour
    best_len = tour_length(tour, D)
    n = len(tour)
    for i in range(n-1):
        for j in range(i+2, n):
            if j == n-1 and i == 0:
                continue
            new = tour.copy()
            new[i:j+1] = new[i:j+1][::-1]
            new_len = tour_length(new, D)
            if new_len < best_len:
                return new
    return best

def exchange_move(tour, D, remove_count, insert_count):
    n = len(tour)
    best, best_len = tour, tour_length(tour, D)
    # 隨機選 remove_count 個位置，抽出子序列，插到任意 insert 位置
    for _ in range(10):  # 測試幾次擾動
        idxs = sorted(random.sample(range(n), remove_count))
        segment = [tour[i] for i in idxs]
        rest = [tour[i] for i in range(n) if i not in idxs]
        # 再插入 segment 但每次只插入 insert_count 個
        for ins in range(len(rest)+1):
            cand = rest[:ins] + segment[:insert_count] + rest[ins:]
            cand += segment[insert_count:]  # 多出部分放到尾端
            length = tour_length(cand, D)
            if length < best_len:
                best, best_len = cand, length
    return best

def local_search_mix(tour, D):
    """隨機選一種 move 套一次，若改進就返回新路徑"""
    ops = [
      lambda t: two_opt(t, D),
    #   lambda t: three_opt(t, D),
      lambda t: exchange_move(t, D, 1, 1),
      lambda t: exchange_move(t, D, 1, 2),
      lambda t: exchange_move(t, D, 2, 2),
    ]
    op = random.choice(ops)
    return op(tour)

def three_opt(route, D):
    """Full 3-opt until no improving move (first-improvement)."""
    route = route.copy()
    n = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            a, b = route[i], route[(i + 1) % n]
            for j in range(i + 2, n):
                c, d = route[j], route[(j + 1) % n]
                if j == i:  # 同邊，略
                    continue
                for k in range(j + 2, n + (i > 0)):  # 允許繞回起點
                    k_mod = k % n
                    e, f = route[k_mod], route[(k_mod + 1) % n]
                    if k_mod in (i, j):   # 三段重疊，略
                        continue

                    # 舊邊長
                    old = D[a][b] + D[c][d] + D[e][f]

                    # 7 個候選重接（依圖列舉）
                    cand = []
                    # 1) a-c, b-e, d-f
                    cand.append(D[a][c] + D[b][e] + D[d][f])
                    # 2) a-c, b-d, e-f
                    cand.append(D[a][c] + D[b][d] + D[e][f])
                    # 3) a-d, e-b, c-f
                    cand.append(D[a][d] + D[e][b] + D[c][f])
                    # 4) a-d, e-c, b-f
                    cand.append(D[a][d] + D[e][c] + D[b][f])
                    # 5) a-e, d-b, c-f
                    cand.append(D[a][e] + D[d][b] + D[c][f])
                    # 6) a-e, d-c, b-f
                    cand.append(D[a][e] + D[d][c] + D[b][f])
                    # 7) a-f, e-c, b-d
                    cand.append(D[a][f] + D[e][c] + D[b][d])

                    best_new = min(cand)
                    gain = best_new - old
                    if gain < 0:      # 找到改善
                        m = cand.index(best_new) + 1  # move 編號 1-7
                        route = apply_3opt_move(route, i, j, k_mod, move=m)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return route  # 整輪沒改進


def apply_3opt_move(r, i, j, k, move):
    """依 move 編號把 r (list) 三段重接後回傳新 list."""
    a = r[:i+1]
    b = r[i+1:j+1]
    c = r[j+1:k+1]
    d = r[k+1:]
    if move == 1:  # a-c, b-e, d-f (∼ b, c 反轉)
        b.reverse()
        return a + c + b + d
    elif move == 2:  # a-c, b-d, e-f
        return a + c[::-1] + b + d
    elif move == 3:  # a-d, e-b, c-f
        return a + b[::-1] + c + d
    elif move == 4:  # a-d, e-c, b-f
        return a + b[::-1] + c[::-1] + d
    elif move == 5:  # a-e, d-b, c-f
        return a + c + b[::-1] + d
    elif move == 6:  # a-e, d-c, b-f
        return a + c[::-1] + b[::-1] + d
    elif move == 7:  # a-f, e-c, b-d
        return a + b + c[::-1] + d
    else:  # 不應進來
        return r
