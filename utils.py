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
