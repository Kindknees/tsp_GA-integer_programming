
import numpy as np
from utils import tour_length          # ← 跟主程式同一支長度函式

# localsearch.py

def _or_once(route, k, D):
    n   = len(route)
    # 一開始也把 route 轉成整數 numpy array
    route = np.array(route, dtype=int)
    base  = tour_length(route, D)
    best_gain = 0.0
    best      = None

    for i in range(n - k + 1):
        seg  = route[i : i+k]
        rest = np.concatenate((route[:i], route[i+k:]))
        for j in range(len(rest) + 1):
            cand = np.concatenate((rest[:j], seg, rest[j:]))
            # 確保 cand 是整數
            cand = cand.astype(int)
            gain = base - tour_length(cand, D)
            if gain > best_gain:
                best_gain, best = gain, cand.copy()

    if best is not None:
        return best.tolist(), best_gain
    else:
        # 返回的 route 也最好是 list
        return route.tolist(), 0.0

def local_search_full(route, D):
    # 不要傳 numpy.float64 進來，保證都是純整數 list
    cur = list(map(int, route))
    while True:
        for k in (1, 2, 3):
            nxt, gain = _or_once(cur, k, D)
            if gain > 0:
                cur = nxt   # 已經是 list[int]
                break
        else:
            return cur
