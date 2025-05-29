# tabu.py －－ Variable-Neighborhood Tabu Search
import numpy as np
from utils import tour_length

# ===== 鄰域操作 =====
def relocate(route, i, j, k):
    """將 [i, i+k) 段落搬到 j 之前；route 不含首尾 0"""
    seg = route[i:i+k]
    rest = route[:i] + route[i+k:]
    insert_at = j if j < i else j - k  # 刪除後 index 會左移
    new_route = rest[:insert_at] + seg + rest[insert_at:]
    return new_route

def double_bridge(route):
    """經典 4-edge ‘kick’（route 不含首尾 0）"""
    n = len(route)
    a, b, c = sorted(np.random.choice(range(1, n-1), 3, replace=False))
    p1, p2, p3, p4 = route[:a], route[a:b], route[b:c], route[c:]
    return p1 + p3 + p2 + p4   # p2、p3 互換

# ===== 主程式 =====
def tabu_search_vns(route, D,
                    max_iter=600, tenure=25,
                    k_max=3, kick_interval=120, seed=None):
    """
    Variable-Neighborhood Tabu Search:
      - relocate k=1..k_max  (Or-opt)
      - 適時 3-opt / double-bridge kick
    """
    rng = np.random.default_rng(seed)
    n = len(route)
    best_route = route[:]
    best_len   = tour_length(best_route, D)

    cur_route  = route[:]
    cur_len    = best_len
    n_city = max(route) + 1        # 48
    tabu_age = np.zeros((n_city, n_city), dtype=int)   # ← 取代原本的 (n, n)

    t_global   = 0

    while t_global < max_iter:
        t_global += 1
        best_move  = None
        best_delta = np.inf

        # ===== ① Or-opt / relocate k=1..k_max =====
        for k in range(1, k_max+1):
            for i in range(n-k+1):
                for j in range(n-k+1):
                    if j == i or j == i+k:       # 位置沒變
                        continue
                    # 檢查被刪邊 (a,i) (i+k-1,b) 是否 tabu
                    a = cur_route[i-1] if i > 0 else cur_route[-1]
                    b = cur_route[(i+k) % n]
                    if tabu_age[a, cur_route[i]] > t_global or \
                       tabu_age[cur_route[i+k-1], b] > t_global:
                        continue
                    new_route = relocate(cur_route, i, j, k)
                    new_len   = tour_length(new_route, D)
                    delta     = new_len - cur_len
                    if new_len < best_len:   # Aspiration
                        best_move, best_delta = (new_route, a, cur_route[i], cur_route[i+k-1], b), delta
                        break
                    if delta < best_delta:
                        best_move, best_delta = (new_route, a, cur_route[i], cur_route[i+k-1], b), delta
            if best_move and best_move[0] != cur_route:
                break  # 找到改善或最小Δ即跳出 k 迴圈

        # ===== ② 若 relocate 無進步 → 嘗試 double-bridge kick =====
        if best_move is None or best_delta >= 0:
            new_route = double_bridge(cur_route)
            new_len   = tour_length(new_route, D)
            best_move = (new_route, None, None, None, None)
            best_delta = new_len - cur_len

        # ===== ③ 執行 move =====
        cur_route = best_move[0]
        cur_len   = cur_len + best_delta

        # 更新 tabu（只記被刪的兩條邊）
        if best_move[1] is not None:
            a, u, v, b = best_move[1:]
            tabu_age[a, u] = t_global + tenure
            tabu_age[v, b] = t_global + tenure

        # ===== ④ 更新全域最佳 =====
        if cur_len < best_len:
            best_len, best_route = cur_len, cur_route[:]

        # ===== ⑤ 定期 kick 以多樣化 =====
        if kick_interval and t_global % kick_interval == 0:
            cur_route = double_bridge(best_route)
            cur_len   = tour_length(cur_route, D)

    return best_len, best_route
