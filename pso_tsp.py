# pso_tsp.py
import numpy as np
from utils import tour_length, local_search_mix

class PSO_TSP:
    """Permutation PSO 版本：Vel.=swap 序列；Pos.=tour"""
    def __init__(self, D, pop_size=100, w=0.8, c1=1.5, c2=1,
                 local_opt=True, seed=None):
        self.D, self.n = D, D.shape[0] - 1            # n = 47 (不含 city0)
        self.rng = np.random.default_rng(seed)
        self.pop_size, self.w, self.c1, self.c2 = pop_size, w, c1, c2
        self.local_opt = local_opt                    # 是否 two-opt 微調
        self.stall_count = 0    #收斂檢查
        self.stall_limit = 50

        # --- 粒子狀態 -------------------------------------------------------
        self.X = None      # 位置：array(pop, n)    (city1…city47)
        self.V = None      # 速度：list of swap-list  ([(i,j), …])
        self.pbest = None  # 每粒子歷史最優
        self.pbest_len = None
        self.gbest = None  # 全域最優
        self.gbest_len = np.inf

    # ---------- 初始化 -------------------------------------------------------
    def init_population(self, init_tours):
        """init_tours: list[ list[int] ] ，⾧度 = pop_size"""
        self.X = np.array(init_tours, dtype=int)
        self.V = [ [] for _ in range(self.pop_size) ]

        self.pbest = self.X.copy()
        self.pbest_len = np.array([tour_length(t, self.D) for t in self.X])

        best_idx = np.argmin(self.pbest_len)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_len = self.pbest_len[best_idx]

    # ---------- 速度 & 位置更新 ----------------------------------------------
    def _subtract(self, a, b):
        """取得把序列 b 變成 a 的 swap 序列 (=‘差’)"""
        swaps = []
        b = list(b)
        for i in range(self.n):
            if b[i] != a[i]:
                j = b.index(a[i])
                swaps.append((i, j))
                b[i], b[j] = b[j], b[i]
        return swaps

    def _apply_swaps(self, x, swaps):
        x = x.copy()
        for i, j in swaps:
            x[i], x[j] = x[j], x[i]
        return x

    def step(self):
        
        for k in range(self.pop_size):
            # --- 更新速度 ----------------------------------------------------
            cognitive = self._subtract(self.pbest[k], self.X[k])
            social    = self._subtract(self.gbest,    self.X[k])

            # 隨機擷取部份 swap 作為貢獻
            def rand_subset(swaps, ratio):
                self.rng.shuffle(swaps)
                keep = int(np.ceil(ratio * len(swaps)))
                return swaps[:keep]

            v_new =  rand_subset(self.V[k],           self.w)
            v_new += rand_subset(cognitive, self.rng.random()*self.c1)
            v_new += rand_subset(social,    self.rng.random()*self.c2)
            self.V[k] = v_new

            # --- 更新位置 ----------------------------------------------------
            self.X[k] = self._apply_swaps(self.X[k], v_new)

            # 局部 two-opt 微調（可關閉）
            if self.local_opt and self.rng.random() < .2:
                self.X[k] = local_search_mix(self.X[k], self.D)


            # --- 更新個體／全域最好 ----------------------------------------
            f = tour_length(self.X[k], self.D)
            if f < self.pbest_len[k]:
                self.pbest[k] = self.X[k].copy()
                self.pbest_len[k] = f
            if f < self.gbest_len:
                self.gbest = self.X[k].copy()
                self.gbest_len = f

    # ---------- 主迴圈 -------------------------------------------------------
    def run(self, iters=1000, verbose=False):
        for gen in range(iters):
            self.step()
            if verbose and gen % 100 == 0:
                print(f"gen {gen:4d} | best = {self.gbest_len:.2f}")
        return self.gbest_len, [0]+self.gbest.tolist()+[0]
