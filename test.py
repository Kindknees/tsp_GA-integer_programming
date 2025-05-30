import numpy as np
import random
from copy import deepcopy
import time

# 讀取距離矩陣
def read_distance_matrix(filename):
    with open(filename, 'r') as file:
        dist = [list(map(int, line.split())) for line in file]
    return np.array(dist)

# 計算路徑總距離
def calculate_distance(path, dist_matrix):
    distance = 0
    for i in range(len(path)):
        distance += dist_matrix[path[i], path[(i + 1) % len(path)]]
    return distance

def nearest_neighbor(dist_matrix, start):
    num_cities = len(dist_matrix)
    unvisited = list(range(num_cities))
    path = [start]
    unvisited.remove(start)
    
    while unvisited:
        last_city = path[-1]
        next_city = min(unvisited, key=lambda city: dist_matrix[last_city][city])
        path.append(next_city)
        unvisited.remove(next_city)
    return path

def initialize_particles(num_particles, num_cities, dist_matrix, nn):
    particles = []
    velocities = []
    
    # 速度定義為一個 list of tuples: (probability_of_swap, city_index1, city_index2)
    # probability_of_swap: 在更新位置時，實際執行此交換的機率
    
    # 根據是否使用最近鄰，決定速度向量的長度 (交換操作的數量)
    # nn=True時，前1/4的粒子使用較短的速度向量長度，因為它們的初始解品質較高，可能不需要太多隨機擾動
    default_velocity_len = num_cities // 2
    nn_velocity_len = num_cities // 4

    used_start_cities_for_nn = [] # 避免重複的NN起始點

    for i in range(num_particles):
        current_velocity_len = default_velocity_len
        if nn:
            if i < num_particles // 4: # 前 1/4 粒子使用最近鄰
                start_city = random.randint(0, num_cities - 1)
                while start_city in used_start_cities_for_nn: # 確保起始城市不重複
                    start_city = random.randint(0, num_cities - 1)
                used_start_cities_for_nn.append(start_city)
                particle = nearest_neighbor(dist_matrix, start_city)
                current_velocity_len = nn_velocity_len # NN初始化的粒子用較短速度
            else: # 其餘隨機初始化
                particle = list(np.random.permutation(num_cities))
        else: # 不使用最近鄰，全部隨機初始化
            particle = list(np.random.permutation(num_cities))
        
        particles.append(particle)
        # 初始化速度：每個交換操作包含 (執行機率, 城市索引1, 城市索引2)
        velocity = [(random.random(), 
                     random.randint(0, num_cities - 1), 
                     random.randint(0, num_cities - 1)) for _ in range(current_velocity_len)]
        velocities.append(velocity)
        
    return particles, velocities


# 更新粒子位置
def update_position(particle, velocity): # num_cities is not needed here
    new_particle = particle.copy()
    for prob, idx1, idx2 in velocity:
        if random.random() < prob:
            # 執行交換操作
            new_particle[idx1], new_particle[idx2] = new_particle[idx2], new_particle[idx1]
    return new_particle

# 獲取將 path1 轉換為 path2 所需的交換序列
def get_swap_sequence(path1, path2):
    """
    Calculates a sequence of swaps to transform path1 into path2.
    Returns a list of (idx1, idx2) tuples representing swaps.
    """
    p1_copy = list(path1)
    swaps = []
    num_cities = len(p1_copy)
    
    # 建立一個映射，方便快速查找 p1_copy 中城市的位置
    city_to_index_map = {city: i for i, city in enumerate(p1_copy)}

    for i in range(num_cities):
        if p1_copy[i] != path2[i]:
            # city_to_move 是 path2[i]，它目前在 p1_copy 的什麼位置？
            city_in_p1_at_i = p1_copy[i] # 當前在位置i的城市
            target_city_for_i = path2[i] # 應該在位置i的城市
            
            # 找到 target_city_for_i 在 p1_copy 中的當前位置
            current_pos_of_target_city = city_to_index_map[target_city_for_i]

            # 交換 p1_copy[i] 和 p1_copy[current_pos_of_target_city]
            p1_copy[i], p1_copy[current_pos_of_target_city] = p1_copy[current_pos_of_target_city], p1_copy[i]
            
            # 更新 city_to_index_map
            city_to_index_map[city_in_p1_at_i] = current_pos_of_target_city
            city_to_index_map[target_city_for_i] = i
            
            # 記錄這個交換操作 (作用於原始索引)
            swaps.append((i, current_pos_of_target_city))
            
    return swaps


# PSO 主程式
def pso_tsp(dist_matrix, num_particles=50, max_iter=1000, w_max=0.9, w_min=0.4, c1_max=0.8, c1_min=0.2, c2_max=0.8, c2_min=0.2, nn=False):
    num_cities = len(dist_matrix)
    
    particles, velocities = initialize_particles(num_particles, num_cities, dist_matrix, nn)
    pbest_positions = deepcopy(particles)
    pbest_fitness = [calculate_distance(p, dist_matrix) for p in particles]
    
    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = deepcopy(pbest_positions[gbest_idx]) # Use deepcopy for gbest_position
    gbest_fitness = pbest_fitness[gbest_idx]
    
    # 初始化速度向量長度
    default_velocity_len = num_cities // 2

    for iteration in range(max_iter):
        # 動態調整 w, c1, c2
        # w = w_max - (w_max - w_min) * (iteration / max_iter) # 線性遞減慣性
        w = w_min + (w_max - w_min) * (1 - iteration / max_iter) ** 2 # 非線性遞減慣性
        c1 = c1_min + (c1_max - c1_min) * (1 - iteration / max_iter) ** 2 # 認知學習因子隨迭代次數增加
        c2 = c2_min + (c2_max - c2_min) * (iteration / max_iter) ** 2 # 社會學習因子隨迭代次數減少
        # c1 = c1_max - (c1_max - c1_min) * iteration / max_iter
        # c2 = c2_min + (c2_max - c2_min) * iteration / max_iter

        for i in range(num_particles):
            current_particle_pos = particles[i]
            current_velocity = velocities[i] # List of (prob, idx1, idx2)
            particle_pbest_pos = pbest_positions[i]

            # 決定此粒子的速度向量長度
            current_velocity_len = default_velocity_len
            
            new_velocity_choice = [] # 儲存候選的交換操作-> [idx1, idx2]

            # 慣性
            # 從舊速度中按 w 機率保留一些操作
            for _, op_idx1, op_idx2 in current_velocity:
                if random.random() < w: # w 是保留舊速度分量的機率
                    new_velocity_choice.append((op_idx1, op_idx2))
            
            # pbest
            # 按 c1 機率決定是否從 "pbest - current_pos" 中學習
            if random.random() < c1:
                swaps_to_pbest = get_swap_sequence(current_particle_pos, particle_pbest_pos)
                if swaps_to_pbest: # 如果當前解不是pbest，才會有交換操作
                    # 從指導性交換中隨機選取一個或多個加入候選
                    # 這裡簡單選取一個，也可以選取多個，或按比例選取
                    chosen_swap = random.choice(swaps_to_pbest)
                    new_velocity_choice.append(chosen_swap)

            # 3. gbest
            # 按 c2 機率決定是否從 "gbest - current_pos" 中學習
            if random.random() < c2:
                swaps_to_gbest = get_swap_sequence(current_particle_pos, gbest_position)
                if swaps_to_gbest: # 如果當前解不是gbest
                    chosen_swap = random.choice(swaps_to_gbest)
                    new_velocity_choice.append(chosen_swap)
            
            final_new_velocity_with_prob = []
            if not new_velocity_choice: # 如果沒有任何具指導性的候選操作 (w, c1, c2都很小或解已接近)
                # 全部用隨機交換操作填充
                for _ in range(current_velocity_len):
                    idx1, idx2 = random.sample(range(num_cities), 2)
                    final_new_velocity_with_prob.append((random.random(), idx1, idx2))
            else:
                # 從候選操作中隨機選取 (有放回)，填滿速度向量
                for _ in range(current_velocity_len):
                    op_idx1, op_idx2 = random.choice(new_velocity_choice)
                    # 每個選中的操作賦予一個新的隨機執行機率
                    final_new_velocity_with_prob.append((random.random(), op_idx1, op_idx2))
            
            velocities[i] = final_new_velocity_with_prob
            
            # 得到新的velocities後，便更新位置
            particles[i] = update_position(current_particle_pos, velocities[i])
            
            # 計算新適應度
            fitness = calculate_distance(particles[i], dist_matrix)
            
            # 更新 pbest
            if fitness < pbest_fitness[i]:
                pbest_positions[i] = deepcopy(particles[i])
                pbest_fitness[i] = fitness
            
            # 更新 gbest
            if fitness < gbest_fitness:
                gbest_position = deepcopy(particles[i])
                gbest_fitness = fitness
        
        if iteration % 1000 == 0 or iteration == max_iter - 1:
            print(f"Iteration {iteration}, Best Distance: {gbest_fitness}")
    
    return gbest_position, gbest_fitness

# 主程式
if __name__ == "__main__":
    
    dist_matrix = read_distance_matrix("att48_d.txt") # 請確保此檔案存在
    params = {
        "dist_matrix" : dist_matrix,
        "num_particles" : 150,  # 粒子數量
        "max_iter" : 25000, # 迭代次數
        "w_max" : 0.9,  # 慣性權重
        "w_min" : 0.15, 
        "c1_max" : 0.9,  # pbest學習因子
        "c1_min" : 0.3,
        "c2_max" : 0.85,    # gbest學習因子
        "c2_min" : 0.1,
        "nn" : False
    }
    
    current_time = time.time()
    best_path, best_distance = pso_tsp(**params)
    end_time = time.time()
    print("Execution Time:", end_time - current_time, " seconds")
    print("\nBest Path Found:")
    # 輸出路徑時，城市編號從1開始
    path_str = " -> ".join(map(lambda x: str(x + 1), best_path))
    print(path_str)
    print("Best Distance:", best_distance)