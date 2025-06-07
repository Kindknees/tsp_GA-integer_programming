import matplotlib.pyplot as plt
import sys

def read_coordinates_from_file(filename):
    coords = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            coords.append((float(parts[0]), float(parts[1])))
    return coords

def read_tour_indices_from_file(filename):
    tour_indices = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            tour_indices.append(int(line))

    return tour_indices

def plot_tsp_tour(tour_indices_filename, coords_filename):
    coords = read_coordinates_from_file(coords_filename)
    tour_indices = read_tour_indices_from_file(tour_indices_filename)
    # 提取繪圖所需的X, Y座標 (所有城市)
    x_coords_all = [c[0] for c in coords]
    y_coords_all = [c[1] for c in coords]

    # 根據路徑順序提取座標
    tour_x = [coords[i][0] for i in tour_indices]
    tour_y = [coords[i][1] for i in tour_indices]

    if tour_indices:
        tour_x.append(coords[tour_indices[0]][0])
        tour_y.append(coords[tour_indices[0]][1])

    plt.figure(figsize=(10, 8))

    # 繪製所有城市點
    plt.scatter(x_coords_all, y_coords_all, c='blue', label='city', s=50, zorder=2)

    # 繪製TSP路徑
    if tour_x and tour_y:
        plt.plot(tour_x, tour_y, 'r-', label='TSP route', zorder=1)

    # 標記起點
    if tour_indices and coords:
        start_node_index = tour_indices[0]
        plt.scatter(coords[start_node_index][0], coords[start_node_index][1], c='green', s=100, marker='*', label='start', zorder=3)

    # 添加城市編號
    # for i, (x, y) in enumerate(coords):
    #     plt.text(x + 0.05 * (max(x_coords_all) - min(x_coords_all)),
    #              y + 0.05 * (max(y_coords_all) - min(y_coords_all)), 
    #              str(i), fontsize=9)
    for i, (x, y) in enumerate(coords):
        plt.text(x + 0.13,
                 y + 0.13, 
                 str(i), fontsize=9)

    plt.title('TSP solution)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    
    tour_file = sys.argv[1]
    coords_file = sys.argv[2]
    plot_tsp_tour(tour_file, coords_file)