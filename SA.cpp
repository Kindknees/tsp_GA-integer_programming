#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <ctime>
#include <cstdlib>

using namespace std;

const int NUM_CITIES = 48;

struct TSPSolution {
    int tour[NUM_CITIES];
    double cost;
};

// 讀取距離檔案
void read_distance_matrix(const string& filename, int dist_matrix[NUM_CITIES][NUM_CITIES]) {
    ifstream file(filename);
    string line_str;
    for (int i = 0; i < NUM_CITIES; ++i) {
        getline(file, line_str);
        istringstream iss(line_str);
        for (int j = 0; j < NUM_CITIES; ++j) 
        {
            iss >> dist_matrix[i][j];
        }
    }
    file.close();
}

// 計算總距離
double calculate_tour_cost(const int tour[NUM_CITIES], const int dist_matrix[NUM_CITIES][NUM_CITIES]) {
    double current_total_cost = 0.0;
    for (int i = 0; i < NUM_CITIES - 1; ++i) {
        current_total_cost += dist_matrix[tour[i]][tour[i + 1]];
    }
    current_total_cost += dist_matrix[tour[NUM_CITIES - 1]][tour[0]];
    return current_total_cost;
}

// 生成初始解 
void generate_initial_solution(int tour[NUM_CITIES], mt19937& rng_engine) {
    for (int i = 0; i < NUM_CITIES; i++) {
        tour[i] = i;
    }
    for (int i = NUM_CITIES - 1; i > 0; i--) {
        uniform_int_distribution<int> dist(0, i);
        int j = dist(rng_engine);
        swap(tour[i], tour[j]);
    }
}

// 2-opt
void generate_2opt(const int current_tour[NUM_CITIES], int neighbor_tour[NUM_CITIES], mt19937& rng_engine) {
    for (int k = 0; k < NUM_CITIES; k++) {
        neighbor_tour[k] = current_tour[k];
    }
    if (NUM_CITIES < 4) return;
    uniform_int_distribution<int> dist_indices(0, NUM_CITIES - 1);
    int i = dist_indices(rng_engine);
    int j = dist_indices(rng_engine);
    while (i == j || (i + 1) % NUM_CITIES == j || (j + 1) % NUM_CITIES == i) {
        i = dist_indices(rng_engine);
        j = dist_indices(rng_engine);
    }
    if (i > j) swap(i, j);
    int start_reverse_idx = i + 1;
    int end_reverse_idx = j;
    while (start_reverse_idx < end_reverse_idx) {
        swap(neighbor_tour[start_reverse_idx], neighbor_tour[end_reverse_idx]);
        start_reverse_idx++;
        end_reverse_idx--;
    }
}

// 模擬退火
TSPSolution simulated_annealing(const int dist_matrix[NUM_CITIES][NUM_CITIES],
                                double initial_temperature,
                                double final_temp,
                                double cooling_rate,
                                int iterations_per_temp,
                                unsigned int seed) {
    mt19937 rng_engine(seed);
    TSPSolution current_s;
    generate_initial_solution(current_s.tour, rng_engine);
    current_s.cost = calculate_tour_cost(current_s.tour, dist_matrix);
    TSPSolution best_s = current_s;
    double temp = initial_temperature;
    long long total_iterations = 0;
    int neighbor_tour_buffer[NUM_CITIES];
    while (temp > final_temp) {
        for (int iter = 0; iter < iterations_per_temp; iter++) {
            generate_2opt(current_s.tour, neighbor_tour_buffer, rng_engine);
            double neighbor_cost = calculate_tour_cost(neighbor_tour_buffer, dist_matrix);
            double cost_delta = neighbor_cost - current_s.cost;
            if (cost_delta < 0) {
                for (int k = 0; k < NUM_CITIES; k++) current_s.tour[k] = neighbor_tour_buffer[k];
                current_s.cost = neighbor_cost;
                if (current_s.cost < best_s.cost) best_s = current_s;
            } else {
                uniform_real_distribution<double> prob_dist(0.0, 1.0);
                if (prob_dist(rng_engine) < exp(-cost_delta / temp)) {
                    for (int k = 0; k < NUM_CITIES; ++k) current_s.tour[k] = neighbor_tour_buffer[k];
                    current_s.cost = neighbor_cost;
                }
            }
            total_iterations++;
        }
        temp *= cooling_rate;
    }
    cout << "total iterations: " << total_iterations << endl;
    return best_s;
}

// 寫入最佳路徑到文件，讓python讀取
void write_tour_indices_for_python(const string& output_filename,
                                   const int best_tour[NUM_CITIES]) {
    ofstream outfile(output_filename);

    for (int i = 0; i < NUM_CITIES; ++i) {
        outfile << best_tour[i] << endl;
    }
    outfile.close();
    cout << "best route file: " << output_filename << endl;
}


int main() {
    string dist_filename = "att48_d.txt";
    string coord_filename = "att48_xy.txt"; 
    string python_tour_data_filename = "tsp_tour_indices.txt"; 
    string python_script_name = "SA_plot_tsp.py"; 

    int city_distances[NUM_CITIES][NUM_CITIES];

    srand(time(NULL));
    read_distance_matrix(dist_filename, city_distances);
    
    // 模擬退火參數
    double initial_temperature = 10000.0;
    double final_temperature = 0.01;
    double cooling_factor = 0.997;
    int iterations_at_each_temp = 100;
    unsigned int random_s = rand();

    cout << "initial temperature: " << initial_temperature << endl;
    cout << "end temperature: " << final_temperature << endl;
    cout << "cooling factor: " << cooling_factor << endl;
    cout << "iterations per temperature: " << iterations_at_each_temp << endl;
    cout << "random seed: " << random_s << endl;

    clock_t t_start = clock();
    TSPSolution final_solution = simulated_annealing(city_distances,
                                                    initial_temperature,
                                                    final_temperature,
                                                    cooling_factor,
                                                    iterations_at_each_temp,
                                                    random_s);
    clock_t t_end = clock();
    double time_taken_sec = (double)(t_end - t_start) / CLOCKS_PER_SEC;

    cout << "cost: " << fixed << setprecision(2) << final_solution.cost << endl;
    cout << "route: ";
    for (int i = 0; i < NUM_CITIES; ++i) {
        cout << final_solution.tour[i] << (i == NUM_CITIES - 1 ? "" : " -> ");
    }
    cout << " -> " << final_solution.tour[0] << endl;
    cout << "time: " << fixed << setprecision(5) << time_taken_sec << " s" << endl;

    // 將路徑索引寫入文件供Python使用
    write_tour_indices_for_python(python_tour_data_filename, final_solution.tour);

    string command = "python " + python_script_name + " " + python_tour_data_filename + " " + coord_filename;
    int result = system(command.c_str());

    return 0;
}